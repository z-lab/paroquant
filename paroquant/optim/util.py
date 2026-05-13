from __future__ import annotations

import gc
import logging
import math
import random
import warnings
from typing import TypeVar

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


RETAINED_KWARG_KEYS = ("shared_kv_states",)


def to_device(value, device: torch.device | str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def get_blocks(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model = model.model.language_model
    elif hasattr(model, "model"):
        model = model.model

    if hasattr(model, "layers"):
        return model.layers

    raise NotImplementedError(type(model))


_Linear_T = TypeVar("Linear", bound=nn.Module)


def get_named_linears(module: nn.Module, subclass: type[_Linear_T] = nn.Linear) -> dict[str, _Linear_T]:
    return {name: m for name, m in module.named_modules() if isinstance(m, subclass)}


def get_module_by_name(module, module_name):
    for name, m in module.named_modules():
        if name == module_name:
            return m
    return None


def set_module_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def load_model(model_path: str, device_map: str | None = None, dtype=torch.float32, **kwargs) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, dtype=dtype, **kwargs)
    return model


def load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def move_embed(model, device):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        # AutoModelForCausalLM returns Gemma4ForConditionalGeneration for Gemma 4. Need to unwrap.
        model = model.model.language_model
    elif hasattr(model, "model"):
        model = model.model
    else:
        raise NotImplementedError(type(model))

    def _move(module_name):
        module = getattr(model, module_name, None)
        if module is not None:
            module.to(device)

    _move("embed_tokens")
    _move("rotary_emb")

    # Gemma 4
    _move("embed_tokens_per_layer")
    _move("per_layer_model_projection")
    _move("per_layer_projection_norm")


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def get_mixed_calib_dataset(
    datasets: list[str],
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    per_dataset_len = n_samples // len(datasets)
    results = []
    for i, dataset in enumerate(datasets):
        dataset_samples = per_dataset_len if i < len(datasets) - 1 else n_samples - len(results)
        results.extend(
            get_calib_dataset(
                data=dataset,
                tokenizer=tokenizer,
                n_samples=dataset_samples,
                block_size=block_size,
                seed=seed,
                split=split,
            )
        )
    assert len(results) == n_samples, f"Expected {n_samples} samples, got {len(results)}"

    rand = random.Random(seed)
    rand.shuffle(results)

    return results


# Adapted from awq-llm
def get_calib_dataset(
    data="pileval",
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
) -> list[torch.Tensor]:
    if data == "pileval":
        if split != "validation":
            warnings.warn("The split argument is ignored when data is 'pileval'.")
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        dataset = dataset.shuffle(seed=seed)
    elif data == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        dataset = dataset.shuffle(seed=seed)
    elif data == "c4":
        if split == "train":
            dataset = load_dataset(
                "allenai/c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split=split,
            )
        elif split == "validation":
            dataset = load_dataset(
                "allenai/c4",
                data_files={"validation": "en/c4-validation.00001-of-00008.json.gz"},
                split=split,
            )
        dataset = dataset.shuffle(seed=seed)
    elif data == "redpajama":
        test_split, val_split = 0.2, 0.1
        dataset = load_dataset(
            "liang2kl/RedPajama-Data-1T-Sample-Backup",
            split="train",
            trust_remote_code=True,
        )
        dataset = dataset.shuffle(seed=seed)
        test_size = int(len(dataset) * test_split)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - test_size - val_size
        if split == "test":
            dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))
        elif split == "validation":
            dataset = dataset.select(range(len(dataset) - test_size - val_size, len(dataset) - test_size))
        elif split == "train":
            dataset = dataset.select(range(0, train_size))
        else:
            raise ValueError(f"Invalid split: {split}")
    else:
        raise NotImplementedError

    samples = []
    total_len = 0
    for row in dataset:
        line = row["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        total_len += len(line_encoded)
        if total_len >= n_samples * block_size:
            break
    samples = torch.cat(samples, dim=1).squeeze(0)
    n_split = min(samples.shape[0] // block_size, n_samples)

    return [samples[i * block_size : (i + 1) * block_size] for i in range(n_split)]


@torch.no_grad()
def capture_layer_inputs_and_args(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: torch.Tensor,
    batch_size: int | None,
) -> tuple[list[torch.Tensor], list[dict], list[list[tuple[torch.Tensor, ...]]], list[dict]]:
    device = samples.device
    kwargs_list: list[dict] = [{} for _ in range(len(layers))]
    first_layer_input_batches: list[torch.Tensor] = []
    other_args_batches_list: list[list[tuple[torch.Tensor, ...]]] = [[] for _ in range(len(layers))]
    retained_kwargs_batches: list[dict] = []

    class Catcher(nn.Module):

        def __init__(self, module, layer_idx):
            super().__init__()
            # Bypass __setattr__ of nn.Module
            object.__setattr__(self, "module", module)
            object.__setattr__(self, "layer_idx", layer_idx)

        def forward(self, *args, **kwargs):
            if self.layer_idx == 0:
                first_layer_input_batches.append(args[0].cpu())

            # Capture kwargs for all layers.
            layer_kwargs = kwargs_list[self.layer_idx]
            if len(layer_kwargs) == 0:
                layer_kwargs.update({k: v for k, v in kwargs.items() if k not in RETAINED_KWARG_KEYS})
                layer_kwargs.pop("use_cache", None)
                layer_kwargs.pop("past_key_value", None)
                layer_kwargs.pop("past_key_values", None)

            other_args_batches_list[self.layer_idx].append(
                tuple(arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args[1:])
            )
            retained_kwargs = {k: to_device(kwargs[k], "cpu") for k in RETAINED_KWARG_KEYS if k in kwargs}
            if self.layer_idx == 0:
                retained_kwargs_batches.append(retained_kwargs)

            return torch.empty_like(args[0], device=device)

        def __getattr__(self, name):
            return getattr(self.module, name)

    for layer_idx, layer in enumerate(layers):
        layers[layer_idx] = Catcher(layer, layer_idx)

    batch_size = samples.shape[0] if batch_size is None or batch_size <= 0 else batch_size
    num_batches = samples.shape[0] // batch_size
    samples_batch = samples.chunk(num_batches)
    for samples in samples_batch:
        model(samples)

    for layer_idx, layer in enumerate(layers):
        layers[layer_idx] = layer.module

    return (
        first_layer_input_batches,
        kwargs_list,
        other_args_batches_list,
        retained_kwargs_batches,
    )


def _move_tensor_batch(batch, device: torch.device | str) -> torch.Tensor | tuple[torch.Tensor, ...]:
    return to_device(batch, device)


def _tensor_batch_device(batch) -> torch.device:
    if isinstance(batch, tuple):
        return batch[0].device
    return batch.device


class CachedTensorShards:

    def __init__(
        self,
        batches: list[torch.Tensor] | list[tuple[torch.Tensor, ...]],
        num_shards: int,
        *,
        target_device: torch.device | str,
        offload_device: torch.device | str = torch.device("cpu"),
    ):
        assert len(batches) % num_shards == 0
        if _tensor_batch_device(batches[0]) != offload_device:
            self.batches = [_move_tensor_batch(b, offload_device) for b in batches]
        else:
            self.batches = batches
        self.num_shards = num_shards
        self.current_shard: int = None
        self.cached_shard: list[torch.Tensor] | list[tuple[torch.Tensor, ...]] = None
        self.target_device = target_device

    def _switch_shard(self, shard_index: int) -> None:
        if self.current_shard == shard_index:
            return
        self.current_shard = shard_index
        start, end = self._get_shard_range(shard_index)
        self.cached_shard = self.batches[start:end]
        self.cached_shard = [_move_tensor_batch(b, self.target_device) for b in self.cached_shard]

    def _get_shard_range(self, index: int) -> tuple[int, int]:
        if self.num_shards == 1:
            return 0, len(self.batches)
        shard_size = len(self.batches) // self.num_shards
        start = shard_size * index
        if index == self.num_shards - 1:
            end = len(self.batches)
        else:
            end = shard_size * (index + 1)
        return start, end

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, ...]:
        shard_len = len(self.batches) // self.num_shards
        shard_index = index // shard_len
        if self.current_shard != shard_index:
            self._switch_shard(shard_index)
        offset = index % shard_len
        return self.cached_shard[offset]

    def __iter__(self) -> "Iterator":
        return self.Iterator(self)

    def __len__(self) -> int:
        return len(self.batches)

    class Iterator:
        def __init__(self, batches: "CachedTensorShards"):
            self.batches = batches
            self.current_index = 0

        def __iter__(self):
            return self

        def __next__(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
            if self.current_index >= len(self.batches):
                raise StopIteration
            result = self.batches[self.current_index]
            self.current_index += 1
            return result

        def __len__(self) -> int:
            return len(self.batches)


class CosineAnnealingParam:
    def __init__(self, start_value: float, end_value: float, T_max: int):
        """
        Args:
            start_value (float): The initial value (equivalent to eta_max).
            end_value (float): The final value (equivalent to eta_min).
            T_max (int): Maximum number of steps.
        """
        self.start_value = start_value
        self.end_value = end_value
        self.T_max = T_max
        self._step = -1

    def step(self) -> float:
        self._step += 1

        if self._step >= self.T_max:
            return self.end_value

        cos_val = math.cos(math.pi * self._step / self.T_max)
        return self.end_value + (self.start_value - self.end_value) * (1 + cos_val) / 2


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = get_logger("ParoQuant")
