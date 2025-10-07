import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import gc
from typing import Optional, TypeVar
import warnings
import random
import logging
from tqdm import tqdm


def get_blocks(model: nn.Module) -> nn.ModuleList:
    model_class_name = model.__class__.__name__
    if model_class_name in (
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "LlamaForCausalLM",
    ):
        m = model.model
    else:
        raise NotImplementedError(type(model))

    return m.layers


_Linear_T = TypeVar("Linear", bound=nn.Module)


def get_named_linears(
    module: nn.Module, subclass: type[_Linear_T] = nn.Linear
) -> dict[str, _Linear_T]:
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


def load_model(
    model_path: str, device_map: str = None, dtype=torch.float32, **kwargs
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype, **kwargs
    )
    return model


def load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def move_embed(model, device):
    model_class_name = model.__class__.__name__
    if model_class_name in (
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "LlamaForCausalLM",
    ):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    else:
        raise NotImplementedError(type(model))


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
        dataset_samples = (
            per_dataset_len if i < len(datasets) - 1 else n_samples - len(results)
        )
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
    assert (
        len(results) == n_samples
    ), f"Expected {n_samples} samples, got {len(results)}"

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
            "togethercomputer/RedPajama-Data-1T-Sample",
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
            dataset = dataset.select(
                range(len(dataset) - test_size - val_size, len(dataset) - test_size)
            )
        elif split == "train":
            dataset = dataset.select(range(0, train_size))
        else:
            raise ValueError(f"Invalid split: {split}")
    else:
        raise NotImplementedError

    samples = []
    total_len = 0
    for data in dataset:
        line = data["text"]
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
def catch_first_layer_input(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: torch.Tensor,
    batch_size: Optional[int],
) -> tuple[torch.Tensor, dict]:
    layer_kwargs = {}
    batched = batch_size is not None
    inps: list[torch.Tensor] = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            # Bypass __setattr__ of nn.Module
            object.__setattr__(self, "module", module)

        def forward(self, inp, **kwargs):
            inps.append(inp)
            if len(layer_kwargs) == 0:
                layer_kwargs.update(kwargs)
            raise ValueError

        def __getattr__(self, name):
            return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    batch_size = samples.shape[0] if not batched or batch_size <= 0 else batch_size
    num_batches = samples.shape[0] // batch_size
    samples_batch = samples.chunk(num_batches)
    for samples in samples_batch:
        try:
            model(samples.to(next(model.parameters()).device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    if not batched:
        inps = inps[0]

    layer_kwargs["use_cache"] = False
    if "past_key_value" in layer_kwargs:
        del layer_kwargs["past_key_value"]
    if "past_key_values" in layer_kwargs:
        del layer_kwargs["past_key_values"]

    return inps, layer_kwargs


class CachedTensorShards:
    def __init__(
        self,
        batches: list[torch.Tensor],
        num_shards: int,
        *,
        target_device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
    ):
        assert len(batches) % num_shards == 0
        if batches[0].device != offload_device:
            self.batches = [b.to(offload_device) for b in batches]
        else:
            self.batches = batches
        self.num_shards = num_shards
        self.current_shard: int = None
        self.cached_shard: list[torch.Tensor] = None
        self.target_device = target_device

    def _switch_shard(self, shard_index: int) -> None:
        if self.current_shard == shard_index:
            return
        self.current_shard = shard_index
        start, end = self._get_shard_range(shard_index)
        self.cached_shard = self.batches[start:end]
        self.cached_shard = [b.to(self.target_device) for b in self.cached_shard]

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

    def __getitem__(self, index: int) -> torch.Tensor:
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

        def __next__(self) -> torch.Tensor:
            if self.current_index >= len(self.batches):
                raise StopIteration
            result = self.batches[self.current_index]
            self.current_index += 1
            return result

        def __len__(self) -> int:
            return len(self.batches)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def clamp_ste(
    x: torch.Tensor,
    min: Optional[torch.Tensor] = None,
    max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return (x.clamp(min, max) - x).detach() + x


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
