"""Model loading for the Transformers backend."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from .modules import RotateQuantizedLinear


def _iter_linear_layers(module: nn.Module, prefix: str = "") -> Iterator[tuple[str, nn.Module, str, nn.Linear]]:
    """Yield (dotted_path, parent_module, attr_name, linear) for every nn.Linear in the tree."""
    for attr, child in module.named_children():
        path = f"{prefix}.{attr}" if prefix else attr
        if isinstance(child, nn.Linear):
            yield path, module, attr, child
        yield from _iter_linear_layers(child, path)


def _resolve_model_dir(model_id: str) -> Path:
    """Return local directory for a model, downloading from HF Hub if necessary."""
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id))


def _open_safetensor_shards(model_dir: Path) -> tuple[dict[str, str], dict]:
    """Open all safetensor shards and return (weight_map, file_handles)."""
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        sf = next(model_dir.glob("*.safetensors"))
        with safe_open(str(sf), framework="pt") as fh:
            weight_map = {k: sf.name for k in fh.keys()}

    handles: dict[str, object] = {}
    for shard_name in set(weight_map.values()):
        handles[shard_name] = safe_open(str(model_dir / shard_name), framework="pt")

    return weight_map, handles


@torch.no_grad()
def load(path: str) -> nn.Module:
    """Load a ParoQuant AutoAWQ checkpoint into a patched HuggingFace model."""
    config = AutoConfig.from_pretrained(path)

    qcfg = getattr(config, "quantization_config", None)
    if not qcfg:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype="auto", low_cpu_mem_usage=True,
            attn_implementation="sdpa", device_map="cuda",
        )

    if not isinstance(qcfg, dict):
        qcfg = qcfg.to_dict()
    if qcfg.get("quant_method") != "paroquant":
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype="auto", low_cpu_mem_usage=True,
            attn_implementation="sdpa", device_map="cuda",
        )

    import paroquant.kernels.cuda  # noqa: F401 — registers torch.ops.rotation.rotate

    bits = qcfg.get("bits", 4)
    group_size = qcfg.get("group_size", 128)

    model_dir = _resolve_model_dir(path)
    weight_map, handles = _open_safetensor_shards(model_dir)

    def _load_tensor(key: str) -> torch.Tensor:
        return handles[weight_map[key]].get_tensor(key)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", device_map="cpu",
    )

    # Build a map from checkpoint key prefix to model param prefix.
    # from_pretrained may strip prefixes (e.g. "model.language_model." → "model.").
    ckpt_qw_keys = {k for k in weight_map if k.endswith(".qweight")}

    def _resolve(param_path: str) -> str | None:
        """Find the checkpoint prefix that matches a model parameter path."""
        if f"{param_path}.qweight" in ckpt_qw_keys:
            return param_path
        for ck in ckpt_qw_keys:
            if ck.endswith(f"{param_path}.qweight") or ck.endswith(f".{param_path}.qweight"):
                return ck.removesuffix(".qweight")
        return None

    for layer_path, parent, attr, linear in tqdm(list(_iter_linear_layers(model)), desc="Loading quantized layers"):
        ckpt_prefix = _resolve(layer_path)
        if ckpt_prefix is None:
            continue

        rl = RotateQuantizedLinear(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size, bits=bits,
            rotation_angles=_load_tensor(f"{ckpt_prefix}.theta").cuda(),
            rotation_pairs=_load_tensor(f"{ckpt_prefix}.pairs").cuda(),
            channel_scales=_load_tensor(f"{ckpt_prefix}.channel_scales").cuda(),
            device="cuda",
        )
        rl.qlinear.qweight.data.copy_(_load_tensor(f"{ckpt_prefix}.qweight").cuda())
        rl.qlinear.qzeros.data.copy_(_load_tensor(f"{ckpt_prefix}.qzeros").cuda())
        rl.qlinear.scales.data.copy_(_load_tensor(f"{ckpt_prefix}.scales").cuda())
        if linear.bias is not None:
            rl.qlinear.bias.data.copy_(linear.bias.data.cuda())

        setattr(parent, attr, rl)

    model.cuda()
    return model
