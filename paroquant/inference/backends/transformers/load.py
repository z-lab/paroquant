"""Model loading for the Transformers backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from .nn import RotateLinearW4A16


def _iter_linears(module: nn.Module, prefix: str = "") -> Iterator[tuple[str, nn.Module, str, nn.Linear]]:
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, module, name, child
        yield from _iter_linears(child, full)


def _resolve_local_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id))


def _open_weight_shards(local_dir: Path) -> tuple[dict[str, str], dict]:
    from safetensors import safe_open

    index_path = local_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        sf = next(local_dir.glob("*.safetensors"))
        with safe_open(str(sf), framework="pt") as fh:
            weight_map = {k: sf.name for k in fh.keys()}

    handles: dict[str, object] = {}
    for shard_name in set(weight_map.values()):
        handles[shard_name] = safe_open(str(local_dir / shard_name), framework="pt")

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

    local_dir = _resolve_local_dir(path)
    weight_map, handles = _open_weight_shards(local_dir)

    def get(key: str) -> torch.Tensor:
        return handles[weight_map[key]].get_tensor(key)

    model = AutoModelForCausalLM.from_pretrained(
        local_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", device_map="cpu",
    )

    for full, parent, key, linear in tqdm(list(_iter_linears(model)), desc="Loading quantized layers"):
        qw_key = f"{full}.qweight"
        if qw_key not in weight_map:
            continue

        rl = RotateLinearW4A16(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size, bits=bits,
            rotation_angles=get(f"{full}.theta").cuda(),
            rotation_pairs=get(f"{full}.pairs").cuda(),
            channel_scales=get(f"{full}.channel_scales").cuda(),
            device="cuda",
        )
        rl.qlinear.qweight.data.copy_(get(f"{full}.qweight").cuda())
        rl.qlinear.qzeros.data.copy_(get(f"{full}.qzeros").cuda())
        rl.qlinear.scales.data.copy_(get(f"{full}.scales").cuda())
        if linear.bias is not None:
            rl.qlinear.bias.data.copy_(linear.bias.data.cuda())

        setattr(parent, key, rl)

    model.cuda()
    return model
