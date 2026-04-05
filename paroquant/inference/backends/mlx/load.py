from __future__ import annotations

import logging
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

from .modules import RotateQuantizedLinear, RotateSwitchGLU

logger = logging.getLogger(__name__)

_BITS = 4
_MASK = (1 << _BITS) - 1
_SHIFTS = np.arange(0, 32, _BITS, dtype=np.int64)
_INV_REORDER = np.array([0, 4, 1, 5, 2, 6, 3, 7])


def _unpack_and_reorder(packed: np.ndarray) -> np.ndarray:
    """Unpack AutoAWQ int32 → raw uint8, undoing the [0,2,4,6,1,3,5,7] reorder."""
    raw = ((packed.astype(np.int64)[:, :, None] >> _SHIFTS) & _MASK).astype(np.uint8)
    return raw[:, :, _INV_REORDER].reshape(packed.shape[0], -1)


def _pack_mlx(w: np.ndarray, bits: int = _BITS) -> np.ndarray:
    """Pack raw uint8 values into uint32 (MLX sequential layout)."""
    pf = 32 // bits
    w = w.reshape(w.shape[0], -1, pf)
    p = w[:, :, 0].astype(np.uint32)
    for i in range(1, pf):
        p |= w[:, :, i].astype(np.uint32) << (bits * i)
    return p


def _unpack_int16_nibbles(packed: np.ndarray, bits: int = _BITS) -> np.ndarray:
    """Unpack int16 values packed along the output dimension into uint8 nibbles."""
    pf = 16 // bits
    shifts = np.arange(0, 16, bits, dtype=np.int32)
    mask = (1 << bits) - 1
    nibbles = ((packed.astype(np.int32)[:, :, None] >> shifts) & mask).astype(np.uint8)
    return nibbles.reshape(packed.shape[0] * pf, packed.shape[1])


def _convert_awq_linear(weights: dict, prefix: str, group_size: int) -> dict:
    """Convert one AutoAWQ linear layer (qweight/scales/qzeros) → MLX (weight/scales/biases)."""
    qw = np.array(weights[f"{prefix}qweight"])
    weight = mx.array(_pack_mlx(_unpack_and_reorder(qw).T))
    scales_np = np.array(weights[f"{prefix}scales"]).astype(np.float32)
    scales = mx.array(scales_np.T.copy())
    zeros = _unpack_and_reorder(np.array(weights[f"{prefix}qzeros"])).astype(np.float32)
    biases = mx.array((-scales_np * zeros).T.copy().astype(np.float16))
    return {"weight": weight, "scales": scales, "biases": biases}


def _convert_autoawq(weights: dict, group_size: int) -> dict:
    """Convert AutoAWQ int32 checkpoint to MLX quantized format."""
    prefixes = {k.removesuffix("qweight") for k in weights if k.endswith(".qweight")}
    if not prefixes:
        return weights

    out = {}
    for key, val in weights.items():
        pfx = next((p for p in prefixes if key.startswith(p)), None)
        if pfx is None:
            out[key] = val
            continue

        suffix = key[len(pfx) :]
        if suffix == "qweight":
            converted = _convert_awq_linear(weights, pfx, group_size)
            for k, v in converted.items():
                out[f"{pfx}{k}"] = v
        elif suffix in ("qzeros", "scales"):
            pass  # consumed by qweight branch
        elif suffix == "channel_scales":
            out[key] = val.reshape(1, -1) if val.ndim == 1 else val
        else:
            out[key] = val

    return out


def _convert_paro_native(weights: dict, group_size: int, bits: int = _BITS) -> dict:
    """Convert PARO-native checkpoint format (qlinear.*/rotation.*) to MLX format."""
    qlinear_keys = [k for k in weights if k.endswith(".qlinear.qweight")]
    if not qlinear_keys:
        return weights

    prefixes = {k.removesuffix(".qlinear.qweight") for k in qlinear_keys}

    out = {}
    for key, val in weights.items():
        pfx = next((p for p in prefixes if key.startswith(p + ".")), None)
        if pfx is None:
            out[key] = val
            continue

        rel = key[len(pfx) + 1 :]

        if rel == "qlinear.qweight":
            raw = _unpack_int16_nibbles(np.array(val), bits)
            out[f"{pfx}.weight"] = mx.array(_pack_mlx(raw, bits))
        elif rel == "qlinear.scales":
            out[f"{pfx}.scales"] = mx.array(np.array(val).T.copy())
        elif rel == "qlinear.scaled_zeros":
            out[f"{pfx}.biases"] = mx.array((-np.array(val)).T.copy().astype(np.float16))
        elif rel == "rotation.theta":
            out[f"{pfx}.theta"] = val
        elif rel == "rotation.pairs":
            out[f"{pfx}.pairs"] = val
        elif rel == "rotation.channel_scales":
            out[f"{pfx}.channel_scales"] = val.reshape(1, -1) if val.ndim == 1 else val
        else:
            out[key] = val

    return out


_EXPERT_RE = re.compile(r"^(.+)\.experts\.(\d+)\.(\w+)\.(\w+)$")
_STACKABLE = ("weight", "scales", "biases")
_SHARED_ROT_RE = re.compile(
    r"^(.+)\.experts\.(gate_up_weight|down_weight)_(theta|pairs|channel_scales)$"
)


def _stack_moe_expert_weights(weights: dict) -> dict:
    """Stack per-expert params into the switch_mlp format for SwitchGLU.

    ``...experts.{e}.{proj}.{suffix}`` → ``...switch_mlp.{proj}.{suffix}``
    """
    groups: dict[str, dict[int, dict[str, str]]] = {}

    for key in list(weights):
        m = _EXPERT_RE.match(key)
        if m is None:
            continue
        base, expert_str, proj, suffix = m.groups()
        if suffix not in _STACKABLE:
            continue
        group_key = f"{base}.switch_mlp.{proj}"
        groups.setdefault(group_key, {}).setdefault(int(expert_str), {})[suffix] = key

    for group_key, experts_dict in groups.items():
        num_experts = max(experts_dict) + 1
        for suffix in _STACKABLE:
            dest = f"{group_key}.{suffix}"
            if dest in weights:
                continue
            src = [experts_dict.get(e, {}).get(suffix) for e in range(num_experts)]
            if all(src):
                weights[dest] = mx.stack([weights.pop(k) for k in src])

    return weights


def _remap_shared_moe_rotation(weights: dict) -> dict:
    """Remap shared expert rotation keys to the switch_mlp namespace.

    ``...experts.gate_up_weight_{theta|pairs|channel_scales}``
    → ``...switch_mlp.gate_up_rot_{theta|pairs|channel_scales}``

    ``...experts.down_weight_{theta|pairs|channel_scales}``
    → ``...switch_mlp.down_rot_{theta|pairs|channel_scales}``
    """
    remap = {
        "gate_up_weight": "gate_up_rot",
        "down_weight": "down_rot",
    }
    for key in list(weights):
        m = _SHARED_ROT_RE.match(key)
        if m is None:
            continue
        base, proj, suffix = m.groups()
        new_key = f"{base}.switch_mlp.{remap[proj]}_{suffix}"
        weights[new_key] = weights.pop(key)
    return weights


def _create_model(config: dict, is_vlm: bool):
    """Instantiate model from config, dispatching to mlx_vlm or mlx_lm."""
    if is_vlm:
        from mlx_vlm.utils import get_model_and_args, update_module_configs

        mod, _ = get_model_and_args(config=config)
        cfg = mod.ModelConfig.from_dict(config)
        cfg = update_module_configs(cfg, mod, config, ["text", "vision", "perceiver", "projector", "audio"])
        return mod.Model(cfg)

    from mlx_lm.utils import _get_classes

    model_class, model_args_class = _get_classes(config)
    return model_class(model_args_class.from_dict(config))


def _load_processor(local_dir: Path, model, is_vlm: bool):
    """Load tokenizer (LLM) or processor (VLM)."""
    if not is_vlm:
        from mlx_lm.utils import load_tokenizer

        return load_tokenizer(local_dir)

    from mlx_vlm.utils import load_image_processor, load_processor

    try:
        processor = load_processor(local_dir, trust_remote_code=True)
        image_processor = load_image_processor(local_dir)
        if image_processor is not None:
            processor.image_processor = image_processor
        return processor
    except (ImportError, TypeError, AttributeError):
        pass

    from mlx_vlm.utils import load_tokenizer

    processor = load_tokenizer(local_dir)
    if not hasattr(processor, "stopping_criteria"):
        eos = getattr(model.config, "eos_token_id", None)
        eos_set = set(eos) if isinstance(eos, list) else {eos} if eos is not None else set()
        processor.stopping_criteria = lambda token: int(token) in eos_set
    return processor


def _get_module(root, path: str):
    node = root
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, (list, tuple)) else getattr(node, part)
    return node


def _set_module(root, path: str, value):
    parts = path.split(".")
    parent = _get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    if isinstance(parent, list):
        parent[int(parts[-1])] = value
    else:
        setattr(parent, parts[-1], value)


def _patch_quantized_layers(model, weights: dict, bits: int, group_size: int):
    """Replace linear layers with appropriate quantized variants based on weight keys.

    - Dense layers with .theta → RotateQuantizedLinear
    - SwitchLinear with stacked uint32 weight → QuantizedSwitchLinear
    - SwitchGLU with shared rotation keys → RotateSwitchGLU
    """
    try:
        from mlx_lm.models.switch_layers import (
            SwitchGLU,
            SwitchLinear,
            QuantizedSwitchLinear,
        )

        _has_switch = True
    except ImportError:
        _has_switch = False

    # Dense rotation layers
    for prefix in sorted(k.removesuffix(".theta") for k in weights if k.endswith(".theta")):
        original = _get_module(model, prefix)
        _set_module(
            model,
            prefix,
            RotateQuantizedLinear(
                input_dims=original.weight.shape[-1],
                output_dims=original.weight.shape[0],
                bias="bias" in original,
                group_size=group_size,
                bits=bits,
                krot=weights[f"{prefix}.theta"].shape[0],
            ),
        )

    if not _has_switch:
        return

    # SwitchLinear → QuantizedSwitchLinear (quantized experts without rotation)
    for path, mod in model.named_modules():
        if not isinstance(mod, SwitchLinear):
            continue
        w = weights.get(f"{path}.weight")
        if w is not None and w.dtype in (mx.uint32, mx.uint16, mx.uint8):
            _set_module(
                model,
                path,
                QuantizedSwitchLinear(
                    mod.input_dims,
                    mod.output_dims,
                    mod.num_experts,
                    bias="bias" in mod,
                    group_size=group_size,
                    bits=bits,
                ),
            )

    # SwitchGLU → RotateSwitchGLU (when shared rotation keys are present)
    for path, mod in model.named_modules():
        if not isinstance(mod, SwitchGLU):
            continue
        rot_key = f"{path}.gate_up_rot_theta"
        if rot_key not in weights:
            continue
        krot = weights[rot_key].shape[0]
        _set_module(model, path, RotateSwitchGLU(mod, group_size, krot))



def _is_io_layer(path, module):
    """Predicate for mlx_nn.quantize: only quantize embed_tokens/lm_head natively."""
    return hasattr(module, "to_quantized") and (path.endswith("embed_tokens") or path.endswith("lm_head"))


def load(model_path: str, lazy: bool = False, force_text: bool = False) -> tuple:
    """Load a ParoQuant model for MLX. Returns (model, processor, is_vlm)."""
    from huggingface_hub import snapshot_download

    local_dir = Path(model_path)
    if not local_dir.is_dir():
        local_dir = Path(snapshot_download(model_path))

    try:
        from mlx_vlm.utils import load_config
    except ImportError:
        from mlx_lm.utils import load_config
    config = load_config(local_dir)

    paro = config.get("quantization_config", {})
    group_size = int(paro.get("group_size", 128))
    bits = int(paro.get("bits", 4))

    weights = {}
    for sf in sorted(local_dir.glob("*.safetensors")):
        weights.update(mx.load(str(sf)))

    has_vision = any(k.startswith(("vision_tower.", "model.visual.", "visual.")) for k in weights)
    is_vlm = "vision_config" in config and not force_text and has_vision

    model = _create_model(config, is_vlm)

    if any(k.endswith(".qlinear.qweight") for k in weights):
        weights = _convert_paro_native(weights, group_size, bits)
    elif any(k.endswith(".qweight") for k in weights):
        weights = _convert_autoawq(weights, group_size)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    if is_vlm and hasattr(model, "vision_tower") and hasattr(model.vision_tower, "sanitize"):
        weights = model.vision_tower.sanitize(weights)

    weights = _stack_moe_expert_weights(weights)
    weights = _remap_shared_moe_rotation(weights)
    _patch_quantized_layers(model, weights, bits, group_size)
    model.load_weights(list(weights.items()), strict=False)
    mlx_nn.quantize(model, group_size, bits, mode="affine", class_predicate=_is_io_layer)

    if not lazy:
        mx.eval(model.parameters())
    model.eval()

    processor = _load_processor(local_dir, model, is_vlm)
    logger.info("Loaded %s model from %s (VLM=%s).", config.get("model_type", "unknown"), model_path, is_vlm)

    return model, processor, is_vlm
