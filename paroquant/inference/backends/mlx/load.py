"""Model loading for the MLX backend (auto-detects LLM vs VLM)."""

import importlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

from .modules import RotateQuantizedLinear

# ---------------------------------------------------------------------------
# AutoAWQ → MLX weight conversion
# ---------------------------------------------------------------------------

_BITS = 4
_PF = 32 // _BITS
_MASK = (1 << _BITS) - 1
_SHIFTS = np.arange(0, 32, _BITS, dtype=np.int64)
_INV_REORDER = np.array([0, 4, 1, 5, 2, 6, 3, 7])


def _unpack_and_reorder(packed: np.ndarray) -> np.ndarray:
    """Unpack AutoAWQ int32 → raw uint8, undoing the [0,2,4,6,1,3,5,7] reorder."""
    raw = ((packed.astype(np.int64)[:, :, None] >> _SHIFTS) & _MASK).astype(np.uint8)
    return raw[:, :, _INV_REORDER].reshape(packed.shape[0], -1)


def _pack_mlx(w: np.ndarray) -> np.ndarray:
    """Pack raw uint8 values into uint32 (MLX sequential layout)."""
    w = w.reshape(w.shape[0], -1, _PF)
    p = w[:, :, 0].astype(np.uint32)
    for i in range(1, _PF):
        p |= w[:, :, i].astype(np.uint32) << (_BITS * i)
    return p


def _convert_autoawq(weights, group_size):
    """Convert AutoAWQ int32 checkpoint to MLX quantized format."""
    prefixes = {k.removesuffix("qweight")
                for k in weights if k.endswith(".qweight") and f"{k[:-len('qweight')]}theta" in weights}
    if not prefixes:
        return weights

    out = {}
    for key, val in weights.items():
        pfx = next((p for p in prefixes if key.startswith(p)), None)
        if pfx is None:
            out[key] = val
            continue

        suffix = key[len(pfx):]
        if suffix == "qweight":
            out[f"{pfx}weight"] = mx.array(_pack_mlx(_unpack_and_reorder(np.array(val)).T))
        elif suffix == "scales":
            out[f"{pfx}scales"] = mx.array(np.array(val).T.copy())
        elif suffix == "qzeros":
            zeros = _unpack_and_reorder(np.array(val)).astype(np.float32)
            scales = np.array(weights[f"{pfx}scales"]).astype(np.float32)
            out[f"{pfx}biases"] = mx.array((-scales * zeros).T.copy().astype(np.float16))
        elif suffix in ("theta", "pairs", "channel_scales", "bias"):
            out[key] = val.reshape(1, -1) if suffix == "channel_scales" and val.ndim == 1 else val

    return out


# ---------------------------------------------------------------------------
# Model construction (LLM vs VLM)
# ---------------------------------------------------------------------------

def _create_model(config: dict, is_vlm: bool):
    if is_vlm:
        from mlx_vlm.utils import get_model_and_args, update_module_configs
        mod, _ = get_model_and_args(config=config)
        cfg = mod.ModelConfig.from_dict(config)
        cfg = update_module_configs(cfg, mod, config, ["text", "vision", "perceiver", "projector", "audio"])
        return mod.Model(cfg)

    from mlx_lm.utils import MODEL_REMAPPING
    model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])
    for name in (model_type, *(model_type.removesuffix(s) for s in ("_text", "_lm") if model_type.endswith(s))):
        try:
            arch = importlib.import_module(f"mlx_lm.models.{name}")
            return arch.Model(arch.ModelArgs.from_dict(config))
        except (ImportError, ModuleNotFoundError):
            continue
    raise ValueError(f"No mlx_lm model for model_type={model_type!r}")


def _load_processor(local_dir: Path, model, is_vlm: bool):
    if not is_vlm:
        from mlx_lm.utils import load_tokenizer
        return load_tokenizer(local_dir)

    try:
        from mlx_vlm.utils import load_image_processor, load_processor
        processor = load_processor(local_dir, trust_remote_code=True)
        image_processor = load_image_processor(local_dir)
        if image_processor is not None:
            processor.image_processor = image_processor
        return processor
    except Exception:
        from mlx_vlm.utils import load_tokenizer
        processor = load_tokenizer(local_dir)

    if not hasattr(processor, "stopping_criteria"):
        eos = getattr(model.config, "eos_token_id", None)
        eos_set = set(eos) if isinstance(eos, list) else {eos} if eos is not None else set()
        processor.stopping_criteria = lambda token: int(token) in eos_set

    return processor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_module(root, path):
    node = root
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, (list, tuple)) else getattr(node, part)
    return node


def _set_module(root, path, value):
    parts = path.split(".")
    parent = _get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    if isinstance(parent, list):
        parent[int(parts[-1])] = value
    else:
        setattr(parent, parts[-1], value)


def _patch_rotation_layers(model, weights, bits, group_size):
    for prefix in sorted(k.removesuffix(".theta") for k in weights if k.endswith(".theta")):
        original = _get_module(model, prefix)
        _set_module(
            model, prefix,
            RotateQuantizedLinear(
                input_dims=original.weight.shape[-1],
                output_dims=original.weight.shape[0],
                bias="bias" in original,
                group_size=group_size, bits=bits,
                krot=weights[f"{prefix}.theta"].shape[0],
            ),
        )


def _is_io_layer(path, module):
    return hasattr(module, "to_quantized") and (path.endswith("embed_tokens") or path.endswith("lm_head"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(model_path: str, lazy: bool = False) -> tuple:
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
    is_vlm = "vision_config" in config

    # 1. Create model
    model = _create_model(config, is_vlm)

    # 2. Load + convert weights
    weights = {}
    for sf in sorted(local_dir.glob("*.safetensors")):
        weights.update(mx.load(str(sf)))
    if any(k.endswith(".qweight") for k in weights):
        weights = _convert_autoawq(weights, group_size)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # 3. Patch rotation layers + load weights
    _patch_rotation_layers(model, weights, bits, group_size)
    model.load_weights(list(weights.items()), strict=False)
    mlx_nn.quantize(model, group_size, bits, mode="affine", class_predicate=_is_io_layer)

    if not lazy:
        mx.eval(model.parameters())
    model.eval()

    # 4. Load processor/tokenizer
    processor = _load_processor(local_dir, model, is_vlm)

    return model, processor, is_vlm
