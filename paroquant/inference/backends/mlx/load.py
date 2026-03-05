import importlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

from .nn import RotateQuantizedLinear

_BITS = 4
_PF = 32 // _BITS  # 8 values per int32
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
    """Convert AutoAWQ int32 checkpoint to MLX quantized format.

    AutoAWQ: qweight (in, out//8) int32, scales (groups, out), qzeros (groups, out//8)
    MLX:     weight  (out, in//8) uint32, scales (out, groups), biases (out, groups)
    """
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
# Model loading
# ---------------------------------------------------------------------------

def _get_model_classes(config):
    from mlx_lm.utils import MODEL_REMAPPING

    model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])
    for name in (model_type, *(model_type.removesuffix(s) for s in ("_text", "_lm") if model_type.endswith(s))):
        try:
            arch = importlib.import_module(f"mlx_lm.models.{name}")
            return arch.Model, arch.ModelArgs
        except (ImportError, ModuleNotFoundError):
            continue
    raise ValueError(f"No model module for model_type={model_type!r}")


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


def _is_io_layer(path, module):
    return hasattr(module, "to_quantized") and (path.endswith("embed_tokens") or path.endswith("lm_head"))


def load(path_or_hf_repo: str, tokenizer_config: dict | None = None, lazy: bool = False):
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_config, load_tokenizer

    model_path = Path(path_or_hf_repo)
    if not model_path.is_dir():
        model_path = Path(snapshot_download(str(path_or_hf_repo)))

    config = load_config(model_path)
    paro = config.get("quantization_config", {})
    group_size = int(paro.get("group_size", 128))
    bits = int(paro.get("bits", 4))

    Model, ModelArgs = _get_model_classes(config)
    model = Model(ModelArgs.from_dict(config))

    weights = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        weights.update(mx.load(str(sf)))

    if any(k.endswith(".qweight") for k in weights):
        weights = _convert_autoawq(weights, group_size)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    _patch_rotation_layers(model, weights, bits, group_size)
    model.load_weights(list(weights.items()), strict=False)

    mlx_nn.quantize(model, group_size, bits, mode="affine", class_predicate=_is_io_layer)

    if not lazy:
        mx.eval(model.parameters())
    model.eval()

    return model, load_tokenizer(model_path, tokenizer_config)
