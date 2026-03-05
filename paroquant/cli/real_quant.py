"""Convert optimization .pt results into AutoAWQ-format HuggingFace checkpoints.

    python -m paroquant.cli.real_quant --model <hf_id> --result-dir <dir> --output-path <dir>
"""

from __future__ import annotations

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from paroquant.inference.backends.transformers.modules import RotateQuantizedLinear
from paroquant.kernels.cuda import scaled_pairwise_rotation
from paroquant.optim.util import get_named_linears, set_module_by_name


_AWQ_REORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _load_model(model_id: str, dtype: torch.dtype) -> torch.nn.Module:
    kwargs = dict(
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def _get_blocks(model: torch.nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise NotImplementedError(f"Unsupported model structure: {type(model)}")


def _stack_if_numbered(state_dict: dict, key: str) -> torch.Tensor:
    if key in state_dict:
        return state_dict[key]

    values = []
    index = 0
    while f"{key}.{index}" in state_dict:
        values.append(state_dict[f"{key}.{index}"])
        index += 1
    if values:
        return torch.stack(values, dim=0)

    raise KeyError(key)


def _get_scalar(state_dict: dict, *keys: str) -> int:
    for key in keys:
        if key in state_dict:
            value = state_dict[key]
            if isinstance(value, torch.Tensor):
                return int(value.item())
            return int(value)
    raise KeyError(f"None of keys found: {keys}")


def _pack_awq(values: torch.Tensor, bits: int) -> torch.Tensor:
    if values.dtype != torch.int32:
        values = values.to(torch.int32)

    if bits != 4:
        raise ValueError(f"Only 4-bit is supported, got {bits}")

    pack_factor = 32 // bits
    if values.ndim != 2:
        raise ValueError(f"Expected 2D tensor to pack, got shape={tuple(values.shape)}")
    if values.shape[1] % pack_factor != 0:
        raise ValueError(f"Last dim must be divisible by {pack_factor}, got {tuple(values.shape)}")

    unpacked = values.view(values.shape[0], -1, pack_factor)
    unpacked = unpacked[:, :, _AWQ_REORDER]

    packed = torch.zeros(unpacked.shape[0], unpacked.shape[1], dtype=torch.int32, device=values.device)
    for index in range(pack_factor):
        packed |= (unpacked[:, :, index] & 0xF) << (bits * index)
    return packed


def _flatten_paroquant_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flatten RotateQuantizedLinear keys to AutoAWQ-style flat names.

    Transforms:
    - `<prefix>.rotation.theta` -> `<prefix>.theta`
    - `<prefix>.rotation.pairs` -> `<prefix>.pairs`
    - `<prefix>.rotation.channel_scales` -> `<prefix>.channel_scales`
    - `<prefix>.qlinear.qweight` -> `<prefix>.qweight`
    - `<prefix>.qlinear.qzeros` -> `<prefix>.qzeros`
    - `<prefix>.qlinear.scales` -> `<prefix>.scales`
    - `<prefix>.qlinear.bias` -> `<prefix>.bias`
    """
    flat = {}
    for key, value in state_dict.items():
        if ".rotation." in key:
            new_key = key.replace(".rotation.", ".")
        elif ".qlinear." in key:
            new_key = key.replace(".qlinear.", ".")
        else:
            new_key = key
        flat[new_key] = value
    return flat


def _load_optimize_args(result_dir: Path) -> dict:
    args_path = result_dir / "args.json"
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        return data
    return {}


def _build_dynamic_quant_config(args_data: dict, model: torch.nn.Module) -> dict[str, dict]:
    skipped_modules = args_data.get("skipped_modules", [])
    return {f"-:.*{keyword}": {} for keyword in skipped_modules}


def _resolve_model_dir(model_id: str) -> Path | None:
    path = Path(model_id)
    if path.exists() and path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=model_id))
    except Exception:
        return None


def _copy_non_weight_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy config/tokenizer files from the source model, skipping weights and READMEs."""
    weight_suffixes = {".safetensors", ".bin", ".pt", ".ckpt"}
    for src_path in src_dir.rglob("*"):
        if not src_path.is_file():
            continue
        if src_path.name.lower().startswith("readme"):
            continue
        if src_path.suffix in weight_suffixes:
            continue
        if src_path.name == "model.safetensors.index.json":
            continue

        relative = src_path.relative_to(src_dir)
        dst_path = dst_dir / relative
        if dst_path.exists():
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


@torch.no_grad()
def _convert_state_dict(state_dict: dict, device: str) -> dict[str, torch.Tensor | int]:
    """Convert a per-layer optimization .pt checkpoint to AutoAWQ int32 format."""
    weight = state_dict["weight"].to(device=device, dtype=torch.float32)
    out_features, in_features = weight.shape

    group_size = _get_scalar(state_dict, "group_size", "quantizer.group_size")
    bits = _get_scalar(state_dict, "n_bits", "quantizer.n_bits")
    if bits != 4:
        raise ValueError(f"Only 4-bit real quantization is supported, got n_bits={bits}")
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} is not divisible by group_size={group_size}")

    if "quantizer.scale" not in state_dict or "quantizer.zero_point_float" not in state_dict:
        raise ValueError("Checkpoint is missing quantizer parameters: quantizer.scale/zero_point_float")

    pairs = _stack_if_numbered(state_dict, "pairs_grouped").to(device=device, dtype=torch.short)
    theta = _stack_if_numbered(state_dict, "angles_grouped").to(device=device, dtype=torch.float32)

    channel_scales_opt = state_dict["channel_scales"].to(device=device, dtype=torch.float32)
    if channel_scales_opt.ndim == 1:
        channel_scales_opt = channel_scales_opt.unsqueeze(0)

    rotated_weight = scaled_pairwise_rotation(
        weight * channel_scales_opt,
        pairs,
        theta,
        None,
        group_size,
    )

    n_groups = in_features // group_size
    pack_factor = 32 // bits
    if out_features % pack_factor != 0:
        raise ValueError(f"out_features={out_features} is not divisible by pack={pack_factor}")

    scales_flat = state_dict["quantizer.scale"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    zp_flat = state_dict["quantizer.zero_point_float"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    if scales_flat.shape[0] != out_features * n_groups:
        raise ValueError(
            f"Unexpected quantizer.scale shape {tuple(state_dict['quantizer.scale'].shape)}, "
            f"expected {out_features * n_groups} rows"
        )

    zero_points = torch.clamp(-torch.round(zp_flat), 0, (1 << bits) - 1)

    grouped_weight = rotated_weight.reshape(-1, group_size)
    quantized = torch.clamp(torch.round(grouped_weight / scales_flat) + zero_points, 0, (1 << bits) - 1)
    quantized = quantized.to(torch.int32).reshape(out_features, in_features)

    qweight = _pack_awq(quantized.transpose(0, 1).contiguous(), bits).cpu()
    scales = scales_flat.reshape(out_features, n_groups).transpose(0, 1).contiguous().to(torch.float16).cpu()
    qzeros = _pack_awq(
        zero_points.to(torch.int32).reshape(out_features, n_groups).transpose(0, 1).contiguous(), bits
    ).cpu()

    channel_scales = (1.0 / channel_scales_opt).to(torch.float16).cpu()
    if channel_scales.ndim == 1:
        channel_scales = channel_scales.unsqueeze(0)

    converted = {
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "theta": theta.to(torch.float16).cpu(),
        "pairs": pairs.to(torch.short).cpu(),
        "channel_scales": channel_scales,
        "bits": bits,
        "group_size": group_size,
        "krot": int(theta.shape[0]),
    }
    if "bias" in state_dict and state_dict["bias"] is not None:
        converted["bias"] = state_dict["bias"].to(torch.float16).cpu()

    return converted


@torch.no_grad()
def main() -> None:
    parser = ArgumentParser(description="Convert optimization results to AutoAWQ-format HuggingFace checkpoint")
    parser.add_argument("--model", type=str, required=True, help="Base model HF id or local path")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory with per-layer .pt files")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for the quantized model")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real quant conversion (rotation kernel runs on CUDA).")

    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    model = _load_model(args.model, dtype=torch.float16)
    blocks = _get_blocks(model)
    optimize_args = _load_optimize_args(result_dir)

    converted_count = 0
    config_bits = 4
    config_group_size = 128
    config_krot = 8

    for layer_idx, layer in enumerate(tqdm(blocks, desc="Converting layers")):
        for name, module in get_named_linears(layer).items():
            result_file = result_dir / f"{layer_idx}.{name}.pt"
            if not result_file.exists():
                continue

            sd = torch.load(result_file, map_location="cpu", weights_only=False)
            converted = _convert_state_dict(sd, device="cuda")

            rl = RotateQuantizedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                group_size=int(converted["group_size"]),
                bits=int(converted["bits"]),
                rotation_angles=converted["theta"],
                rotation_pairs=converted["pairs"],
                channel_scales=converted["channel_scales"],
                device="cpu",
            )
            rl.qlinear.qweight.copy_(converted["qweight"])
            rl.qlinear.qzeros.copy_(converted["qzeros"])
            rl.qlinear.scales.copy_(converted["scales"])

            if module.bias is not None:
                bias = converted.get("bias", module.bias.detach().to(torch.float16))
                rl.qlinear.bias.copy_(bias)

            set_module_by_name(layer, name, rl)
            converted_count += 1
            config_bits = int(converted["bits"])
            config_group_size = int(converted["group_size"])
            config_krot = int(converted["krot"])

        layer.cpu()
        torch.cuda.empty_cache()

    if converted_count == 0:
        raise RuntimeError(f"No checkpoint files were matched in {result_dir}")

    quant_config = {
        "quant_method": "paroquant",
        "bits": config_bits,
        "group_size": config_group_size,
        "krot": config_krot,
    }
    dynamic = _build_dynamic_quant_config(args_data=optimize_args, model=model)
    if dynamic:
        quant_config["dynamic"] = dynamic
    model.config.quantization_config = quant_config

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, state_dict=_flatten_paroquant_keys(model.state_dict()))

    source_dir = _resolve_model_dir(args.model)
    if source_dir is not None:
        _copy_non_weight_files(source_dir, output_path)

    print(f"Converted {converted_count} linear layers")
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
