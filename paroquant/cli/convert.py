from __future__ import annotations

import json
import shutil
from contextlib import suppress
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from paroquant.optim.util import get_named_linears, set_module_by_name

_AWQ_REORDER = (0, 2, 4, 6, 1, 3, 5, 7)
_LAYER_PATHS = ["model.layers", "model.language_model.layers", "language_model.layers"]


def _resolve_source_dir(model_id: str) -> Path:
    model_path = Path(model_id).expanduser()
    if model_path.is_dir():
        return model_path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_id, repo_type="model"))


def _is_markdown(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".mdx", ".markdown"}


def _is_safetensor_related(path: Path) -> bool:
    return path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_source_non_md_files(
    source_dir: Path,
    output_dir: Path,
    keep_config: bool = False,
) -> set[Path]:
    copied: set[Path] = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for src in source_dir.rglob("*"):
        if not src.is_file():
            continue
        if _is_markdown(src) or _is_safetensor_related(src):
            continue
        if not keep_config and src.name == "config.json":
            continue
        rel = src.relative_to(source_dir)
        _copy_file(src, output_dir / rel)
        copied.add(rel)

    return copied


def _remove_safetensor_files(output_dir: Path) -> None:
    for existing in output_dir.rglob("*"):
        if existing.is_file() and _is_safetensor_related(existing):
            with suppress(FileNotFoundError):
                existing.unlink()


def _prune_non_md_extras(output_dir: Path, source_non_md_files: set[Path]) -> None:
    for existing in output_dir.rglob("*"):
        if not existing.is_file():
            continue
        if _is_markdown(existing) or _is_safetensor_related(existing) or existing.name == "config.json":
            continue
        rel = existing.relative_to(output_dir)
        if rel not in source_non_md_files:
            with suppress(FileNotFoundError):
                existing.unlink()


def _write_config_json(
    source_dir: Path,
    output_dir: Path,
    quant_config: dict[str, Any] | None,
) -> None:
    output_cfg = output_dir / "config.json"

    source_cfg = source_dir / "config.json"
    if source_cfg.exists():
        config = json.loads(source_cfg.read_text())
    elif output_cfg.exists():
        config = json.loads(output_cfg.read_text())
    else:
        raise FileNotFoundError(f"config.json not found in {source_dir} or {output_dir}")

    if quant_config is not None:
        config["quantization_config"] = quant_config

    output_cfg.write_text(json.dumps(config, indent=2) + "\n")


def _load_model(model_id: str, device_map: str = "cpu") -> torch.nn.Module:
    kwargs = dict(torch_dtype=torch.float16, device_map=device_map, low_cpu_mem_usage=True, trust_remote_code=True)
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def _get_blocks(model: torch.nn.Module):
    for path in _LAYER_PATHS:
        node = model
        try:
            for attr in path.split("."):
                node = getattr(node, attr)
            return node
        except AttributeError:
            continue
    raise NotImplementedError(f"Unsupported model structure: {type(model)}")


def _get_value(state_dict: dict, *keys: str):
    for key in keys:
        if key in state_dict:
            val = state_dict[key]
            return int(val.item()) if isinstance(val, torch.Tensor) else val
    raise KeyError(f"None of {keys} found")


def _stack_if_numbered(state_dict: dict, key: str) -> torch.Tensor:
    if key in state_dict:
        return state_dict[key]
    parts = []
    i = 0
    while f"{key}.{i}" in state_dict:
        parts.append(state_dict[f"{key}.{i}"])
        i += 1
    if parts:
        return torch.stack(parts)
    raise KeyError(key)


def _pack_awq(values: torch.Tensor, bits: int = 4) -> torch.Tensor:
    pack_factor = 32 // bits
    reordered = values.to(torch.int32).view(values.shape[0], -1, pack_factor)[:, :, _AWQ_REORDER]
    packed = torch.zeros(reordered.shape[0], reordered.shape[1], dtype=torch.int32, device=values.device)
    for i in range(pack_factor):
        packed |= (reordered[:, :, i] & 0xF) << (bits * i)
    return packed


@torch.no_grad()
def _convert_pseudo(model: torch.nn.Module, result_dir: Path) -> int:
    from paroquant.optim.qlinear import PseudoQuantizedLinear

    blocks = _get_blocks(model)
    count = 0
    for layer_idx, layer in enumerate(tqdm(blocks, desc="Pseudo-quantizing")):
        layer = layer.cuda()
        for name, module in get_named_linears(layer).items():
            pt_file = result_dir / f"{layer_idx}.{name}.pt"
            if not pt_file.exists():
                continue
            sd = torch.load(pt_file, weights_only=False, map_location="cuda")
            module.weight.data.copy_(PseudoQuantizedLinear.from_state_dict(sd).pseudo_weight())
            count += 1
        layer.cpu()
    return count


@torch.no_grad()
def _quantize_layer(state_dict: dict, device: str) -> tuple[dict[str, torch.Tensor], int, int, int]:
    from paroquant.kernels.cuda import scaled_pairwise_rotation

    weight = state_dict["weight"].to(device=device, dtype=torch.float32)
    out_features, in_features = weight.shape

    bits = int(_get_value(state_dict, "n_bits", "quantizer.n_bits"))
    group_size = int(_get_value(state_dict, "group_size", "quantizer.group_size"))

    pairs = _stack_if_numbered(state_dict, "pairs_grouped").to(device=device, dtype=torch.short)
    theta = _stack_if_numbered(state_dict, "angles_grouped").to(device=device, dtype=torch.float32)

    channel_scales_opt = state_dict["channel_scales"].to(device=device, dtype=torch.float32)
    if channel_scales_opt.ndim == 1:
        channel_scales_opt = channel_scales_opt.unsqueeze(0)

    rotated = scaled_pairwise_rotation(weight * channel_scales_opt, pairs, theta, None, group_size)

    n_groups = in_features // group_size
    scales_flat = state_dict["quantizer.scale"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    zp_flat = state_dict["quantizer.zero_point_float"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    zero_points = torch.clamp(-torch.round(zp_flat), 0, (1 << bits) - 1)

    quantized = (
        torch.clamp(
            torch.round(rotated.reshape(-1, group_size) / scales_flat) + zero_points,
            0,
            (1 << bits) - 1,
        )
        .to(torch.int32)
        .reshape(out_features, in_features)
    )

    channel_scales = (1.0 / channel_scales_opt).to(torch.float16).cpu()
    if channel_scales.ndim == 1:
        channel_scales = channel_scales.unsqueeze(0)

    buffers: dict[str, torch.Tensor] = {
        "qweight": _pack_awq(quantized.T.contiguous()).cpu(),
        "qzeros": _pack_awq(zero_points.to(torch.int32).reshape(out_features, n_groups).T.contiguous()).cpu(),
        "scales": scales_flat.reshape(out_features, n_groups).T.contiguous().to(torch.float16).cpu(),
        "theta": theta.to(torch.float16).cpu(),
        "pairs": pairs.cpu(),
        "channel_scales": channel_scales,
    }
    if "bias" in state_dict and state_dict["bias"] is not None:
        buffers["bias"] = state_dict["bias"].to(torch.float16).cpu()

    return buffers, bits, group_size, int(theta.shape[0])


@torch.no_grad()
def _convert_real(model: torch.nn.Module, result_dir: Path) -> tuple[int, dict[str, Any]]:
    from paroquant.inference.backends.transformers.modules import RotateQuantizedLinear

    blocks = _get_blocks(model)
    count = 0
    bits = group_size = krot = 0

    for layer_idx, layer in enumerate(tqdm(blocks, desc="Quantizing")):
        for name, module in get_named_linears(layer).items():
            pt_file = result_dir / f"{layer_idx}.{name}.pt"
            if not pt_file.exists():
                continue

            sd = torch.load(pt_file, map_location="cpu", weights_only=False)
            buffers, bits, group_size, krot = _quantize_layer(sd, device="cuda")

            rl = RotateQuantizedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                group_size=group_size,
                bits=bits,
                krot=krot,
            )
            rl.load_state_dict(buffers, strict=False)
            set_module_by_name(layer, name, rl)
            count += 1

        layer.cpu()
        torch.cuda.empty_cache()

    quant_config: dict[str, Any] = {
        "quant_method": "paroquant",
        "bits": bits,
        "group_size": group_size,
        "krot": krot,
    }
    return count, quant_config


@torch.no_grad()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--mode", choices=["real", "pseudo"], default="real")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    source_dir = _resolve_source_dir(args.model)
    model = _load_model(str(source_dir), device_map="cpu" if args.mode == "real" else "cuda")

    quant_config: dict[str, Any] | None = None
    if args.mode == "pseudo":
        count = _convert_pseudo(model, result_dir)
    else:
        count, quant_config = _convert_real(model, result_dir)
        model.config.quantization_config = quant_config

    if count == 0:
        raise RuntimeError(f"No checkpoint files matched in {result_dir}")

    output_path = Path(args.output_path)
    source_non_md = _copy_source_non_md_files(source_dir, output_path)
    _remove_safetensor_files(output_path)

    model.save_pretrained(output_path, safe_serialization=True)

    # Restore original non-MD assets in case save_pretrained overwrote any.
    source_non_md = _copy_source_non_md_files(source_dir, output_path)
    _prune_non_md_extras(output_path, source_non_md)
    _write_config_json(source_dir, output_path, quant_config)

    if not any(_is_safetensor_related(p) for p in output_path.rglob("*") if p.is_file()):
        raise RuntimeError("No safetensors artifacts were generated.")

    print(f"Converted {count} layers ({args.mode}) → {output_path}")


if __name__ == "__main__":
    main()
