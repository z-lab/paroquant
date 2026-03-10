from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .math import dequantize_awq_4bit, fuse_inverse_rotations

PARO_BUFFER_SUFFIXES = {
    "qweight",
    "qzeros",
    "scales",
    "theta",
    "pairs",
    "channel_scales",
}

_REMOVE_IF_TEXT_ONLY = {
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "vision_config",
}

_SKIP_FILES_IF_TEXT_ONLY = {
    "preprocessor_config.json",
    "video_preprocessor_config.json",
}


@dataclass(frozen=True)
class HfExportResult:
    source_dir: Path
    output_dir: Path
    num_converted_layers: int
    bits: int
    group_size: int


def _resolve_model_dir(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    return Path(snapshot_download(model))


def _copy_non_weight_files(source_dir: Path, output_dir: Path, text_only: bool) -> None:
    for path in source_dir.iterdir():
        if path.name.endswith(".safetensors"):
            continue
        if path.name == "model.safetensors.index.json":
            continue
        if text_only and path.name in _SKIP_FILES_IF_TEXT_ONLY:
            continue

        target = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)


def _rewrite_config(output_dir: Path, bits: int, group_size: int, text_only: bool) -> None:
    cfg_path = output_dir / "config.json"
    if not cfg_path.exists():
        return

    cfg = json.loads(cfg_path.read_text())

    qcfg = cfg.get("quantization_config", {})
    if qcfg.get("quant_method") == "paroquant":
        qcfg.pop("quant_method", None)
        cfg.pop("quantization_config", None)

    cfg["_paroquant_export"] = {
        "source_quantization": "paroquant",
        "bits": bits,
        "group_size": group_size,
    }

    if text_only:
        for key in _REMOVE_IF_TEXT_ONLY:
            cfg.pop(key, None)

        if isinstance(cfg.get("architectures"), list):
            cfg["architectures"] = [
                arch.replace("ForConditionalGeneration", "ForCausalLM") for arch in cfg["architectures"]
            ]

    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def _rebuild_safetensors_index(source_dir: Path, output_dir: Path) -> None:
    source_index = source_dir / "model.safetensors.index.json"
    files = sorted(output_dir.glob("*.safetensors"))
    if not files:
        return

    if not source_index.exists() and len(files) == 1:
        return

    weight_map: dict[str, str] = {}
    total_size = 0

    for file in files:
        total_size += file.stat().st_size
        with safe_open(str(file), framework="pt") as handle:
            for key in handle.keys():
                weight_map[key] = file.name

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def _find_quantized_prefixes(state_dict: dict[str, torch.Tensor]) -> list[str]:
    prefixes = []
    for key in state_dict:
        if key.endswith(".qweight"):
            prefixes.append(key[: -len(".qweight")])
    return sorted(prefixes)


def _convert_quantized_prefix(
    prefix: str,
    state_dict: dict[str, torch.Tensor],
    bits: int,
    group_size: int,
) -> torch.Tensor:
    qweight_key = f"{prefix}.qweight"
    qzeros_key = f"{prefix}.qzeros"
    scales_key = f"{prefix}.scales"
    theta_key = f"{prefix}.theta"
    pairs_key = f"{prefix}.pairs"
    cscale_key = f"{prefix}.channel_scales"

    required = [qweight_key, qzeros_key, scales_key, theta_key, pairs_key, cscale_key]
    missing = [k for k in required if k not in state_dict]
    if missing:
        raise KeyError(f"Missing required ParoQuant tensors for {prefix}: {missing}")

    if bits != 4:
        raise ValueError(f"Only 4-bit ParoQuant export is supported currently, got bits={bits}")

    dense = dequantize_awq_4bit(
        qweight=state_dict[qweight_key],
        qzeros=state_dict[qzeros_key],
        scales=state_dict[scales_key],
        group_size=group_size,
    )

    fused = fuse_inverse_rotations(
        weight=dense,
        pairs=state_dict[pairs_key],
        theta=state_dict[theta_key],
        channel_scales=state_dict[cscale_key],
        group_size=group_size,
    )

    return fused.to(torch.float16)


def convert_paro_to_hf(
    model: str,
    output_dir: Path,
    text_only: bool = True,
) -> HfExportResult:
    source_dir = _resolve_model_dir(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = source_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {source_dir}")

    cfg = json.loads(cfg_path.read_text())
    qcfg = cfg.get("quantization_config", {})
    if qcfg.get("quant_method") != "paroquant":
        raise ValueError(f"Model is not a ParoQuant checkpoint: {model}")

    bits = int(qcfg.get("bits", 4))
    group_size = int(qcfg.get("group_size", 128))

    _copy_non_weight_files(source_dir, output_dir, text_only=text_only)

    converted_layers = 0
    src_safetensors = sorted(source_dir.glob("*.safetensors"))
    if not src_safetensors:
        raise FileNotFoundError(f"No safetensors files found in {source_dir}")

    for src_file in src_safetensors:
        state_dict = load_file(str(src_file), device="cpu")
        prefixes = _find_quantized_prefixes(state_dict)

        converted: dict[str, torch.Tensor] = {}
        skip_keys: set[str] = set()

        for prefix in prefixes:
            converted[f"{prefix}.weight"] = _convert_quantized_prefix(prefix, state_dict, bits=bits, group_size=group_size)
            converted_layers += 1

            if f"{prefix}.bias" in state_dict:
                converted[f"{prefix}.bias"] = state_dict[f"{prefix}.bias"].to(torch.float16)

            for suffix in PARO_BUFFER_SUFFIXES:
                skip_keys.add(f"{prefix}.{suffix}")

        out_state: dict[str, torch.Tensor] = {}

        for key, val in state_dict.items():
            if key in skip_keys:
                continue
            out_state[key] = val

        out_state.update(converted)
        save_file(out_state, str(output_dir / src_file.name))

    _rewrite_config(output_dir, bits=bits, group_size=group_size, text_only=text_only)
    _rebuild_safetensors_index(source_dir, output_dir)

    return HfExportResult(
        source_dir=source_dir,
        output_dir=output_dir,
        num_converted_layers=converted_layers,
        bits=bits,
        group_size=group_size,
    )
