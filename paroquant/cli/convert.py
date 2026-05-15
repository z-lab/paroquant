from __future__ import annotations

from collections import defaultdict
import json
import shutil
from contextlib import suppress
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from paroquant.optim.qexperts import PseudoQuantizedMoEExperts, get_named_moe_experts, is_fused_moe_experts
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


def _quantize_rotated_weight(
    *,
    weight: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    channel_scales: torch.Tensor,
    scales_flat: torch.Tensor,
    zp_flat: torch.Tensor,
    bits: int,
    group_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from paroquant.kernels.cuda import scaled_pairwise_rotation

    out_features, in_features = weight.shape
    if channel_scales.ndim == 1:
        channel_scales = channel_scales.unsqueeze(0)

    rotated = scaled_pairwise_rotation(weight * channel_scales, pairs, theta, None, group_size)
    n_groups = in_features // group_size

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
    scales_2d = scales_flat.reshape(out_features, n_groups).to(device=device, dtype=torch.float32)
    zeros_2d = zero_points.reshape(out_features, n_groups).to(device=device, dtype=torch.int32)
    return quantized, scales_2d, zeros_2d


def _to_awq_buffers(
    quantized: torch.Tensor,
    scales_2d: torch.Tensor,
    zeros_2d: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "qweight": _pack_awq(quantized.T.contiguous()).cpu(),
        "qzeros": _pack_awq(zeros_2d.T.contiguous()).cpu(),
        "scales": scales_2d.T.contiguous().to(torch.float16).cpu(),
    }


@torch.no_grad()
def _convert_pseudo(model: torch.nn.Module, result_dir: Path) -> int:
    from paroquant.optim.qlinear import PseudoQuantizedLinear

    blocks = _get_blocks(model)
    count = 0
    for layer_idx, layer in enumerate(tqdm(blocks, desc="Pseudo-quantizing")):
        layer = layer.cuda()
        modules: dict[str, torch.nn.Module] = {}
        modules.update(get_named_linears(layer))
        modules.update(get_named_moe_experts(layer))

        for name, module in modules.items():
            pt_file = result_dir / f"{layer_idx}.{name}.pt"
            if not pt_file.exists():
                continue
            sd = torch.load(pt_file, weights_only=False, map_location="cuda")
            if isinstance(module, torch.nn.Linear):
                qlinear = PseudoQuantizedLinear.from_state_dict(sd)
                module.weight.data.copy_(qlinear.pseudo_weight())
                if module.bias is not None and qlinear.bias is not None:
                    module.bias.data.copy_(qlinear.bias.data)
            elif is_fused_moe_experts(module):
                qexperts = PseudoQuantizedMoEExperts.from_state_dict(sd, module, "cuda")
                module.gate_up_proj.data.copy_(qexperts.gate_up_proj.data)
                module.down_proj.data.copy_(qexperts.down_proj.data)
            else:
                raise NotImplementedError(f"Unsupported module type: {type(module)}")
            count += 1
        layer.cpu()
    return count


@torch.no_grad()
def _quantize_layer(state_dict: dict, device: str) -> tuple[dict[str, torch.Tensor], int, int, int]:
    weight = state_dict["weight"].to(device=device, dtype=torch.float32)

    bits = int(_get_value(state_dict, "n_bits", "quantizer.n_bits"))
    group_size = int(_get_value(state_dict, "group_size", "quantizer.group_size"))

    pairs = _stack_if_numbered(state_dict, "pairs_grouped").to(device=device, dtype=torch.int16)
    theta = _stack_if_numbered(state_dict, "angles_grouped").to(device=device, dtype=torch.float32)

    channel_scales_opt = state_dict["channel_scales"].to(device=device, dtype=torch.float32)
    scales_flat = state_dict["quantizer.scale"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    zp_flat = state_dict["quantizer.zero_point_float"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    quantized, scales_2d, zeros_2d = _quantize_rotated_weight(
        weight=weight,
        pairs=pairs,
        theta=theta,
        channel_scales=channel_scales_opt,
        scales_flat=scales_flat,
        zp_flat=zp_flat,
        bits=bits,
        group_size=group_size,
        device=device,
    )

    channel_scales = (1.0 / channel_scales_opt).to(torch.float16).cpu()
    if channel_scales.ndim == 1:
        channel_scales = channel_scales.unsqueeze(0)

    buffers: dict[str, torch.Tensor] = {
        **_to_awq_buffers(quantized, scales_2d, zeros_2d),
        "theta": theta.to(torch.float16).cpu(),
        "pairs": pairs.cpu(),
        "channel_scales": channel_scales,
    }
    if "bias" in state_dict and state_dict["bias"] is not None:
        buffers["bias"] = state_dict["bias"].to(torch.float16).cpu()

    return buffers, bits, group_size, int(theta.shape[0])


@torch.no_grad()
def _quantize_moe(
    state_dict: dict,
    device: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], int, int, int]:
    bits = int(_get_value(state_dict, "n_bits", "quantizer.n_bits"))
    group_size = int(_get_value(state_dict, "group_size", "quantizer.group_size"))

    gate_up = state_dict["gate_up_weight"].to(device=device, dtype=torch.float32)
    down = state_dict["down_weight"].to(device=device, dtype=torch.float32)
    num_experts, gate_up_out, gate_up_in = gate_up.shape
    _, down_out, down_in = down.shape
    if gate_up_in != down_out:
        raise ValueError(f"Unexpected MoE shapes: gate_up={tuple(gate_up.shape)} down={tuple(down.shape)}")

    gate_up_pairs = _stack_if_numbered(state_dict, "gate_up_pairs_grouped").to(device=device, dtype=torch.int16)
    gate_up_theta = _stack_if_numbered(state_dict, "gate_up_angles_grouped").to(device=device, dtype=torch.float32)
    gate_up_channel_scales = state_dict["gate_up_channel_scales"].to(device=device, dtype=torch.float32)
    gate_up_scales_flat = state_dict["gate_up_quantizer.scale"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    gate_up_zp_flat = (
        state_dict["gate_up_quantizer.zero_point_float"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    )

    down_pairs = _stack_if_numbered(state_dict, "down_pairs_grouped").to(device=device, dtype=torch.int16)
    down_theta = _stack_if_numbered(state_dict, "down_angles_grouped").to(device=device, dtype=torch.float32)
    down_channel_scales = state_dict["down_channel_scales"].to(device=device, dtype=torch.float32)
    down_scales_flat = state_dict["down_quantizer.scale"].to(device=device, dtype=torch.float32).reshape(-1, 1)
    down_zp_flat = state_dict["down_quantizer.zero_point_float"].to(device=device, dtype=torch.float32).reshape(-1, 1)

    gate_up_q, gate_up_sc, gate_up_zp = _quantize_rotated_weight(
        weight=gate_up.reshape(-1, gate_up_in),
        pairs=gate_up_pairs,
        theta=gate_up_theta,
        channel_scales=gate_up_channel_scales,
        scales_flat=gate_up_scales_flat,
        zp_flat=gate_up_zp_flat,
        bits=bits,
        group_size=group_size,
        device=device,
    )
    down_q, down_sc, down_zp = _quantize_rotated_weight(
        weight=down.reshape(-1, down_in),
        pairs=down_pairs,
        theta=down_theta,
        channel_scales=down_channel_scales,
        scales_flat=down_scales_flat,
        zp_flat=down_zp_flat,
        bits=bits,
        group_size=group_size,
        device=device,
    )

    half = gate_up_out // 2
    n_groups_gate = gate_up_in // group_size
    n_groups_down = down_in // group_size

    gate_up_q = gate_up_q.reshape(num_experts, gate_up_out, gate_up_in)
    gate_up_sc = gate_up_sc.reshape(num_experts, gate_up_out, n_groups_gate)
    gate_up_zp = gate_up_zp.reshape(num_experts, gate_up_out, n_groups_gate)

    down_q = down_q.reshape(num_experts, down_out, down_in)
    down_sc = down_sc.reshape(num_experts, down_out, n_groups_down)
    down_zp = down_zp.reshape(num_experts, down_out, n_groups_down)

    proj_buffers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for proj, q_rows, sc_rows, zp_rows in (
        ("gate_proj", gate_up_q[:, :half, :], gate_up_sc[:, :half, :], gate_up_zp[:, :half, :]),
        ("up_proj", gate_up_q[:, half:, :], gate_up_sc[:, half:, :], gate_up_zp[:, half:, :]),
        ("down_proj", down_q, down_sc, down_zp),
    ):
        proj_buffers[proj]["qweight"] = torch.stack(
            [_to_awq_buffers(q_rows[e], sc_rows[e], zp_rows[e])["qweight"] for e in range(num_experts)],
            dim=0,
        )
        proj_buffers[proj]["qzeros"] = torch.stack(
            [_to_awq_buffers(q_rows[e], sc_rows[e], zp_rows[e])["qzeros"] for e in range(num_experts)],
            dim=0,
        )
        proj_buffers[proj]["scales"] = torch.stack(
            [_to_awq_buffers(q_rows[e], sc_rows[e], zp_rows[e])["scales"] for e in range(num_experts)],
            dim=0,
        )

    gate_up_channel_scales_inv = (1.0 / gate_up_channel_scales).to(torch.float16).cpu()
    down_channel_scales_inv = (1.0 / down_channel_scales).to(torch.float16).cpu()
    if gate_up_channel_scales_inv.ndim == 1:
        gate_up_channel_scales_inv = gate_up_channel_scales_inv.unsqueeze(0)
    if down_channel_scales_inv.ndim == 1:
        down_channel_scales_inv = down_channel_scales_inv.unsqueeze(0)

    rotation_buffers = {
        "gate_up_weight_theta": gate_up_theta.to(torch.float16).cpu(),
        "gate_up_weight_pairs": gate_up_pairs.cpu(),
        "gate_up_weight_channel_scales": gate_up_channel_scales_inv,
        "down_weight_theta": down_theta.to(torch.float16).cpu(),
        "down_weight_pairs": down_pairs.cpu(),
        "down_weight_channel_scales": down_channel_scales_inv,
    }

    return dict(proj_buffers), rotation_buffers, bits, group_size, int(gate_up_theta.shape[0])


def _inject_quantized_moe_state_dict(
    state_dict: dict[str, torch.Tensor],
    moe_entries: list[tuple[int, str, dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    updated = dict(state_dict)

    for layer_idx, module_name, buffers, rotations in moe_entries:
        suffix = f"layers.{layer_idx}.{module_name}.gate_up_proj"
        matches = [k for k in updated if k.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"Could not uniquely resolve MoE prefix for layer={layer_idx} name={module_name}")
        base_prefix = matches[0].removesuffix(".gate_up_proj")

        updated.pop(f"{base_prefix}.gate_up_proj", None)
        updated.pop(f"{base_prefix}.down_proj", None)
        num_experts = buffers["gate_proj"]["qweight"].shape[0]
        for expert_id in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix_name in ("qweight", "qzeros", "scales"):
                    updated[f"{base_prefix}.{expert_id}.{proj}.{suffix_name}"] = buffers[proj][suffix_name][expert_id]
        for rot_name, rot_tensor in rotations.items():
            updated[f"{base_prefix}.{rot_name}"] = rot_tensor

    return updated


@torch.no_grad()
def _convert_real(
    model: torch.nn.Module,
    result_dir: Path,
) -> tuple[int, dict[str, Any] | None, dict[str, torch.Tensor]]:
    from paroquant.inference.backends.transformers.modules import RotateQuantizedLinear

    blocks = _get_blocks(model)
    count = 0
    bits = group_size = krot = 0
    moe_entries: list[tuple[int, str, dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]] = []

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

        for name, module in get_named_moe_experts(layer).items():
            pt_file = result_dir / f"{layer_idx}.{name}.pt"
            if not pt_file.exists():
                continue

            sd = torch.load(pt_file, weights_only=False, map_location="cpu")
            buffers_moe, rotations_moe, bits_moe, group_size_moe, krot_moe = _quantize_moe(sd, device="cuda")
            moe_entries.append((layer_idx, name, buffers_moe, rotations_moe))
            bits = bits or bits_moe
            group_size = group_size or group_size_moe
            krot = krot or krot_moe
            count += 1

        layer.cpu()
        torch.cuda.empty_cache()

    quant_config = {
        "quant_method": "paroquant",
        "bits": bits,
        "group_size": group_size,
        "krot": krot,
    }
    state_dict = _inject_quantized_moe_state_dict(model.state_dict(), moe_entries)
    return count, quant_config, state_dict


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
    save_state_dict: dict[str, torch.Tensor] | None = None
    if args.mode == "pseudo":
        count = _convert_pseudo(model, result_dir)
    else:
        count, quant_config, save_state_dict = _convert_real(model, result_dir)
        model.config.quantization_config = quant_config

    if count == 0:
        raise RuntimeError(f"No checkpoint files matched in {result_dir}")

    output_path = Path(args.output_path)
    source_non_md = _copy_source_non_md_files(source_dir, output_path)
    _remove_safetensor_files(output_path)

    model.save_pretrained(output_path, safe_serialization=True, state_dict=save_state_dict)

    # Restore original non-MD assets in case save_pretrained overwrote any.
    source_non_md = _copy_source_non_md_files(source_dir, output_path)
    _prune_non_md_extras(output_path, source_non_md)
    _write_config_json(source_dir, output_path, quant_config)

    if not any(_is_safetensor_related(p) for p in output_path.rglob("*") if p.is_file()):
        raise RuntimeError("No safetensors artifacts were generated.")

    print(f"Converted {count} layers ({args.mode}) → {output_path}")


if __name__ == "__main__":
    main()
