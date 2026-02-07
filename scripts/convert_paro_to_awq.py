#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from huggingface_hub import snapshot_download


def pack_cols(
    q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int
) -> torch.Tensor:
    assert q_w.shape == (size_k, size_n)
    pack_factor = 32 // num_bits
    assert size_n % pack_factor == 0

    orig_device = q_w.device
    q_w = q_w.cpu().numpy().astype(np.uint32)
    q_res = np.zeros((size_k, size_n // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << (num_bits * i)
    q_res = torch.from_numpy(q_res.astype(np.int32)).to(orig_device).contiguous()
    return q_res


def unpack_awq_llm_qweight(
    qweight: torch.Tensor, interleave: int, kstride: int
) -> torch.Tensor:
    """Inverse of qmodule.pack_intweight for AWQ-LLM int16 layout."""
    q = qweight.cpu().numpy().astype(np.uint16)
    n_div, k = q.shape

    v0 = q & 0xF
    v1 = (q >> 4) & 0xF
    v2 = (q >> 8) & 0xF
    v3 = (q >> 12) & 0xF
    packed = np.stack([v0, v1, v2, v3], axis=-1)  # (n_div, k, 4)

    # Undo interleaving (inverse of pack_intweight interleave block)
    packed = packed.reshape(n_div, k // kstride, kstride, interleave)
    packed = packed.reshape(n_div, k // kstride, interleave, kstride)
    packed = packed.transpose(0, 2, 1, 3)
    packed = packed.reshape(n_div * interleave, k)

    # Undo 8-weight reordering (inverse of pack_intweight step C)
    packed = packed.reshape(n_div * interleave, k // 32, 4, 2, 4)
    packed = packed.transpose(0, 1, 2, 4, 3)
    packed = packed.reshape(n_div * interleave, k // 32, 4, 8)
    packed = packed.reshape(n_div * interleave, k)

    # Undo 32-weight permutation (inverse of pack_intweight step B)
    packed = packed.reshape(n_div * interleave, k // 32, 4, 4, 2)
    packed = packed.transpose(0, 1, 3, 2, 4)
    packed = packed.reshape(n_div * interleave, k)

    return torch.from_numpy(packed.astype(np.int32)).to(qweight.device)


def convert_one_module(
    tensors: Dict[str, torch.Tensor],
    prefix: str,
    w_bit: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight_key = f"{prefix}qlinear.qweight"
    scales_key = f"{prefix}qlinear.scales"
    scaled_zeros_key = f"{prefix}qlinear.scaled_zeros"

    qweight = tensors[qweight_key]
    scales = tensors[scales_key]
    scaled_zeros = tensors[scaled_zeros_key]

    interleave = 4
    int16_pack_num = 16 // w_bit
    kstride = 64

    out_features = qweight.shape[0] * interleave
    in_features = (qweight.shape[1] // interleave) * int16_pack_num
    num_groups = in_features // group_size

    unpacked = unpack_awq_llm_qweight(qweight, interleave=interleave, kstride=kstride)
    # AutoAWQ uses interleaved ordering on the output dimension.
    if w_bit == 4:
        interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=unpacked.device)
    elif w_bit == 8:
        interleave = torch.tensor([0, 2, 1, 3], device=unpacked.device)
    else:
        raise ValueError(f"Unsupported w_bit: {w_bit}")
    unpacked = unpacked.view(out_features // interleave.numel(), interleave.numel(), -1)
    unpacked = unpacked[:, interleave, :].reshape(out_features, in_features)

    qweight_awq = pack_cols(
        unpacked.transpose(0, 1).contiguous(),
        num_bits=w_bit,
        size_k=in_features,
        size_n=out_features,
    )

    scales = scales[:num_groups, :]
    scaled_zeros = scaled_zeros[:num_groups, :]

    zeros = -scaled_zeros / scales
    zeros = torch.round(zeros).clamp_(0, 2**w_bit - 1).to(torch.int32)

    # AutoAWQ uses interleaved ordering on the output dimension.
    if w_bit == 4:
        interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=zeros.device)
    elif w_bit == 8:
        interleave = torch.tensor([0, 2, 1, 3], device=zeros.device)
    else:
        raise ValueError(f"Unsupported w_bit: {w_bit}")
    zeros = zeros.view(num_groups, out_features // interleave.numel(), -1)
    zeros = zeros[:, :, interleave].reshape(num_groups, out_features)

    qzeros = pack_cols(
        zeros.contiguous(),
        num_bits=w_bit,
        size_k=num_groups,
        size_n=out_features,
    )

    return qweight_awq, qzeros, scales


def convert_file(
    in_path: Path,
    out_path: Path,
    w_bit: int,
    group_size: int,
) -> None:
    tensors = load_file(in_path)
    out_tensors: Dict[str, torch.Tensor] = dict(tensors)

    prefixes = []
    for key in tensors.keys():
        if key.endswith("qlinear.qweight"):
            prefixes.append(key[: -len("qlinear.qweight")])

    for prefix in prefixes:
        qzeros_key = f"{prefix}qlinear.qzeros"
        scaled_zeros_key = f"{prefix}qlinear.scaled_zeros"
        scales_key = f"{prefix}qlinear.scales"
        qweight_key = f"{prefix}qlinear.qweight"

        if qzeros_key in tensors:
            continue
        if scales_key not in tensors or scaled_zeros_key not in tensors:
            continue

        qweight_awq, qzeros, scales = convert_one_module(
            tensors, prefix, w_bit=w_bit, group_size=group_size
        )
        out_tensors[qweight_key] = qweight_awq
        out_tensors[qzeros_key] = qzeros
        out_tensors[scales_key] = scales
        if scaled_zeros_key in out_tensors:
            del out_tensors[scaled_zeros_key]

    metadata = None
    with safe_open(in_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, out_path, metadata=metadata)


def resolve_input_dir(input_path: str) -> Path:
    path = Path(input_path)
    if path.exists() and path.is_dir():
        return path
    local_dir = snapshot_download(repo_id=input_path)
    return Path(local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ParoQuant AWQ-LLM weights to AutoAWQ-compatible tensors."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Local directory or HuggingFace repo id.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    in_dir = resolve_input_dir(args.input)
    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in in_dir.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, out_dir / item.name)

    files = sorted(in_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {in_dir}")

    for in_path in files:
        out_path = out_dir / in_path.name
        convert_file(
            in_path,
            out_path,
            w_bit=args.w_bit,
            group_size=args.group_size,
        )

    index_in = in_dir / "model.safetensors.index.json"
    index_out = out_dir / "model.safetensors.index.json"
    if index_in.exists():
        with index_in.open("r", encoding="utf-8") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        updated_map = dict(weight_map)
        for key, shard in weight_map.items():
            if key.endswith("qlinear.scaled_zeros"):
                new_key = key.replace("qlinear.scaled_zeros", "qlinear.qzeros")
                updated_map[new_key] = shard
                if key in updated_map:
                    del updated_map[key]

        index_data["weight_map"] = updated_map
        with index_out.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
