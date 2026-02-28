#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from huggingface_hub import snapshot_download
from inference_engine.utils.awq_conversion_utils import (
    convert_awq_llm_module_to_autoawq,
)


def convert_one_module(
    tensors: Dict[str, torch.Tensor],
    prefix: str,
    w_bit: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight_key = f"{prefix}qlinear.qweight"
    scales_key = f"{prefix}qlinear.scales"
    scaled_zeros_key = f"{prefix}qlinear.scaled_zeros"

    qweight = tensors[qweight_key]
    scales = tensors[scales_key]
    scaled_zeros = tensors[scaled_zeros_key]

    return convert_awq_llm_module_to_autoawq(
        qweight=qweight,
        scales=scales,
        scaled_zeros=scaled_zeros,
        w_bit=w_bit,
        group_size=group_size,
    )


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
