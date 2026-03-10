from __future__ import annotations

import argparse
import json
from pathlib import Path

from paroquant.export import DEFAULT_GGUF_QUANTS, DEFAULT_TARGETS, ExportOptions, run_export


def _csv_to_tuple(raw: str) -> tuple[str, ...]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return tuple(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ParoQuant checkpoints for MLX / GGUF runtimes.")
    parser.add_argument("--model", required=True, type=str, help="HF repo id or local checkpoint path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for generated artifacts.")
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        type=str,
        help=f"Comma-separated targets. Default: {','.join(DEFAULT_TARGETS)}",
    )
    parser.add_argument(
        "--text-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export text-only compatibility model (default: true).",
    )
    parser.add_argument(
        "--gguf-quants",
        default=",".join(DEFAULT_GGUF_QUANTS),
        type=str,
        help=f"Comma-separated GGUF quant levels. Default: {','.join(DEFAULT_GGUF_QUANTS)}",
    )
    parser.add_argument("--llama-cpp-dir", type=Path, default=None, help="Path to llama.cpp checkout (GGUF targets).")
    parser.add_argument(
        "--hf-publish-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepare publish-ready output layout (default: true).",
    )
    parser.add_argument("--mlx-q-bits", type=int, default=4, help="MLX quantization bits.")
    parser.add_argument("--mlx-q-group-size", type=int, default=128, help="MLX quantization group size.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    opts = ExportOptions(
        model=args.model,
        output_dir=args.output_dir,
        targets=_csv_to_tuple(args.targets),
        text_only=args.text_only,
        gguf_quants=_csv_to_tuple(args.gguf_quants),
        llama_cpp_dir=args.llama_cpp_dir,
        hf_publish_layout=args.hf_publish_layout,
        mlx_q_bits=args.mlx_q_bits,
        mlx_q_group_size=args.mlx_q_group_size,
    )

    result = run_export(opts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
