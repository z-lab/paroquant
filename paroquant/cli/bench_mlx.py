"""Benchmark ParoQuant vs MLX community quantized checkpoints.

Example:
    python -m paroquant.cli.bench_mlx \\
        --sizes 0.8B,2B,4B,9B \\
        --max-tokens 64 \\
        --warmup-runs 3 \\
        --runs 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_model, load_tokenizer

from paroquant.inference.backends.mlx.load import load as load_paro

MODEL_PAIRS = {
    "0.8B": {
        "paro": "z-lab/Qwen3.5-0.8B-PARO",
        "mlx": "mlx-community/Qwen3.5-0.8B-4bit",
    },
    "2B": {
        "paro": "z-lab/Qwen3.5-2B-PARO",
        "mlx": "mlx-community/Qwen3.5-2B-4bit",
    },
    "4B": {
        "paro": "z-lab/Qwen3.5-4B-PARO",
        "mlx": "mlx-community/Qwen3.5-4B-4bit",
    },
    "9B": {
        "paro": "z-lab/Qwen3.5-9B-PARO",
        "mlx": "mlx-community/Qwen3.5-9B-4bit",
    },
}


def _trimmed_mean(values: list[float]) -> float | None:
    """Mean after dropping one min and one max value. Falls back to plain mean for <= 2 values."""
    if len(values) <= 2:
        return statistics.mean(values) if values else None
    trimmed = sorted(values)[1:-1]
    return statistics.mean(trimmed)


@dataclass
class BenchResult:
    ok: bool
    label: str
    repo: str
    error: str | None
    load_seconds: float | None
    load_peak_gb: float | None
    load_active_gb: float | None
    load_cache_gb: float | None
    gen_peak_gb: float | None
    tps_runs: list[float] = field(default_factory=list)
    tps_avg: float | None = None
    tps_std: float | None = None
    tps_median: float | None = None
    tps_trimmed: float | None = None


def _load_native(repo: str):
    model_path = Path(snapshot_download(repo))
    model, _ = load_model(model_path, lazy=False, strict=False)
    tokenizer = load_tokenizer(model_path)
    return model, tokenizer


def _load_paro(repo: str):
    model, processor, _is_vlm = load_paro(repo)
    return model, processor


def _load_and_warmup(
    repo: str,
    loader: Callable[[str], tuple[object, object]],
    *,
    prompt: str,
    max_tokens: int,
    warmup_runs: int,
    temp: float,
    top_p: float,
    seed: int,
):
    """Load a model, run warmup generations, and return (model, tokenizer, load_stats)."""
    mx.random.seed(seed)
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    model, tokenizer = loader(repo)
    load_seconds = time.perf_counter() - t0
    load_peak = mx.get_peak_memory() / 1e9
    load_active = mx.get_active_memory() / 1e9
    load_cache = mx.get_cache_memory() / 1e9

    sampler = make_sampler(temp=temp, top_p=top_p)
    mx.reset_peak_memory()

    for _ in range(warmup_runs):
        for _ in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
            pass

    return (
        model,
        tokenizer,
        {
            "load_seconds": load_seconds,
            "load_peak_gb": load_peak,
            "load_active_gb": load_active,
            "load_cache_gb": load_cache,
        },
    )


def _measure_one(model, tokenizer, *, prompt, max_tokens, temp, top_p) -> float:
    """Run one generation and return tok/s."""
    sampler = make_sampler(temp=temp, top_p=top_p)
    last = None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        last = response
    if last is None:
        raise RuntimeError("No generation response received.")
    return float(last.generation_tps)


def _make_result(label: str, repo: str, load_stats: dict, tps_runs: list[float]) -> BenchResult:
    return BenchResult(
        ok=True,
        label=label,
        repo=repo,
        error=None,
        load_seconds=load_stats["load_seconds"],
        load_peak_gb=load_stats["load_peak_gb"],
        load_active_gb=load_stats["load_active_gb"],
        load_cache_gb=load_stats["load_cache_gb"],
        gen_peak_gb=mx.get_peak_memory() / 1e9,
        tps_runs=tps_runs,
        tps_avg=statistics.mean(tps_runs),
        tps_std=statistics.pstdev(tps_runs),
        tps_median=statistics.median(tps_runs),
        tps_trimmed=_trimmed_mean(tps_runs),
    )


def _make_error_result(label: str, repo: str, exc: Exception) -> BenchResult:
    return BenchResult(
        ok=False,
        label=label,
        repo=repo,
        error=f"{type(exc).__name__}: {exc}",
        load_seconds=None,
        load_peak_gb=None,
        load_active_gb=None,
        load_cache_gb=None,
        gen_peak_gb=None,
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark ParoQuant vs MLX community quantized models")
    parser.add_argument(
        "--sizes",
        type=str,
        default="0.8B,2B,4B,9B",
        help="Comma-separated subset of model sizes",
    )
    parser.add_argument("--prompt", type=str, default="Kernel dispatch check:")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path for raw benchmark JSON",
    )
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    unknown = [s for s in sizes if s not in MODEL_PAIRS]
    if unknown:
        raise ValueError(f"Unknown sizes: {unknown}. Available: {sorted(MODEL_PAIRS.keys())}")

    all_rows: list[dict] = []
    gen_params = dict(prompt=args.prompt, max_tokens=args.max_tokens, temp=args.temp, top_p=args.top_p)

    print(
        f"Benchmark config: max_tokens={args.max_tokens}, warmup_runs={args.warmup_runs}, "
        f"runs={args.runs}, temp={args.temp}, top_p={args.top_p}"
    )

    for size in sizes:
        pair = MODEL_PAIRS[size]
        print(f"\n=== {size} ===")
        print(f"Paro: {pair['paro']}")
        print(f"MLX : {pair['mlx']}")

        paro_res: BenchResult | None = None
        mlx_res: BenchResult | None = None

        try:
            paro_model, paro_tok, paro_stats = _load_and_warmup(
                pair["paro"],
                _load_paro,
                warmup_runs=args.warmup_runs,
                seed=args.seed,
                **gen_params,
            )
        except Exception as exc:
            paro_res = _make_error_result("paro", pair["paro"], exc)
            paro_model = paro_tok = paro_stats = None

        try:
            mlx_model, mlx_tok, mlx_stats = _load_and_warmup(
                pair["mlx"],
                _load_native,
                warmup_runs=args.warmup_runs,
                seed=args.seed,
                **gen_params,
            )
        except Exception as exc:
            mlx_res = _make_error_result("mlx", pair["mlx"], exc)
            mlx_model = mlx_tok = mlx_stats = None

        paro_tps: list[float] = []
        mlx_tps: list[float] = []
        mx.reset_peak_memory()

        for i in range(args.runs):
            if paro_model is not None and paro_res is None:
                try:
                    paro_tps.append(_measure_one(paro_model, paro_tok, **gen_params))
                except Exception as exc:
                    paro_res = _make_error_result("paro", pair["paro"], exc)

            if mlx_model is not None and mlx_res is None:
                try:
                    mlx_tps.append(_measure_one(mlx_model, mlx_tok, **gen_params))
                except Exception as exc:
                    mlx_res = _make_error_result("mlx", pair["mlx"], exc)

        if paro_res is None:
            paro_res = _make_result("paro", pair["paro"], paro_stats, paro_tps)
        if mlx_res is None:
            mlx_res = _make_result("mlx", pair["mlx"], mlx_stats, mlx_tps)

        if not paro_res.ok:
            print(f"[paro] failed: {paro_res.error}")
        else:
            print(
                f"[paro] trimmed={_fmt(paro_res.tps_trimmed)} avg={_fmt(paro_res.tps_avg)} "
                f"std={_fmt(paro_res.tps_std)} runs={[round(v, 1) for v in paro_res.tps_runs]}"
            )
        if not mlx_res.ok:
            print(f"[mlx ] failed: {mlx_res.error}")
        else:
            print(
                f"[mlx ] trimmed={_fmt(mlx_res.tps_trimmed)} avg={_fmt(mlx_res.tps_avg)} "
                f"std={_fmt(mlx_res.tps_std)} runs={[round(v, 1) for v in mlx_res.tps_runs]}"
            )

        ratio = (
            (paro_res.tps_trimmed / mlx_res.tps_trimmed) if paro_res.ok and mlx_res.ok and mlx_res.tps_trimmed else None
        )
        all_rows.append(
            {
                "size": size,
                "paro": asdict(paro_res),
                "mlx": asdict(mlx_res),
                "paro_over_mlx": ratio,
            }
        )

        del paro_model, paro_tok, mlx_model, mlx_tok
        mx.clear_cache()

    print("\n=== Summary ===")
    print("| Size | Paro tok/s | MLX tok/s | Paro/MLX | Paro gen peak | MLX gen peak |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in all_rows:
        size = row["size"]
        paro = row["paro"]
        mlx = row["mlx"]
        ratio = row["paro_over_mlx"]
        print(
            f"| {size} | {_fmt(paro['tps_trimmed'])} | {_fmt(mlx['tps_trimmed'])} | "
            f"{_fmt(ratio)} | {_fmt(paro['gen_peak_gb'])} GB | {_fmt(mlx['gen_peak_gb'])} GB |"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.write_text(json.dumps(all_rows, indent=2))
        print(f"\nSaved JSON results to {out}")


if __name__ == "__main__":
    main()
