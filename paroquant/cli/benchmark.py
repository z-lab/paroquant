"""Benchmark ParoQuant generation throughput across backends.

    python -m paroquant.cli.benchmark --model z-lab/Qwen3.5-0.8B-PARO-mm --backend mlx
    python -m paroquant.cli.benchmark --model z-lab/Qwen3.5-0.8B-PARO-mm --backend vllm
    python -m paroquant.cli.benchmark --model z-lab/Qwen3.5-0.8B-PARO-mm --backend transformers
"""

import argparse
import asyncio
import time

from paroquant.inference import create_generator, GenerationParams


async def bench(gen, prompt: str, max_tokens: int, warmup: int, runs: int):
    params = GenerationParams(max_tokens=max_tokens, temperature=0.0)
    messages = [{"role": "user", "content": prompt}]

    for _ in range(warmup):
        await gen.generate(messages, params)

    times, tokens = [], []
    for _ in range(runs):
        start = time.perf_counter()
        result = await gen.generate(messages, params)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        tokens.append(result.stats.num_tokens)

    avg_tokens = sum(tokens) / len(tokens)
    avg_time = sum(times) / len(times)
    avg_tps = avg_tokens / avg_time if avg_time > 0 else 0
    return avg_tokens, avg_time, avg_tps


async def main():
    parser = argparse.ArgumentParser(description="Benchmark generation throughput")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", type=str, default="mlx", choices=["mlx", "vllm", "transformers"])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="Write a short essay about the history of computing.")
    args = parser.parse_args()

    print(f"Loading {args.model} with {args.backend} backend...")
    start = time.perf_counter()
    gen = create_generator(args.backend, args.model)
    load_time = time.perf_counter() - start
    print(f"Loaded in {load_time:.1f}s")

    print(f"Benchmarking: {args.warmup} warmup + {args.runs} runs, {args.max_tokens} tokens each")
    avg_tokens, avg_time, avg_tps = await bench(gen, args.prompt, args.max_tokens, args.warmup, args.runs)
    print(f"  Avg tokens: {avg_tokens:.0f}")
    print(f"  Avg time:   {avg_time:.2f}s")
    print(f"  Throughput:  {avg_tps:.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
