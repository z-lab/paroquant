import asyncio
import time
import argparse

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm import ModelRegistry

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_engine.model_executor.models.qwen3_vllm import Qwen3ParoForCausalLM

ModelRegistry.register_model("Qwen3ParoForCausalLM", Qwen3ParoForCausalLM)


def _make_engine_args(model: str) -> AsyncEngineArgs:
    if "PARO" in model.upper():
        hf_overrides = {"architectures": ["Qwen3ParoForCausalLM"]}
    else:
        hf_overrides = {}

    return AsyncEngineArgs(
        model=model,
        hf_overrides=hf_overrides,
        compilation_config={},
    )


async def run_demo(model: str):
    engine_args = _make_engine_args(model)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    prompt = "Please explain what streaming output is in large language models."
    request_id = "demo_request_full"
    print(f"--- Prompt ---\n{prompt}\n")
    print("--- Output ---")

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    results_generator = engine.generate(prompt, sampling_params, request_id)

    last_output_text = ""
    async for request_output in results_generator:
        current_full_text = request_output.outputs[0].text
        new_text = current_full_text[len(last_output_text):]
        print(new_text, end="", flush=True)
        last_output_text = current_full_text

        if first_token_time is None and new_text:
            first_token_time = time.perf_counter()

        token_count = len(request_output.outputs[0].token_ids)

    end_time = time.perf_counter()

    total_duration = end_time - start_time
    generation_duration = end_time - (first_token_time or start_time)
    tps = token_count / generation_duration if generation_duration > 0 else 0

    print(f"\n\n" + "-" * 30)
    print("Performance Summary:")
    print(f"Total tokens generated: {token_count}")
    print(f"Total time: {total_duration:.2f}s")
    print(
        f"Time to first token (TTFT): {(first_token_time - start_time) * 1000:.2f}ms"
        if first_token_time
        else ""
    )
    print(f"Average generation speed (TPS): {tps:.2f} tokens/s")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model directory or model identifier from Hugging Face Hub.",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(args.model))
