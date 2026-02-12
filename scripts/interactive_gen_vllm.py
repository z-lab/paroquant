import asyncio
import time
import argparse
import json
from typing import Optional
from transformers import AutoTokenizer, GenerationConfig

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
        trust_remote_code=True,
        hf_overrides=hf_overrides,
        compilation_config={},
        gpu_memory_utilization=0.8,
    )


async def run_demo(model: str, recording_output: Optional[str], max_tokens: int):
    engine_args = _make_engine_args(model)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    generation_config = GenerationConfig.from_pretrained(model)
    sampling_params = SamplingParams(
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        max_tokens=max_tokens,
    )

    request_index = 0
    while True:
        try:
            user_prompt = input("What is your prompt? ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_prompt.strip().lower() == "quit":
            break
        if not user_prompt.strip():
            continue

        messages = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        request_id = f"demo_request_{request_index}"
        request_index += 1
        print(f"--- Raw Prompt Sent to Engine ---\n{prompt}\n")
        print("--- Output ---")

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        last_token_count = 0
        token_events = []

        results_generator = engine.generate(prompt, sampling_params, request_id)

        last_output_text = ""
        async for request_output in results_generator:
            current_full_text = request_output.outputs[0].text
            new_text = current_full_text[len(last_output_text) :]

            print(new_text, end="", flush=True)
            last_output_text = current_full_text

            if first_token_time is None and new_text:
                first_token_time = time.perf_counter()

            token_ids = request_output.outputs[0].token_ids
            token_count = len(token_ids)
            if token_count > last_token_count:
                now = time.perf_counter() - start_time
                new_token_ids = token_ids[last_token_count:]
                for token_id in new_token_ids:
                    token_events.append({"t": now, "token_id": token_id})
                last_token_count = token_count

        end_time = time.perf_counter()

        total_duration = end_time - start_time
        generation_duration = end_time - (first_token_time or start_time)
        tps = token_count / generation_duration if generation_duration > 0 else 0

        print(f"\n\n" + "-" * 30)
        print("Performance Summary:")
        print(f"Total tokens generated: {token_count}")
        print(f"Total time: {total_duration:.2f}s")
        if first_token_time:
            print(
                f"Time to first token (TTFT): {(first_token_time - start_time) * 1000:.2f}ms"
            )
        print(f"Average generation speed (TPS): {tps:.2f} tokens/s")
        print("-" * 30)

        if recording_output:
            record = {
                "model": model,
                "tokenizer": model,
                "prompt": user_prompt,
                "raw_prompt": prompt,
                "sampling_params": {
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "max_tokens": sampling_params.max_tokens,
                },
                "token_events": token_events,
                "token_count": token_count,
                "total_time_s": total_duration,
                "ttft_s": (first_token_time - start_time) if first_token_time else None,
                "output_text": last_output_text,
            }
            with open(recording_output, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model directory or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32768, help="Maximum tokens to generate."
    )
    parser.add_argument(
        "--recording-output",
        type=str,
        default=None,
        help="Path to save the recording of the demo session.",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(args.model, args.recording_output, args.max_tokens))
