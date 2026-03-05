from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers quantization config
from paroquant.inference.base import (
    BaseGenerator,
    GenerationParams,
    GenerationResult,
    GenerationStats,
    build_prompt,
)


class VllmGenerator(BaseGenerator):
    def __init__(self, model: str, enable_thinking: bool = False, **engine_kwargs):
        self.enable_thinking = enable_thinking
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=model, **engine_kwargs))
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    async def generate(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        prompt = build_prompt(self.tokenizer, messages, self.enable_thinking)
        sampling = SamplingParams(**asdict(params))

        start = time.perf_counter()
        first_token_time = None
        num_tokens = 0
        last_text = ""

        async for output in self.engine.generate(prompt, sampling, f"req-{time.monotonic_ns()}"):
            text = output.outputs[0].text
            delta = text[len(last_text) :]
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                if on_text:
                    on_text(delta)
            num_tokens = len(output.outputs[0].token_ids)
            last_text = text

        end = time.perf_counter()
        gen_time = end - (first_token_time or start)

        return GenerationResult(
            backend="vllm",
            prompt=prompt,
            output_text=last_text,
            stats=GenerationStats(
                num_tokens=num_tokens,
                latency=end - start,
                ttft=(first_token_time - start) if first_token_time else None,
                tps=num_tokens / gen_time if gen_time > 0 else 0.0,
            ),
        )

    async def close(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None) or getattr(self.engine, "shutdown_background_loop", None)
        if shutdown:
            shutdown()
