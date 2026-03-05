"""MLX inference backend — streaming generation on Apple Silicon (LLM + VLM)."""

from __future__ import annotations

import time
from collections.abc import Callable

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from paroquant.inference.base import BaseGenerator, GenerationParams, GenerationResult, GenerationStats, build_prompt

from .load import load


class MlxGenerator(BaseGenerator):
    def __init__(self, model: str, enable_thinking: bool = False):
        self.enable_thinking = enable_thinking
        self.model, self.processor, self.is_vlm = load(model)

    def _stream(self, prompt: str, params: GenerationParams, image=None):
        """Return a token stream iterator, dispatching to mlx_lm or mlx_vlm."""
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)

        if not self.is_vlm:
            from mlx_lm.generate import stream_generate

            return stream_generate(
                self.model,
                self.processor,
                prompt,
                max_tokens=params.max_tokens,
                sampler=sampler,
            )

        from mlx_vlm import stream_generate

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        input_ids = None if image else mx.array(tokenizer.encode(prompt))[None]
        return stream_generate(
            self.model,
            self.processor,
            prompt,
            image=image,
            input_ids=input_ids,
            max_tokens=params.max_tokens,
            sampler=sampler,
            skip_special_tokens=True,
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
        **kwargs,
    ) -> GenerationResult:
        prompt = build_prompt(self.processor, messages, self.enable_thinking)

        start = time.perf_counter()
        first_token_time = None
        chunks: list[str] = []
        num_tokens = 0

        for resp in self._stream(prompt, params, image=kwargs.get("image")):
            text = resp.text if hasattr(resp, "text") else resp
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(text)
                num_tokens += 1
                if on_text:
                    on_text(text)

        end = time.perf_counter()
        gen_time = end - (first_token_time or start)

        return GenerationResult(
            backend="mlx",
            prompt=prompt,
            output_text="".join(chunks),
            stats=GenerationStats(
                num_tokens=num_tokens,
                latency=end - start,
                ttft=(first_token_time - start) if first_token_time else None,
                tps=num_tokens / gen_time if gen_time > 0 else 0.0,
            ),
        )
