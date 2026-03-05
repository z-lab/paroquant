"""MLX inference backend — streaming generation via mlx_lm on Apple Silicon."""

from __future__ import annotations

import time
from collections.abc import Callable

from paroquant.inference.base import UnifiedGenerator, GenerationParams, GenerationResult, GenerationStats, build_prompt


class Generator(UnifiedGenerator):
    def __init__(self, model: str, enable_thinking: bool = False):
        from .load import load

        self.enable_thinking = enable_thinking
        self.model, self.tokenizer = load(model)

    async def generate(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = build_prompt(self.tokenizer, messages, self.enable_thinking)
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)

        start = time.perf_counter()
        first_token_time = None
        chunks: list[str] = []
        token_count = 0

        for resp in stream_generate(
            self.model, self.tokenizer, prompt,
            max_tokens=params.max_new_tokens, sampler=sampler,
        ):
            if resp.text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(resp.text)
                token_count += 1
                if on_text:
                    on_text(resp.text)

        end = time.perf_counter()
        gen_time = end - (first_token_time or start)

        return GenerationResult(
            backend="mlx",
            prompt=prompt,
            output_text="".join(chunks),
            stats=GenerationStats(
                token_count=token_count,
                total_time_s=end - start,
                ttft_s=(first_token_time - start) if first_token_time else None,
                tokens_per_second=token_count / gen_time if gen_time > 0 else 0.0,
            ),
        )
