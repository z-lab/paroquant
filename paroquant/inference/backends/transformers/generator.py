"""Transformers inference backend — streaming via model.generate() + TextIteratorStreamer."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Thread

from transformers import AutoTokenizer, TextIteratorStreamer

from paroquant.inference.base import BaseGenerator, GenerationParams, GenerationResult, GenerationStats, build_prompt
from .load import load


class TransformersGenerator(BaseGenerator):
    def __init__(self, model: str, enable_thinking: bool = False, trust_remote_code: bool = True):
        self.enable_thinking = enable_thinking
        self.model = load(model)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def generate(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        prompt = build_prompt(self.tokenizer, messages, self.enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=self.model.generate, kwargs={
            **inputs,
            "streamer": streamer,
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
            "temperature": params.temperature if params.temperature > 0 else None,
            "top_k": params.top_k if params.top_k > 0 else None,
            "top_p": params.top_p,
            "cache_implementation": "static",
        })
        thread.start()

        start = time.perf_counter()
        first_token_time = None
        chunks: list[str] = []
        token_count = 0

        for text in streamer:
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(text)
                token_count += len(self.tokenizer.encode(text, add_special_tokens=False))
                if on_text:
                    on_text(text)

        thread.join()
        end = time.perf_counter()
        gen_time = end - (first_token_time or start)

        return GenerationResult(
            backend="transformers",
            prompt=prompt,
            output_text="".join(chunks),
            stats=GenerationStats(
                token_count=token_count,
                total_time_s=end - start,
                ttft_s=(first_token_time - start) if first_token_time else None,
                tokens_per_second=token_count / gen_time if gen_time > 0 else 0.0,
            ),
        )
