"""Generator: streaming generation via model.generate() + TextIteratorStreamer."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Thread

from transformers import AutoTokenizer, TextIteratorStreamer

from paroquant.inference.base import BaseGenerator, GenerationParams, GenerationResult, GenerationStats, build_prompt
from .load import load


class TransformersGenerator(BaseGenerator):
    def __init__(self, model: str, trust_remote_code: bool = True, enable_thinking: bool = False):
        self.model = load(model)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.enable_thinking = enable_thinking

    async def generate(self, messages: list[dict[str, str]], params: GenerationParams,
                       on_text: Callable[[str], None] | None = None) -> GenerationResult:
        prompt = build_prompt(self.tokenizer, messages, self.enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.temperature > 0,
            "temperature": params.temperature if params.temperature > 0 else None,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "cache_implementation": "static",
        }

        start_time = time.perf_counter()
        first_token_time = None
        output_chunks: list[str] = []
        token_count = 0

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_chunks.append(text)
                token_count += len(self.tokenizer.encode(text, add_special_tokens=False))
                if on_text:
                    on_text(text)

        thread.join()

        end_time = time.perf_counter()
        total_time = end_time - start_time
        gen_time = end_time - (first_token_time or start_time)

        return GenerationResult(
            backend="transformers",
            prompt=prompt,
            output_text="".join(output_chunks),
            stats=GenerationStats(
                token_count=token_count,
                total_time_s=total_time,
                ttft_s=(first_token_time - start_time) if first_token_time else None,
                tokens_per_second=token_count / gen_time if gen_time > 0 else 0.0,
            ),
        )
