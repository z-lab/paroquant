"""Transformers inference backend — streaming via model.generate() + TextIteratorStreamer."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    TextIteratorStreamer,
)

import paroquant.inference.backends.transformers.quantizer  # noqa: F401 — registers HfQuantizer
from paroquant.inference.base import (
    BaseGenerator,
    GenerationParams,
    GenerationResult,
    GenerationStats,
    build_prompt,
)

_LOAD_KWARGS = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa", device_map="cuda")


def _load_model(path: str):
    """Load model, trying VLM class first then falling back to CausalLM."""
    try:
        return AutoModelForImageTextToText.from_pretrained(path, **_LOAD_KWARGS)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(path, **_LOAD_KWARGS)


class TransformersGenerator(BaseGenerator):
    def __init__(self, model: str, enable_thinking: bool = False):
        self.enable_thinking = enable_thinking
        with torch.no_grad():
            self.model = _load_model(model)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)

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
        })
        thread.start()

        start = time.perf_counter()
        first_token_time = None
        chunks: list[str] = []

        for text in streamer:
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(text)
                if on_text:
                    on_text(text)

        thread.join()
        end = time.perf_counter()
        gen_time = end - (first_token_time or start)
        output_text = "".join(chunks)
        num_tokens = len(self.tokenizer.encode(output_text, add_special_tokens=False))

        return GenerationResult(
            backend="transformers",
            prompt=prompt,
            output_text=output_text,
            stats=GenerationStats(
                num_tokens=num_tokens,
                latency=end - start,
                ttft=(first_token_time - start) if first_token_time else None,
                tps=num_tokens / gen_time if gen_time > 0 else 0.0,
            ),
        )
