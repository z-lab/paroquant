from __future__ import annotations

from collections.abc import AsyncIterator
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    TextIteratorStreamer,
)

import paroquant.inference.backends.transformers.quantizer  # noqa: F401 — registers HfQuantizer
from paroquant.inference.base import BaseGenerator, GenerationParams

_LOAD_KWARGS = dict(
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    device_map="cuda",
)


def _load_model(path: str):
    try:
        return AutoModelForImageTextToText.from_pretrained(path, **_LOAD_KWARGS)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(path, **_LOAD_KWARGS)


class TransformersGenerator(BaseGenerator):
    backend = "transformers"

    def __init__(self, model: str):
        with torch.no_grad():
            self.model = _load_model(model)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    async def _stream(self, prompt: str, params: GenerationParams) -> AsyncIterator[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=self.model.generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": params.max_tokens,
                "do_sample": params.temperature > 0,
                "temperature": params.temperature if params.temperature > 0 else None,
                "top_k": params.top_k if params.top_k > 0 else None,
                "top_p": params.top_p,
                "repetition_penalty": params.repetition_penalty if params.repetition_penalty > 1.0 else None,
            },
        )
        thread.start()

        for text in streamer:
            if text:
                yield text

        thread.join()
