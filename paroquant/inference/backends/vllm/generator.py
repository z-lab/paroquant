from __future__ import annotations

import itertools
from collections.abc import AsyncIterator
from dataclasses import asdict

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers quantization config
from paroquant.inference.base import BaseGenerator, GenerationParams


class VllmGenerator(BaseGenerator):
    backend = "vllm"

    def __init__(self, model: str, **engine_kwargs):
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=model, **engine_kwargs))
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._req_counter = itertools.count()

    async def stream_generate(self, prompt: str, params: GenerationParams) -> AsyncIterator[str]:
        sampling = SamplingParams(**asdict(params))
        last_len = 0

        async for output in self.engine.generate(prompt, sampling, f"req-{next(self._req_counter)}"):
            text = output.outputs[0].text
            delta = text[last_len:]
            last_len = len(text)
            if delta:
                yield delta

    async def close(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None) or getattr(self.engine, "shutdown_background_loop", None)
        if shutdown:
            shutdown()
