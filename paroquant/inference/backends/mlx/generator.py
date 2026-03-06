from __future__ import annotations

from collections.abc import AsyncIterator

import mlx.core as mx
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from paroquant.inference.base import BaseGenerator, GenerationParams

from .load import load


class MlxGenerator(BaseGenerator):
    backend = "mlx"

    def __init__(self, model: str):
        self.model, self.processor, self.is_vlm = load(model)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    async def _stream(self, prompt: str, params: GenerationParams) -> AsyncIterator[str]:
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)
        logits_processors = make_logits_processors(
            repetition_penalty=params.repetition_penalty if params.repetition_penalty > 1.0 else None,
        )

        if not self.is_vlm:
            from mlx_lm.generate import stream_generate

            for resp in stream_generate(
                self.model,
                self.processor,
                prompt,
                max_tokens=params.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                if resp.text:
                    yield resp.text
            return

        from mlx_vlm import stream_generate

        input_ids = None if not prompt else mx.array(self.tokenizer.encode(prompt))[None]
        for resp in stream_generate(
            self.model,
            self.processor,
            prompt,
            input_ids=input_ids,
            max_tokens=params.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            text = resp.text if hasattr(resp, "text") else resp
            if text:
                yield text
