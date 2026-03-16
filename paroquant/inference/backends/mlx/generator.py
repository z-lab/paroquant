from __future__ import annotations

from collections.abc import AsyncIterator

import mlx.core as mx
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from paroquant.inference.base import BaseGenerator, GenerationParams

from .load import load


class MlxGenerator(BaseGenerator):
    backend = "mlx"

    def __init__(self, model: str):
        self.model, self.processor, self.is_vlm = load(model, force_text=True)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    async def stream_generate(self, prompt: str, params: GenerationParams) -> AsyncIterator[str]:
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)
        logits_processors = make_logits_processors(
            repetition_penalty=params.repetition_penalty if params.repetition_penalty > 1.0 else None,
        )

        kwargs = dict(max_tokens=params.max_tokens, sampler=sampler, logits_processors=logits_processors)

        if self.is_vlm:
            from mlx_vlm import stream_generate

            if prompt:
                kwargs["input_ids"] = mx.array(self.tokenizer.encode(prompt))[None]
        else:
            from mlx_lm.generate import stream_generate

        for resp in stream_generate(self.model, self.processor, prompt, **kwargs):
            text = resp.text if hasattr(resp, "text") else resp
            if text:
                yield text
