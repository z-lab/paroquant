"""vLLM inference backend — async streaming via AsyncLLMEngine."""

import json
import time
from collections.abc import Callable
from pathlib import Path

from transformers import AutoTokenizer

from paroquant.inference.base import UnifiedGenerator, GenerationParams, GenerationResult, GenerationStats, build_prompt


def _read_quantization_config(model: str) -> dict | None:
    """Read quantization_config directly from config.json (avoids AutoConfig model-type errors)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model, "config.json") if not Path(model).is_dir() else f"{model}/config.json"
    with open(path) as f:
        qcfg = json.load(f).get("quantization_config")
    return qcfg if isinstance(qcfg, dict) and qcfg.get("quant_method") == "paroquant" else None


class Generator(UnifiedGenerator):
    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
        enforce_eager: bool = True,
        max_model_len: int | None = None,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers "paroquant" quantization

        self.enable_thinking = enable_thinking

        hf_overrides: dict = {}
        qcfg = _read_quantization_config(model)
        if qcfg:
            hf_overrides["quantization_config"] = {
                "quant_method": "paroquant",
                "bits": qcfg.get("bits", 4),
                "group_size": qcfg.get("group_size", 128),
                "krot": qcfg.get("krot", 8),
            }

        engine_kwargs: dict = dict(
            model=model,
            trust_remote_code=trust_remote_code,
            hf_overrides=hf_overrides,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len

        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def generate(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        from vllm import SamplingParams

        prompt = build_prompt(self.tokenizer, messages, self.enable_thinking)
        sampling = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k is not None else -1,
            max_tokens=params.max_new_tokens,
        )

        start = time.perf_counter()
        first_token_time = None
        token_count = 0
        last_text = ""
        request_id = f"req_{id(self)}_{int(start * 1e6)}"

        async for output in self.engine.generate(prompt, sampling, request_id):
            text = output.outputs[0].text
            delta = text[len(last_text):]
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                if on_text:
                    on_text(delta)
            token_count = len(output.outputs[0].token_ids)
            last_text = text

        end = time.perf_counter()
        gen_time = end - (first_token_time or start)

        return GenerationResult(
            backend="vllm",
            prompt=prompt,
            output_text=last_text,
            stats=GenerationStats(
                token_count=token_count,
                total_time_s=end - start,
                ttft_s=(first_token_time - start) if first_token_time else None,
                tokens_per_second=token_count / gen_time if gen_time > 0 else 0.0,
            ),
        )
