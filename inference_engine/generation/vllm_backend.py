import time
from typing import Callable, Dict, List, Optional

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from inference_engine.model_executor.models.qwen3_vllm import Qwen3ParoForCausalLM

from .base import UnifiedGenerator, GenerationParams, GenerationResult, GenerationStats


class VllmGenerator(UnifiedGenerator):
    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine, ModelRegistry

        self.model = model
        self.trust_remote_code = trust_remote_code
        self.enable_thinking = enable_thinking

        ModelRegistry.register_model("Qwen3ParoForCausalLM", Qwen3ParoForCausalLM)

        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        if hasattr(config, "paroquant_config"):
            hf_overrides = {"architectures": ["Qwen3ParoForCausalLM"]}
        else:
            hf_overrides = {}

        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model,
                trust_remote_code=trust_remote_code,
                hf_overrides=hf_overrides,
                compilation_config={},
                gpu_memory_utilization=gpu_memory_utilization,
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )

        try:
            generation_config = GenerationConfig.from_pretrained(model)
            self.default_temperature = generation_config.temperature
            self.default_top_p = generation_config.top_p
            self.default_top_k = generation_config.top_k
        except Exception:
            self.default_temperature = 0.6
            self.default_top_p = 1.0
            self.default_top_k = -1

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.tokenizer.chat_template is None:
            return messages[-1]["content"]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        params: GenerationParams,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        from vllm import SamplingParams

        prompt = self._build_prompt(messages)

        top_k = params.top_k if params.top_k is not None else self.default_top_k
        if top_k is None:
            top_k = -1

        sampling_params = SamplingParams(
            temperature=(
                params.temperature
                if params.temperature is not None
                else self.default_temperature
            ),
            top_p=(params.top_p if params.top_p is not None else self.default_top_p),
            top_k=top_k,
            max_tokens=params.max_new_tokens,
        )

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        last_token_count = 0
        token_events = []

        request_id = f"chat_request_{int(start_time * 1000)}"
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        last_output_text = ""
        async for request_output in results_generator:
            current_full_text = request_output.outputs[0].text
            new_text = current_full_text[len(last_output_text) :]

            if new_text and on_text:
                on_text(new_text)

            if first_token_time is None and new_text:
                first_token_time = time.perf_counter()

            token_ids = request_output.outputs[0].token_ids
            token_count = len(token_ids)

            if token_count > last_token_count:
                now = time.perf_counter() - start_time
                for token_id in token_ids[last_token_count:]:
                    token_events.append({"t": now, "token_id": token_id})
                last_token_count = token_count

            last_output_text = current_full_text

        end_time = time.perf_counter()
        total_duration = end_time - start_time
        generation_duration = end_time - (first_token_time or start_time)
        tokens_per_second = (
            token_count / generation_duration if generation_duration > 0 else 0.0
        )

        return GenerationResult(
            backend="vllm",
            prompt=prompt,
            output_text=last_output_text,
            stats=GenerationStats(
                token_count=token_count,
                total_time_s=total_duration,
                ttft_s=(first_token_time - start_time) if first_token_time else None,
                tokens_per_second=tokens_per_second,
                token_events=token_events,
            ),
        )
