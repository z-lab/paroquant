import json
import os
import time
from typing import Callable, Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download
from torch.nn.attention import SDPBackend
from transformers import AutoConfig, AutoTokenizer, Qwen2Config
from transformers.utils import CONFIG_NAME

from inference_engine.model_executor.models.cache_utils import StaticCache
from inference_engine.model_executor.models.llama import LlamaForCausalLM
from inference_engine.model_executor.models.llama_fp16 import LlamaForCausalLMFP16
from inference_engine.model_executor.models.qwen3 import Qwen3ForCausalLM
from inference_engine.model_executor.models.qwen3_fp16 import Qwen3ForCausalLMFP16

from .base import UnifiedGenerator, GenerationParams, GenerationResult, GenerationStats


torch.set_grad_enabled(False)


def _load_config_dict(path_or_repo: str):
    if os.path.isdir(path_or_repo):
        cfg_path = os.path.join(path_or_repo, CONFIG_NAME)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config.json not found under: {path_or_repo}")
        with open(cfg_path, "r", encoding="utf-8") as file:
            return json.load(file)

    cfg_path = hf_hub_download(path_or_repo, CONFIG_NAME)
    with open(cfg_path, "r", encoding="utf-8") as file:
        return json.load(file)


def model_from_hf_path(path: str, empty_model: bool = False):
    try:
        config = AutoConfig.from_pretrained(path)
        is_quantized = hasattr(config, "paroquant_config")
    except ValueError as error:
        if "qwen3" not in str(error).lower():
            raise
        config_dict = _load_config_dict(path)
        config = Qwen2Config(**config_dict)
        is_quantized = hasattr(config, "paroquant_config")

    model_type = config.model_type
    if is_quantized or empty_model:
        if model_type == "llama":
            model_cls = LlamaForCausalLM
        elif model_type == "qwen3":
            model_cls = Qwen3ForCausalLM
        else:
            raise ValueError(f"Unsupported quantized model type: {model_type}")
    else:
        if model_type == "llama":
            model_cls = LlamaForCausalLMFP16
        elif model_type == "qwen3":
            model_cls = Qwen3ForCausalLMFP16
        else:
            raise ValueError(f"Unsupported FP16 model type: {model_type}")

    if empty_model:
        model = model_cls(config)
        model = model.to(dtype=torch.float16)
        model.to("cuda")
        model.eval()
    else:
        model = model_cls.from_pretrained(
            path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map="cuda",
        )

    return model, path


def multinomial_sample_one_no_sync(probs_sort: torch.Tensor):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = values.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("inf"), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

    return torch.nn.functional.softmax(logits, dim=-1)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
):
    probs = logits_to_probs(
        logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p
    )
    return multinomial_sample_one_no_sync(probs)


def decode_one_token(
    model,
    cur_token: torch.Tensor,
    past_kv,
    cache_position: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
):
    logits = model(cur_token, past_key_values=past_kv, cache_position=cache_position)[0]
    new_token = sample_next_token(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return new_token, logits


class TransformersGenerator(UnifiedGenerator):
    def __init__(
        self,
        model: str,
        compile_decode: bool = False,
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
    ):
        self.model, self.model_str = model_from_hf_path(model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_str,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.enable_thinking = enable_thinking
        self._period_id = self.tokenizer.encode(".", add_special_tokens=False)[-1]
        self._decode_impl = decode_one_token
        if compile_decode:
            self._decode_impl = torch.compile(
                self._decode_impl,
                mode="max-autotune",
                fullgraph=True,
            )

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
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(0)

        _, seq_length = inputs["input_ids"].shape
        max_cache_len = max(
            seq_length + params.max_new_tokens + 8, 2 * params.max_new_tokens
        )
        past_kv = StaticCache(
            self.model.config,
            1,
            max_cache_len,
            device=0,
            dtype=self.model.dtype,
        )
        cache_position = torch.arange(seq_length, device=0)

        start_time = time.perf_counter()
        first_token_time = None

        logits = self.model(
            **inputs,
            past_key_values=past_kv,
            cache_position=cache_position,
        )[0]

        next_token = sample_next_token(
            logits,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
        )

        generated_token_ids: List[int] = []
        token_events = []
        output_chunks: List[str] = []

        decode_position = torch.tensor([seq_length + 1], device=0)

        for _ in range(params.max_new_tokens):
            token_id = int(next_token[0].item())
            now = time.perf_counter() - start_time

            generated_token_ids.append(token_id)
            token_events.append({"t": now, "token_id": token_id})

            if token_id == self.tokenizer.eos_token_id:
                break

            piece = self.tokenizer.decode([self._period_id, token_id])[1:]
            output_chunks.append(piece)

            if first_token_time is None and piece:
                first_token_time = time.perf_counter()

            if on_text and piece:
                on_text(piece)

            with torch.nn.attention.sdpa_kernel(
                backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
            ):
                next_token, _ = self._decode_impl(
                    self.model,
                    next_token.clone(),
                    past_kv,
                    decode_position,
                    params.temperature,
                    params.top_k,
                    params.top_p,
                )
            decode_position += 1

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_duration = end_time - start_time
        generation_duration = end_time - (first_token_time or start_time)
        token_count = len(generated_token_ids)
        tokens_per_second = (
            token_count / generation_duration if generation_duration > 0 else 0.0
        )

        return GenerationResult(
            backend="transformers",
            prompt=prompt,
            output_text="".join(output_chunks),
            stats=GenerationStats(
                token_count=token_count,
                total_time_s=total_duration,
                ttft_s=(first_token_time - start_time) if first_token_time else None,
                tokens_per_second=tokens_per_second,
                token_events=token_events,
            ),
        )
