"""Shared inference interface: dataclasses, ABC, factory, and utilities."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class GenerationParams:
    """Sampling parameters (follows vLLM SamplingParams convention)."""
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 32


@dataclass
class GenerationStats:
    num_tokens: int
    latency: float
    ttft: float | None
    tps: float


@dataclass
class GenerationResult:
    backend: str
    prompt: str
    output_text: str
    stats: GenerationStats


class BaseGenerator(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult: ...

    async def close(self) -> None:
        pass


def build_prompt(tokenizer, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    """Apply chat template to messages, with fallback for tokenizers without templates."""
    if tokenizer.chat_template is None:
        return messages[-1]["content"]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


_BACKENDS = {
    "transformers": ("paroquant.inference.backends.transformers", "TransformersGenerator", "paroquant[transformers]"),
    "vllm": ("paroquant.inference.backends.vllm", "VllmGenerator", "paroquant[vllm]"),
    "mlx": ("paroquant.inference.backends.mlx", "MlxGenerator", "paroquant[mlx]"),
}


def create_generator(backend: str, model: str, **kwargs) -> BaseGenerator:
    """Factory that instantiates the right backend by name."""
    key = backend.lower()
    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: {', '.join(_BACKENDS)}")

    module_path, class_name, install_hint = _BACKENDS[key]
    try:
        cls = getattr(importlib.import_module(module_path), class_name)
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(f"Backend {backend!r} requires: pip install \"{install_hint}\"") from e

    return cls(model=model, **kwargs)
