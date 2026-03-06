from __future__ import annotations

import importlib
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass


@dataclass
class GenerationParams:
    """Sampling parameters (follows vLLM SamplingParams convention)."""

    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 32
    repetition_penalty: float = 1.0


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
    backend: str = "unknown"

    @abstractmethod
    async def _stream(
        self,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[str]:
        """Yield text chunks one at a time."""
        ...

    async def generate(
        self,
        prompt: str,
        params: GenerationParams,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        start = time.perf_counter()
        first_token_time = None
        chunks: list[str] = []

        async for text in self._stream(prompt, params):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            chunks.append(text)
            if on_text:
                on_text(text)

        end = time.perf_counter()
        output_text = "".join(chunks)
        gen_time = end - (first_token_time or start)
        num_tokens = self._count_tokens(output_text)

        return GenerationResult(
            backend=self.backend,
            prompt=prompt,
            output_text=output_text,
            stats=GenerationStats(
                num_tokens=num_tokens,
                latency=end - start,
                ttft=(first_token_time - start) if first_token_time else None,
                tps=num_tokens / gen_time if gen_time > 0 else 0.0,
            ),
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    async def close(self) -> None:
        pass


def build_prompt(tokenizer, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    """Apply chat template to messages, with fallback for tokenizers without templates."""
    if tokenizer.chat_template is None:
        return messages[-1]["content"]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


_BACKENDS = {
    "transformers": ("paroquant.inference.backends.transformers", "TransformersGenerator", "paroquant[transformers]"),
    "vllm": ("paroquant.inference.backends.vllm", "VllmGenerator", "paroquant[vllm]"),
    "mlx": ("paroquant.inference.backends.mlx", "MlxGenerator", "paroquant[mlx]"),
}


def _detect_backend() -> str:
    import platform

    if platform.processor() == "arm" or platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401

            return "mlx"
        except ImportError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            try:
                import vllm  # noqa: F401

                return "vllm"
            except ImportError:
                return "transformers"
    except ImportError:
        pass
    raise RuntimeError("No backend available. Install paroquant[mlx], paroquant[vllm], or paroquant[transformers].")


def create_generator(backend: str, model: str, **kwargs) -> BaseGenerator:
    """Factory that instantiates the right backend by name."""
    key = backend.lower()
    if key == "auto":
        key = _detect_backend()

    if key not in _BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: {', '.join(_BACKENDS)}")

    module_path, class_name, install_hint = _BACKENDS[key]
    try:
        cls = getattr(importlib.import_module(module_path), class_name)
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(f'Backend {backend!r} requires: pip install "{install_hint}"') from e

    return cls(model=model, **kwargs)
