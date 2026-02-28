from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class GenerationParams:
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: Optional[int] = 32


@dataclass
class GenerationStats:
    token_count: int
    total_time_s: float
    ttft_s: Optional[float]
    tokens_per_second: float
    token_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GenerationResult:
    backend: str
    prompt: str
    output_text: str
    stats: GenerationStats


class UnifiedGenerator(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        params: GenerationParams,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def create_generator(backend: str, model: str, **kwargs) -> UnifiedGenerator:
    normalized = backend.lower()

    if normalized == "transformers":
        from .transformers_backend import TransformersGenerator

        return TransformersGenerator(model=model, **kwargs)

    if normalized == "vllm":
        from .vllm_backend import VllmGenerator

        return VllmGenerator(model=model, **kwargs)

    raise ValueError(f"Unsupported backend: {backend}")
