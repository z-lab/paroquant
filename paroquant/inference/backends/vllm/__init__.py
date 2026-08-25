from __future__ import annotations

from typing import Any

__all__ = ["VllmGenerator", "register"]


def __getattr__(name: str) -> Any:
    if name == "VllmGenerator":
        from .generator import VllmGenerator

        return VllmGenerator
    raise AttributeError(name)


def register() -> None:
    """vLLM general-plugin entry point. Idempotent."""
    import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers ParoQuantConfig
