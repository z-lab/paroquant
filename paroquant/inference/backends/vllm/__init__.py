from .generator import VllmGenerator

__all__ = ["VllmGenerator"]


def register() -> None:
    """vLLM general-plugin entry point. Idempotent."""
    import paroquant.kernels.cuda  # noqa: F401 — registers torch.ops.rotation.rotate
    import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers ParoQuantConfig
