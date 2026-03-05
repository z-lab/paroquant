"""CUDA rotation kernel for ParoQuant."""

import torch

from paroquant_kernels import _C  # noqa: F401 — loads the compiled .so, registers torch.ops.rotation.rotate

try:

    @torch.library.register_fake("rotation::rotate")
    def _fake_kernel(x, idx_ij, theta, scales, group_size=128):
        return torch.empty_like(x)

except RuntimeError:
    pass  # already registered (e.g. vLLM spawned subprocess re-imports)


from .autograd import RotateTensorFunc, scaled_pairwise_rotation

__all__ = ["scaled_pairwise_rotation", "RotateTensorFunc"]
