# Copyright (c) 2025, Haisheng Chen.

import torch
from . import _C


@torch.library.register_fake("rotation::rotate")
def _fake_kernel(x, idx_ij, theta, scales):
    return torch.empty_like(x)


from .interface import RotateTensorFunc, scaled_pairwise_rotation
from ._C import gemm_forward_cuda_new, gemv_forward_cuda_new

__all__ = [
    "scaled_pairwise_rotation",
    "RotateTensorFunc",
    "gemm_forward_cuda_new",
    "gemv_forward_cuda_new",
]
