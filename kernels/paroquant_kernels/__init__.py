# Copyright (c) 2025, Haisheng Chen.

import torch
import _C


@torch.library.register_fake("rotation::rotate")
def _fake_kernel(x, idx_ij, theta, scales):
    return torch.empty_like(x)


from .interface import RotateTensorFunc, fast_givens_transform

__all__ = ["fast_givens_transform", "RotateTensorFunc"]
