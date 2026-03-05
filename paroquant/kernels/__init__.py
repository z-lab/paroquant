# Copyright (c) 2025, Haisheng Chen.

import torch

from . import _C  # loads the compiled .so, which registers torch.ops.rotation.rotate


@torch.library.register_fake("rotation::rotate")
def _fake_kernel(x, idx_ij, theta, scales, group_size=128):
    return torch.empty_like(x)


from .interface import RotateTensorFunc, scaled_pairwise_rotation

__all__ = ["scaled_pairwise_rotation", "RotateTensorFunc"]
