"""RotateQuantizedLinear — pairwise rotation + INT4 quantized matmul for MLX."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from paroquant.kernels.metal import get_rotation_kernel


def _pack_pairs(pairs: mx.array, group_size: int) -> mx.array:
    """Pack int16 pair indices into int32 for the Metal kernel."""
    krot, hidden = int(pairs.shape[0]), int(pairs.shape[1])
    p = np.array(pairs, copy=False).reshape(krot, hidden // group_size, group_size).astype(np.int32, copy=False)
    return mx.array((p[:, :, 0::2] | (p[:, :, 1::2] << 16)).reshape(krot, -1))


class RotateQuantizedLinear(nn.Module):
    """Pairwise Givens rotation + quantized matmul (Metal kernel)."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 128,
        bits: int = 4,
        krot: int = 8,
    ):
        super().__init__()
        self.group_size = group_size
        self.bits = bits

        self.theta = mx.zeros((krot, input_dims // 2))
        self.pairs = mx.zeros((krot, input_dims), dtype=mx.int16)
        self.channel_scales = mx.ones((1, input_dims))

        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // group_size))
        self.biases = mx.zeros((output_dims, input_dims // group_size))

        if bias:
            self.bias = mx.zeros((output_dims,))

        self._cached = False

    def _cache_rotation(self):
        """Pre-compute sin/cos and pack pairs (called once on first forward)."""
        dim = self.theta.shape[1] * 2
        self._dim = dim
        self._half_group = self.group_size // 2
        self._num_groups = dim // self.group_size
        self._krot = int(self.theta.shape[0])
        self._cos = mx.cos(self.theta)
        self._sin = mx.sin(self.theta)
        self._packed_pairs = _pack_pairs(self.pairs, self.group_size)
        self._scales_flat = self.channel_scales.reshape(-1)
        self._cached = True

    def _rotate(self, x: mx.array) -> mx.array:
        batch = x.shape[0]
        tile = 1 if batch <= 1 else 4
        params = mx.array([batch, self._dim, self._krot, self.group_size], dtype=mx.int32)
        grid = (math.ceil(batch / tile) * self._half_group, self._num_groups, 1)
        return get_rotation_kernel(tile)(
            inputs=[x, self._packed_pairs, self._cos, self._sin, self._scales_flat, params],
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
            grid=grid,
            threadgroup=(self._half_group, 1, 1),
        )[0]

    def __call__(self, x: mx.array) -> mx.array:
        if not self._cached:
            self._cache_rotation()

        shape = x.shape
        rotated = self._rotate(x.reshape(-1, self._dim))

        y = mx.quantized_matmul(
            rotated.reshape(shape),
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if "bias" in self:
            y = y + self.bias
        return y
