"""RotateQuantizedLinear — pairwise rotation + INT4 quantized matmul."""

import math

import mlx.core as mx
import mlx.nn as nn

from .rotation import _get_kernel, pack_pairs


class RotateQuantizedLinear(nn.Module):
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

        self._setup_done = False

    def _setup(self):
        hidden_dim = self.theta.shape[1] * 2
        half_group = self.group_size // 2
        num_groups = hidden_dim // self.group_size
        krot = int(self.theta.shape[0])

        self._hidden_dim = hidden_dim
        self._half_group = half_group
        self._num_groups = num_groups
        self._krot = krot

        self._cos_theta = mx.cos(self.theta)
        self._sin_theta = mx.sin(self.theta)
        self._packed_pairs = pack_pairs(self.pairs, self.group_size)
        self._channel_scales_flat = self.channel_scales.reshape(-1)

        self._setup_done = True

    def _rotate(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        rows_per_tile = 1 if batch_size <= 1 else 4
        params = mx.array(
            [batch_size, self._hidden_dim, self._krot, self.group_size],
            dtype=mx.int32,
        )
        grid = (
            math.ceil(batch_size / rows_per_tile) * self._half_group,
            self._num_groups,
            1,
        )
        return _get_kernel(rows_per_tile)(
            inputs=[x, self._packed_pairs, self._cos_theta, self._sin_theta, self._channel_scales_flat, params],
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
            grid=grid,
            threadgroup=(self._half_group, 1, 1),
        )[0]

    def __call__(self, x: mx.array) -> mx.array:
        if not self._setup_done:
            self._setup()

        shape = x.shape
        flat = x.reshape(-1, self._hidden_dim)
        rotated = self._rotate(flat)

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
