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


def _apply_rotation(
    x: mx.array,
    packed_pairs: mx.array,
    cos: mx.array,
    sin: mx.array,
    scales_flat: mx.array,
    dim: int,
    krot: int,
    group_size: int,
) -> mx.array:
    """Dispatch the Metal pairwise-rotation kernel on a 2-D (batch, dim) tensor."""
    batch = x.shape[0]
    if batch == 0:
        return x
    tile = 1 if batch <= 1 else 4
    half_group = group_size // 2
    num_groups = dim // group_size
    params = mx.array([batch, dim, krot, group_size], dtype=mx.int32)
    grid = (math.ceil(batch / tile) * half_group, num_groups, 1)
    return get_rotation_kernel(tile)(
        inputs=[x, packed_pairs, cos, sin, scales_flat, params],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=grid,
        threadgroup=(half_group, 1, 1),
    )[0]


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
        self._krot = int(self.theta.shape[0])
        self._cos = mx.cos(self.theta)
        self._sin = mx.sin(self.theta)
        self._packed_pairs = _pack_pairs(self.pairs, self.group_size)
        self._scales_flat = self.channel_scales.reshape(-1)
        self._cached = True

    def __call__(self, x: mx.array) -> mx.array:
        if not self._cached:
            self._cache_rotation()

        shape = x.shape
        rotated = _apply_rotation(
            x.reshape(-1, self._dim),
            self._packed_pairs,
            self._cos,
            self._sin,
            self._scales_flat,
            self._dim,
            self._krot,
            self.group_size,
        )

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


class RotateQuantizedSwitchLinear(nn.Module):
    """Per-expert pairwise Givens rotation + quantized gather matmul for MoE.

    Drop-in replacement for ``SwitchLinear`` / ``QuantizedSwitchLinear`` inside
    a ``SwitchGLU`` block.  Each expert carries its own rotation parameters
    (theta, pairs, channel_scales) and INT4 weights.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = False,
        group_size: int = 128,
        bits: int = 4,
        krot: int = 8,
    ):
        super().__init__()
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._num_experts = num_experts
        self.group_size = group_size
        self.bits = bits

        self.theta = mx.zeros((num_experts, krot, input_dims // 2))
        self.pairs = mx.zeros((num_experts, krot, input_dims), dtype=mx.int16)
        self.channel_scales = mx.ones((num_experts, 1, input_dims))

        self.weight = mx.zeros((num_experts, output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, input_dims // group_size))
        self.biases = mx.zeros((num_experts, output_dims, input_dims // group_size))

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self._cached = False

    @property
    def input_dims(self):
        return self._input_dims

    @property
    def output_dims(self):
        return self._output_dims

    @property
    def num_experts(self):
        return self._num_experts

    def _cache_rotation(self):
        """Pre-compute sin/cos and pack pairs for all experts."""
        self._krot = int(self.theta.shape[1])
        self._dim = int(self.theta.shape[2]) * 2
        self._cos = mx.cos(self.theta)
        self._sin = mx.sin(self.theta)
        self._packed_pairs = mx.stack([_pack_pairs(self.pairs[e], self.group_size) for e in range(self._num_experts)])
        self._scales_flat = self.channel_scales.reshape(self._num_experts, -1)
        self._cached = True

    def _rotate_by_expert(
        self,
        x: mx.array,
        indices: mx.array,
        *,
        already_sorted: bool = False,
    ) -> mx.array:
        """Apply per-expert rotation, grouping tokens by expert for efficiency."""
        if x.shape[0] == 0:
            return x

        if already_sorted:
            sorted_x, sorted_idx = x, indices
        else:
            order = mx.argsort(indices)
            sorted_idx = indices[order]
            sorted_x = x[order]

        mx.eval(sorted_idx)
        expert_ids, counts = mx.unique(sorted_idx, return_counts=True)
        mx.eval(expert_ids, counts)

        chunks, offset = [], 0
        for i in range(expert_ids.shape[0]):
            eid, cnt = int(expert_ids[i]), int(counts[i])
            chunks.append(
                _apply_rotation(
                    sorted_x[offset : offset + cnt],
                    self._packed_pairs[eid],
                    self._cos[eid],
                    self._sin[eid],
                    self._scales_flat[eid],
                    self._dim,
                    self._krot,
                    self.group_size,
                )
            )
            offset += cnt

        rotated = mx.concatenate(chunks, axis=0)
        if not already_sorted:
            rotated = rotated[mx.argsort(order)]
        return rotated

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        if not self._cached:
            self._cache_rotation()

        has_mid = x.ndim >= 3 and x.shape[-2] == 1
        x_2d = x.squeeze(-2) if has_mid else x
        flat = x_2d.reshape(-1, self._dim)

        rotated = self._rotate_by_expert(flat, indices.reshape(-1), already_sorted=sorted_indices)
        rotated = rotated.reshape(x_2d.shape)
        if has_mid:
            rotated = mx.expand_dims(rotated, -2)

        y = mx.gather_qmm(
            rotated,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            y = y + mx.expand_dims(self["bias"][indices], -2)
        return y
