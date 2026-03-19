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

        self._force_eval = False
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
        if self._force_eval:
            mx.eval(y)
        return y


class _CachedRotation:
    """Mixin-style helper that pre-computes sin/cos and packs pairs for a single rotation."""

    def _init_rotation(self, krot: int, dim: int, group_size: int, prefix: str = ""):
        pfx = f"{prefix}_" if prefix else ""
        setattr(self, f"{pfx}theta", mx.zeros((krot, dim // 2)))
        setattr(self, f"{pfx}pairs", mx.zeros((krot, dim), dtype=mx.int16))
        setattr(self, f"{pfx}channel_scales", mx.ones((1, dim)))
        self._rot_group_size = group_size

    def _cache_single_rotation(self, prefix: str = ""):
        pfx = f"{prefix}_" if prefix else ""
        theta = getattr(self, f"{pfx}theta")
        dim = int(theta.shape[1]) * 2
        krot = int(theta.shape[0])
        cos = mx.cos(theta)
        sin = mx.sin(theta)
        packed_pairs = _pack_pairs(getattr(self, f"{pfx}pairs"), self._rot_group_size)
        scales_flat = getattr(self, f"{pfx}channel_scales").reshape(-1)
        tag = f"_{prefix}" if prefix else ""
        setattr(self, f"_rot{tag}_dim", dim)
        setattr(self, f"_rot{tag}_krot", krot)
        setattr(self, f"_rot{tag}_cos", cos)
        setattr(self, f"_rot{tag}_sin", sin)
        setattr(self, f"_rot{tag}_packed_pairs", packed_pairs)
        setattr(self, f"_rot{tag}_scales_flat", scales_flat)

    def _rotate(self, x: mx.array, prefix: str = "") -> mx.array:
        tag = f"_{prefix}" if prefix else ""
        dim = getattr(self, f"_rot{tag}_dim")
        shape = x.shape
        rotated = _apply_rotation(
            x.reshape(-1, dim),
            getattr(self, f"_rot{tag}_packed_pairs"),
            getattr(self, f"_rot{tag}_cos"),
            getattr(self, f"_rot{tag}_sin"),
            getattr(self, f"_rot{tag}_scales_flat"),
            dim,
            getattr(self, f"_rot{tag}_krot"),
            self._rot_group_size,
        )
        return rotated.reshape(shape)


class RotateSwitchGLU(nn.Module, _CachedRotation):
    """SwitchGLU with shared pairwise rotation injected before each sub-layer.

    All experts share a single set of rotation parameters per projection:
    ``gate_up_rot`` is applied to x before gate_proj/up_proj, and
    ``down_rot`` is applied to the activation output before down_proj.
    """

    def __init__(self, glu: nn.Module, group_size: int, krot: int):
        super().__init__()
        self.gate_proj = glu.gate_proj
        self.up_proj = glu.up_proj
        self.down_proj = glu.down_proj
        self.activation = glu.activation

        gate_up_dim = glu.gate_proj.input_dims
        down_dim = glu.down_proj.input_dims
        self._init_rotation(krot, gate_up_dim, group_size, prefix="gate_up_rot")
        self._init_rotation(krot, down_dim, group_size, prefix="down_rot")
        self._cached = False

    def _cache_rotation(self):
        self._cache_single_rotation("gate_up_rot")
        self._cache_single_rotation("down_rot")
        self._cached = True

    def __call__(self, x, indices) -> mx.array:
        if not self._cached:
            self._cache_rotation()

        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        x = self._rotate(x, "gate_up_rot")

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)

        act = self.activation(x_up, x_gate)
        act = self._rotate(act, "down_rot")

        x = self.down_proj(act, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
