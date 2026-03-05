"""Pairwise Givens rotation Metal kernel for ParoQuant.

Requires Apple Silicon with Metal. The kernel is JIT-compiled on first use
via ``mx.fast.metal_kernel`` and cached for the session lifetime.
"""

import numpy as np
import mlx.core as mx

_MAX_GROUP_SIZE = 128
_MAX_KROT = 16

_KERNEL_SOURCE = """
    constexpr int ROWS_PER_TILE = {ROWS_PER_TILE};
    constexpr int MAX_KROT      = {MAX_KROT};

    const int batch_size  = params[0];
    const int hidden_size = params[1];
    const int krot        = params[2];
    const int group_size  = params[3];

    const int half_gs     = group_size / 2;
    const int half_hidden = hidden_size / 2;

    const int tile_idx  = threadgroup_position_in_grid.x;
    const int group_idx = threadgroup_position_in_grid.y;
    const int tid       = thread_index_in_threadgroup;

    if (tid >= half_gs) return;

    // Preload rotation coefficients and packed pair indices into registers.
    float cos_vals[MAX_KROT], sin_vals[MAX_KROT];
    int pair_vals[MAX_KROT];
    for (int k = 0; k < krot; k++) {{
        int idx = k * half_hidden + group_idx * half_gs + tid;
        cos_vals[k] = float(cos_theta[idx]);
        sin_vals[k] = float(sin_theta[idx]);
        pair_vals[k] = int(packed_pairs[idx]);
    }}

    // Load activation tile into threadgroup memory, fusing channel scales.
    threadgroup float tile[{MAX_GROUP_SIZE} * ROWS_PER_TILE];

    const int ch_lo = group_idx * group_size + tid;
    const int ch_hi = ch_lo + half_gs;
    float scale_lo = float(channel_scales[ch_lo]);
    float scale_hi = float(channel_scales[ch_hi]);

    for (int r = 0; r < ROWS_PER_TILE; r++) {{
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {{
            tile[tid * ROWS_PER_TILE + r]            = float(x[row * hidden_size + ch_lo]) * scale_lo;
            tile[(tid + half_gs) * ROWS_PER_TILE + r] = float(x[row * hidden_size + ch_hi]) * scale_hi;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply rotation rounds in-place on the tile.
    for (int k = 0; k < krot; k++) {{
        int i_local = pair_vals[k] & 0xFFFF;
        int j_local = pair_vals[k] >> 16;
        float c = cos_vals[k], s = sin_vals[k];
        for (int m = 0; m < ROWS_PER_TILE; m++) {{
            float a = tile[i_local * ROWS_PER_TILE + m];
            float b = tile[j_local * ROWS_PER_TILE + m];
            tile[i_local * ROWS_PER_TILE + m] = a * c + b * s;
            tile[j_local * ROWS_PER_TILE + m] = b * c - a * s;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write results back.
    for (int r = 0; r < ROWS_PER_TILE; r++) {{
        int row = tile_idx * ROWS_PER_TILE + r;
        if (row < batch_size) {{
            out[row * hidden_size + ch_lo] = tile[tid * ROWS_PER_TILE + r];
            out[row * hidden_size + ch_hi] = tile[(tid + half_gs) * ROWS_PER_TILE + r];
        }}
    }}
"""

_kernel_cache: dict[int, object] = {}


def _get_kernel(rows_per_tile: int):
    """Return the Metal rotation kernel (lazily compiled and cached)."""
    if rows_per_tile not in _kernel_cache:
        _kernel_cache[rows_per_tile] = mx.fast.metal_kernel(
            name=f"paro_rotate_r{rows_per_tile}",
            input_names=["x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params"],
            output_names=["out"],
            source=_KERNEL_SOURCE.format(
                ROWS_PER_TILE=rows_per_tile,
                MAX_GROUP_SIZE=_MAX_GROUP_SIZE,
                MAX_KROT=_MAX_KROT,
            ),
        )
    return _kernel_cache[rows_per_tile]


def pack_pairs(pairs: mx.array, group_size: int) -> mx.array:
    """Pack raw ``int16`` pair indices into ``int32`` for the Metal kernel.

    Args:
        pairs: ``(krot, hidden_size)`` int16 local pair indices.
        group_size: Rotation group size.

    Returns:
        ``(krot, hidden_size // 2)`` int32 packed pairs.
    """
    krot, hidden_size = int(pairs.shape[0]), int(pairs.shape[1])
    num_groups = hidden_size // group_size
    p = np.array(pairs, copy=False).reshape(krot, num_groups, group_size).astype(np.int32, copy=False)
    packed = p[:, :, 0::2] | (p[:, :, 1::2] << 16)
    return mx.array(packed.reshape(krot, -1))
