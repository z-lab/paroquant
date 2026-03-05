from functools import lru_cache
from pathlib import Path

import mlx.core as mx

_SOURCE = (Path(__file__).parent / "rotation.metal").read_text()
_MAX_GROUP_SIZE = 128
_MAX_KROT = 16


@lru_cache(maxsize=None)
def get_rotation_kernel(rows_per_tile: int):
    """Compile and cache the Metal rotation kernel for a given tile size."""
    return mx.fast.metal_kernel(
        name=f"paro_rotate_r{rows_per_tile}",
        input_names=["x", "packed_pairs", "cos_theta", "sin_theta", "channel_scales", "params"],
        output_names=["out"],
        source=_SOURCE.format(
            ROWS_PER_TILE=rows_per_tile,
            MAX_GROUP_SIZE=_MAX_GROUP_SIZE,
            MAX_KROT=_MAX_KROT,
        ),
    )
