from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_dir = Path(__file__).parent
_C = load(
    name="paroquant_rotation",
    sources=[str(_dir / "pybind.cpp"), str(_dir / "rotation.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-O2", "-std=c++17"],
    verbose=False,
)

try:

    @torch.library.register_fake("rotation::rotate")
    def _fake_kernel(x, idx_ij, theta, scales, group_size=128):
        return torch.empty_like(x)

except RuntimeError:
    pass

from .autograd import RotateTensorFunc, scaled_pairwise_rotation  # noqa: E402

__all__ = ["scaled_pairwise_rotation", "RotateTensorFunc"]
