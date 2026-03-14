import shutil
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_dir = Path(__file__).parent


def _rotation_build_directory() -> Path:
    cache_root = Path.home() / ".cache" / "paroquant" / "torch_extensions"
    abi_tag = (
        f"py{sys.version_info.major}{sys.version_info.minor}_"
        f"torch{torch.__version__}_"
        f"cu{torch.version.cuda or 'none'}"
    )
    abi_tag = "".join(c if c.isalnum() else "_" for c in abi_tag)
    build_dir = cache_root / "paroquant_rotation" / abi_tag
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def _load_rotation_extension():
    build_dir = _rotation_build_directory()
    load_kwargs = dict(
        name="paroquant_rotation",
        sources=[str(_dir / "pybind.cpp"), str(_dir / "rotation.cu")],
        build_directory=str(build_dir),
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
        return load(**load_kwargs)
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=True)
        return load(**load_kwargs)


_C = _load_rotation_extension()

try:

    @torch.library.register_fake("rotation::rotate")
    def _fake_kernel(x, idx_ij, theta, scales=None, group_size=128):
        return torch.empty_like(x)

except RuntimeError:
    pass

from .autograd import RotateTensorFunc, scaled_pairwise_rotation  # noqa: E402

__all__ = ["scaled_pairwise_rotation", "RotateTensorFunc"]
