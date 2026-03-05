"""Build script that optionally compiles the CUDA rotation kernel.

If torch + CUDA are available, builds the paroquant._C extension.
Otherwise, installs the pure-Python package without the kernel
(MLX backend doesn't need it).
"""

import os
from setuptools import setup

ext_modules = []

if os.environ.get("PAROQUANT_SKIP_CUDA", "0") != "1":
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        kernel_dir = "paroquant/kernels"
        ext_modules.append(
            CUDAExtension(
                "paroquant_kernels._C",
                [
                    f"{kernel_dir}/pybind.cpp",
                    f"{kernel_dir}/rotation.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O2", "-std=c++17", "-DENABLE_BF16"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-DENABLE_BF16",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "-w",
                    ],
                },
            )
        )
        cmdclass = {"build_ext": BuildExtension}
    except (ImportError, ModuleNotFoundError):
        cmdclass = {}
else:
    cmdclass = {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=[
        "paroquant_kernels",
    ] if ext_modules else [],
    package_dir={
        "paroquant_kernels": "paroquant/kernels",
    } if ext_modules else {},
)
