from setuptools import setup
from torch.utils import cpp_extension
from typing import Set
import os
import warnings
import torch

# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0", "10.0", "12.0"}

def get_torch_arch_list() -> Set[str]:
    # TORCH_CUDA_ARCH_LIST can have one or more architectures,
    # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
    # compiler to additionally include PTX code that can be runtime-compiled
    # and executed on the 8.6 or newer architectures. While the PTX code will
    # not give the best performance on the newer architectures, it provides
    # forward compatibility.
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # List are separated by ; or space.
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}."
        )
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}."
        )
    return arch_list


compute_capabilities = get_torch_arch_list()
if not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability below 7.0 are not supported."
            )
        compute_capabilities.add(f"{major}.{minor}")

setup(
    name="paroquant_kernels",
    version="0.1.0",
    packages=["paroquant_kernels"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "paroquant_kernels._C",
            [
                "pybind.cpp",
                "paroquant_kernels/src/rotation.cu",
                "paroquant_kernels/awq_inference_kernels/gemv/gemv_cuda.cu",
                "paroquant_kernels/awq_inference_kernels/gemm/gemm_cuda.cu"
            ],
            extra_compile_args={
                "cxx": ["-O2", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
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
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    options={},
)
