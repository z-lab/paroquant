#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "paroquant_kernels/src/rotation.h"
#include "paroquant_kernels/awq_inference_kernels/gemv/gemv_cuda.h"
#include "paroquant_kernels/awq_inference_kernels/gemm/gemm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_forward_cuda_new", &gemm_forward_cuda_new, "New quantized GEMM kernel.");
    m.def("gemv_forward_cuda_new", &gemv_forward_cuda_new, "New quantized GEMV kernel.");
}

