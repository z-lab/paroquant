import torch
from paroquant_kernels import gemm_forward_cuda_new

awq_lib = torch.library.Library("awq", "FRAGMENT")
awq_lib.define(
    "gemm(Tensor input, Tensor qweight, Tensor scales, Tensor scaled_zeros, int interleave=4) -> Tensor"
)


@torch.library.impl("awq::gemm", "CUDA")
def _gemm_cuda(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    interleave: int = 4,
) -> torch.Tensor:
    out_features = qweight.shape[0] * interleave
    assert (
        scales.shape[-1] == out_features and scaled_zeros.shape[-1] == out_features
    ), "scales/scaled_zeros last dim must equal out_features"

    return gemm_forward_cuda_new(input, qweight, scales, scaled_zeros)


@torch.library.register_fake("awq::gemm")
def _gemm_fake(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    interleave: int = 4,
) -> torch.Tensor:
    out_features = qweight.shape[0] * interleave
    assert (
        scales.shape[-1] == out_features and scaled_zeros.shape[-1] == out_features
    ), "scales/scaled_zeros last dim must equal out_features"

    out_shape = (*input.shape[:-1], out_features)
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)
