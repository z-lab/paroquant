import torch
from paroquant_kernels import gemv_forward_cuda_new

awq_lib = torch.library.Library("awq", "DEF")
awq_lib.define(
    "gemv(Tensor input, Tensor qweight, Tensor scales, Tensor scaled_zeros, "
    "int group_size, int interleave=4) -> Tensor"
)

@torch.library.impl("awq::gemv", "CUDA")
def _gemv_cuda(input: torch.Tensor,
               qweight: torch.Tensor,
               scales: torch.Tensor,
               scaled_zeros: torch.Tensor,
               group_size: int,
               interleave: int = 4):
    in_features = input.shape[-1]
    batch = input.numel() // in_features
    out_features = qweight.shape[0] * interleave

    assert scales.shape[-1] == out_features and scaled_zeros.shape[-1] == out_features, \
        "scales/scaled_zeros last dim must equal out_features"

    return gemv_forward_cuda_new(
        input,
        qweight,
        scales,
        scaled_zeros,
        batch,
        out_features,
        in_features,
        group_size,
    )

@torch.library.register_fake("awq::gemv")
def _gemv_fake(input: torch.Tensor,
               qweight: torch.Tensor,
               scales: torch.Tensor,
               scaled_zeros: torch.Tensor,
               group_size: int,
               interleave: int = 4) -> torch.Tensor:
    in_features = input.shape[-1]
    out_features = qweight.shape[0] * interleave
    assert in_features % group_size == 0, "in_features must be divisible by group_size"
    assert scales.shape[-1] == out_features and scaled_zeros.shape[-1] == out_features, \
        "scales/scaled_zeros last dim must equal out_features"

    out_shape = (*input.shape[:-1], out_features)
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)