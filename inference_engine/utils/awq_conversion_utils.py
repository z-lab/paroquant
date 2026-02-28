from __future__ import annotations

import numpy as np
import torch


def pack_cols(
    q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int
) -> torch.Tensor:
    assert q_w.shape == (size_k, size_n)
    pack_factor = 32 // num_bits
    assert size_n % pack_factor == 0

    orig_device = q_w.device
    q_w = q_w.cpu().numpy().astype(np.uint32)
    q_res = np.zeros((size_k, size_n // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << (num_bits * i)
    q_res = torch.from_numpy(q_res.astype(np.int32)).to(orig_device).contiguous()
    return q_res


def unpack_awq_llm_qweight(
    qweight: torch.Tensor, interleave: int, kstride: int
) -> torch.Tensor:
    """Inverse of qmodule.pack_intweight for AWQ-LLM int16 layout."""
    q = qweight.cpu().numpy().astype(np.uint16)
    n_div, k = q.shape

    v0 = q & 0xF
    v1 = (q >> 4) & 0xF
    v2 = (q >> 8) & 0xF
    v3 = (q >> 12) & 0xF
    packed = np.stack([v0, v1, v2, v3], axis=-1)

    packed = packed.reshape(n_div, k // kstride, kstride, interleave)
    packed = packed.reshape(n_div, k // kstride, interleave, kstride)
    packed = packed.transpose(0, 2, 1, 3)
    packed = packed.reshape(n_div * interleave, k)

    packed = packed.reshape(n_div * interleave, k // 32, 4, 2, 4)
    packed = packed.transpose(0, 1, 2, 4, 3)
    packed = packed.reshape(n_div * interleave, k // 32, 4, 8)
    packed = packed.reshape(n_div * interleave, k)

    packed = packed.reshape(n_div * interleave, k // 32, 4, 4, 2)
    packed = packed.transpose(0, 1, 3, 2, 4)
    packed = packed.reshape(n_div * interleave, k)

    return torch.from_numpy(packed.astype(np.int32)).to(qweight.device)


def convert_awq_llm_module_to_autoawq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    w_bit: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    interleave = 4
    int16_pack_num = 16 // w_bit
    kstride = 64

    out_features = qweight.shape[0] * interleave
    in_features = (qweight.shape[1] // interleave) * int16_pack_num
    num_groups = in_features // group_size

    unpacked = unpack_awq_llm_qweight(qweight, interleave=interleave, kstride=kstride)
    if w_bit == 4:
        reorder = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=unpacked.device)
    elif w_bit == 8:
        reorder = torch.tensor([0, 2, 1, 3], device=unpacked.device)
    else:
        raise ValueError(f"Unsupported w_bit: {w_bit}")
    unpacked = unpacked.view(out_features // reorder.numel(), reorder.numel(), -1)
    unpacked = unpacked[:, reorder, :].reshape(out_features, in_features)

    qweight_awq = pack_cols(
        unpacked.transpose(0, 1).contiguous(),
        num_bits=w_bit,
        size_k=in_features,
        size_n=out_features,
    )

    scales = scales[:num_groups, :]
    scaled_zeros = scaled_zeros[:num_groups, :]

    zeros = -scaled_zeros / scales
    zeros = torch.round(zeros).clamp_(0, 2**w_bit - 1).to(torch.int32)

    if w_bit == 4:
        reorder = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=zeros.device)
    elif w_bit == 8:
        reorder = torch.tensor([0, 2, 1, 3], device=zeros.device)
    else:
        raise ValueError(f"Unsupported w_bit: {w_bit}")
    zeros = zeros.view(num_groups, out_features // reorder.numel(), -1)
    zeros = zeros[:, :, reorder].reshape(num_groups, out_features)

    qzeros = pack_cols(
        zeros.contiguous(),
        num_bits=w_bit,
        size_k=num_groups,
        size_n=out_features,
    )

    return qweight_awq, qzeros, scales
