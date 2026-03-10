from __future__ import annotations

import torch

_AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def unpack_awq_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack AutoAWQ int32-packed weights into uint4 values."""
    if packed.dtype not in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
        packed = packed.to(torch.int32)
    pack_factor = 8
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    raw = ((packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF).to(torch.int16)
    raw = raw.view(raw.shape[0], -1, pack_factor)
    return raw[:, :, _AWQ_REVERSE_ORDER].reshape(raw.shape[0], -1)


def dequantize_awq_4bit(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize AWQ 4-bit tensors into dense weight matrix [out_features, in_features]."""
    iweight = unpack_awq_4bit(qweight).to(torch.float32)
    izeros = unpack_awq_4bit(qzeros).to(torch.float32)

    scales_f = scales.to(torch.float32)

    if iweight.shape[0] % group_size != 0:
        raise ValueError(f"Invalid qweight rows={iweight.shape[0]} for group_size={group_size}")
    if scales_f.shape[0] * group_size != iweight.shape[0]:
        raise ValueError(
            f"Scale rows ({scales_f.shape[0]}) * group_size ({group_size}) != in_features ({iweight.shape[0]})"
        )

    scales_rep = scales_f.repeat_interleave(group_size, dim=0)
    zeros_rep = izeros.repeat_interleave(group_size, dim=0)

    dense_in_out = (iweight - zeros_rep) * scales_rep
    return dense_in_out.T.contiguous()


def fuse_inverse_rotations(
    weight: torch.Tensor,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    channel_scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Fuse ParoQuant rotation/scaling into dense weights.

    Given y = rotate(x, pairs, theta, channel_scales) @ W^T, produce W_eff such that
    y = x @ W_eff^T by applying inverse rotations in reverse order and channel scales.
    """
    w = weight.to(torch.float32)
    out_features, in_features = w.shape

    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} must be divisible by group_size={group_size}")

    krot = int(theta.shape[0])
    num_groups = in_features // group_size

    if pairs.shape != (krot, in_features):
        raise ValueError(f"Unexpected pairs shape: {pairs.shape}, expected ({krot}, {in_features})")
    if theta.shape != (krot, in_features // 2):
        raise ValueError(f"Unexpected theta shape: {theta.shape}, expected ({krot}, {in_features // 2})")

    pairs_view = pairs.to(torch.int64).reshape(krot, num_groups, group_size)
    theta_view = theta.to(torch.float32).reshape(krot, num_groups, group_size // 2)

    for ridx in range(krot - 1, -1, -1):
        for gidx in range(num_groups):
            base = gidx * group_size
            pair_row = pairs_view[ridx, gidx]
            angles = theta_view[ridx, gidx]

            idx_i = base + pair_row[0::2]
            idx_j = base + pair_row[1::2]

            col_i = w[:, idx_i].clone()
            col_j = w[:, idx_j].clone()
            cos = torch.cos(angles).to(w.dtype).unsqueeze(0)
            sin = torch.sin(angles).to(w.dtype).unsqueeze(0)

            w[:, idx_i] = cos * col_i - sin * col_j
            w[:, idx_j] = sin * col_i + cos * col_j

    scale = channel_scales.reshape(-1).to(w.dtype)
    if scale.numel() != in_features:
        raise ValueError(f"Unexpected channel_scales shape: {channel_scales.shape}")

    w = w * scale.unsqueeze(0)
    return w
