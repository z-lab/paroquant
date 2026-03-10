from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from paroquant.export.math import dequantize_awq_4bit, fuse_inverse_rotations

AWQ_REORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_awq(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int32).view(values.shape[0], -1, 8)[:, :, AWQ_REORDER]
    out = torch.zeros(values.shape[0], values.shape[1], dtype=torch.int32)
    for i in range(8):
        out |= (values[:, :, i] & 0xF) << (4 * i)
    return out


def test_dequantize_awq_4bit_roundtrip_identity_scales() -> None:
    in_features = 16
    out_features = 32
    group_size = 4

    iweight = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    izeros = torch.zeros((in_features // group_size, out_features), dtype=torch.int32)
    scales = torch.ones((in_features // group_size, out_features), dtype=torch.float16)

    qweight = _pack_awq(iweight)
    qzeros = _pack_awq(izeros)

    dense = dequantize_awq_4bit(qweight, qzeros, scales, group_size=group_size)
    assert dense.shape == (out_features, in_features)
    assert torch.equal(dense.to(torch.int32), iweight.T)


def test_fuse_inverse_rotations_matches_manual_linearization() -> None:
    in_features = 8
    out_features = 4
    group_size = 4

    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    channel_scales = torch.tensor([[1.5, 0.5, 1.0, 0.75, 1.2, 0.8, 0.9, 1.1]], dtype=torch.float32)

    pairs = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.int16)
    theta = torch.tensor([[0.2, -0.4, 0.1, -0.3]], dtype=torch.float32)

    fused = fuse_inverse_rotations(weight, pairs, theta, channel_scales, group_size=group_size)

    # Verify by random inputs against explicit rotated forward path.
    x = torch.randn(7, in_features, dtype=torch.float32)
    x_scaled = x * channel_scales

    x_rot = x_scaled.clone()
    for g in range(in_features // group_size):
        base = g * group_size
        for p in range(group_size // 2):
            i = base + int(pairs[0, g * group_size + 2 * p])
            j = base + int(pairs[0, g * group_size + 2 * p + 1])
            c = torch.cos(theta[0, g * (group_size // 2) + p])
            s = torch.sin(theta[0, g * (group_size // 2) + p])
            xi = x_rot[:, i].clone()
            xj = x_rot[:, j].clone()
            x_rot[:, i] = c * xi + s * xj
            x_rot[:, j] = -s * xi + c * xj

    y_ref = x_rot @ weight.T
    y_fused = x @ fused.T

    assert torch.allclose(y_ref, y_fused, atol=1e-4, rtol=1e-4)
