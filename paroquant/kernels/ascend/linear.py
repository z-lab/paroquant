from __future__ import annotations

import torch
import triton
import triton.language as tl


def prepare_rotation(pairs, theta, group_size=128):
    """Expand pair coefficients once while retaining every rotation stage."""
    rounds, features = pairs.shape
    grouped = pairs.long().view(rounds, features // group_size, group_size)
    first, second = grouped[..., 0::2], grouped[..., 1::2]
    angles = theta.float().view(rounds, features // group_size, group_size // 2)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    partner = torch.empty_like(grouped, dtype=torch.int32)
    cos = torch.empty_like(grouped, dtype=torch.float32)
    sin = torch.empty_like(cos)
    partner.scatter_(2, first, second.int())
    partner.scatter_(2, second, first.int())
    cos.scatter_(2, first, cosine)
    cos.scatter_(2, second, cosine)
    sin.scatter_(2, first, sine)
    sin.scatter_(2, second, -sine)
    return partner.contiguous(), cos.contiguous(), sin.contiguous()


@triton.jit
def _rotate(
    X, Scale, Partner, Cos, Sin, Y, M: tl.constexpr, H: tl.constexpr, R: tl.constexpr, G: tl.constexpr, BM: tl.constexpr
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    group = tl.program_id(1)
    d = tl.arange(0, G)
    offset = group * G + d
    values = tl.load(X + row[:, None] * H + offset[None, :], row[:, None] < M, 0).to(tl.float32)
    scale = tl.load(Scale + offset).to(tl.float32)
    values = (values * scale[None, :]).to(X.dtype.element_ty).to(tl.float32)
    for r in range(R):
        partner = tl.load(Partner + r * H + offset)
        cosine = tl.load(Cos + r * H + offset)
        sine = tl.load(Sin + r * H + offset)
        index = tl.broadcast_to(partner[None, :], (BM, G))
        other = tl.gather(values, index, axis=1)
        values = tl.fma(values, cosine[None, :], other * sine[None, :])
        values = values.to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row[:, None] * H + offset[None, :], values, row[:, None] < M)


def rotate(x, channel_scales, prepared, group_size=128):
    """Fuse channel scaling and all pairwise stages in a group-local tile."""
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    m, h = x.shape
    partner, cos, sin = prepared
    out = torch.empty_like(x)
    bm = 1 if m < 8 else 8
    _rotate[(triton.cdiv(m, bm), h // group_size)](
        x, channel_scales, partner, cos, sin, out, m, h, partner.shape[0], group_size, bm
    )
    return out.view(shape)


@triton.jit
def _unpack(Q, Z, S, kb, nb, K: tl.constexpr, N: tl.constexpr, G: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    k = kb + tl.arange(0, BK)
    packed_n = nb // 8 + tl.arange(0, BN // 8)
    lane = tl.arange(0, 8)
    shift = (lane // 2 + (lane % 2) * 4) * 4
    packed = tl.load(Q + k[:, None] * (N // 8) + packed_n[None, :], k[:, None] < K, 0)
    integers = (packed[:, :, None] >> shift[None, None, :]) & 15
    integers = tl.reshape(integers, (BK, BN)).to(tl.float32)
    zeros = tl.load(Z + (kb // G) * (N // 8) + packed_n)
    zeros = tl.reshape((zeros[:, None] >> shift[None, :]) & 15, (BN,)).to(tl.float32)
    n = nb + tl.arange(0, BN)
    scales = tl.load(S + (kb // G) * N + n).to(tl.float32)
    return ((integers - zeros[None, :]) * scales[None, :]).to(tl.float16)


@triton.jit
def _matmul(
    X,
    Q,
    Z,
    S,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    nb = tl.program_id(1) * BN
    n = nb + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(0, K, BK):
        k = kb + tl.arange(0, BK)
        x = tl.load(X + row[:, None] * K + k[None, :], (row[:, None] < M) & (k[None, :] < K), 0)
        weight = _unpack(Q, Z, S, kb, nb, K, N, G, BK, BN)
        acc = tl.dot(x, weight, acc)
    tl.store(Y + row[:, None] * N + n[None, :], acc, row[:, None] < M)


@triton.jit
def _gemv(
    X,
    Q,
    Z,
    S,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    nb = tl.program_id(1) * BN
    n = nb + tl.arange(0, BN)
    acc = tl.zeros((BN,), tl.float32)
    for kb in range(0, K, BK):
        k = kb + tl.arange(0, BK)
        x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
        weight = _unpack(Q, Z, S, kb, nb, K, N, G, BK, BN).to(tl.float32)
        acc += tl.sum(x[:, None] * weight, axis=0)
    tl.store(Y + row * N + n, acc)


def linear(x, qweight, qzeros, scales, group_size=128, bias=None):
    """Multiply AWQ INT4 weights with bounded on-chip dequantization storage."""
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    m, k = x.shape
    n = scales.shape[1]
    if x.dtype != torch.float16 or group_size != 128 or n % 64:
        raise ValueError("Ascend ParoQuant requires FP16, group_size=128 and output width divisible by64")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m <= 4:
        _gemv[(m, n // 16)](x, qweight, qzeros, scales, out, m, k, n, group_size, 16, 128)
    else:
        _matmul[(triton.cdiv(m, 16), n // 64)](x, qweight, qzeros, scales, out, m, k, n, group_size, 16, 64, 32)
    if bias is not None:
        out = out + bias
    return out.view(*shape[:-1], n)
