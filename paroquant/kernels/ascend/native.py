"""Convert the published AWQ packing for Ascend's native W4A16 operator."""

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _repack(Q, Out, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    word = tl.load(Q + index, index < SIZE, 0).to(tl.uint32)
    packed = tl.full((BLOCK,), 0, tl.uint32)
    for lane in tl.static_range(8):
        source = (lane // 2 + (lane % 2) * 4) * 4
        nibble = ((word >> source) & 15) ^ 8
        packed |= nibble << (lane * 4)
    tl.store(Out + index, packed.to(tl.int32), index < SIZE)


@triton.jit
def _offset(Z, Out, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    word_index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    word = tl.load(Z + word_index, word_index < SIZE, 0).to(tl.uint32)
    lane = tl.arange(0, 8)
    source = (lane // 2 + (lane % 2) * 4) * 4
    zero = (word[:, None] >> source[None, :]) & 15
    offset = 8.0 - zero.to(tl.int32).to(tl.float32)
    tl.store(Out + word_index[:, None] * 8 + lane[None, :], offset, word_index[:, None] < SIZE)


def prepare_native(qweight, qzeros):
    weight = torch.empty_like(qweight)
    offset = torch.empty((qzeros.shape[0], qzeros.shape[1] * 8), device=qzeros.device, dtype=torch.float16)
    _repack[(triton.cdiv(qweight.numel(), 256),)](qweight, weight, qweight.numel(), 256)
    _offset[(triton.cdiv(qzeros.numel(), 128),)](qzeros, offset, qzeros.numel(), 128)
    return weight, offset


def native_linear(x, prepared, scales, group_size=128, bias=None):
    shape = x.shape
    weight, offset = prepared
    output = torch_npu.npu_weight_quant_batchmatmul(
        x.reshape(-1, shape[-1]),
        weight,
        scales,
        offset,
        bias=bias,
        antiquant_group_size=group_size if shape[-1] > group_size else 0,
    )
    return output.view(*shape[:-1], scales.shape[1])
