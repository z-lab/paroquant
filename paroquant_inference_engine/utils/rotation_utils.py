import torch
import gc
import torch.nn as nn
from .quantization_utils import pseudo_quantize_tensor
from paroquant_inference_engine.model_executor.modules.qmodule import WQLinear


def apply_rotation_step(
    t: torch.Tensor, idx_ij: torch.Tensor, theta: torch.Tensor, group_size: int
):
    t = t.view(-1, t.size(-1)).clone()
    hidden = t.size(-1)
    n_groups = hidden // group_size
    half = group_size // 2
    s = torch.sin(theta).view(n_groups, half).to(dtype=t.dtype)
    c = torch.cos(theta).view(n_groups, half).to(dtype=t.dtype)
    idx_ij = idx_ij.to(dtype=torch.int32)
    for g in range(n_groups):
        start = g * group_size
        di = idx_ij[start : start + group_size : 2] + start
        dj = idx_ij[start + 1 : start + group_size : 2] + start
        s_g = s[g].unsqueeze(0)
        c_g = c[g].unsqueeze(0)

        ai = t[:, di]
        bi = t[:, dj]
        t[:, di] = c_g * ai + s_g * bi
        t[:, dj] = -s_g * ai + c_g * bi
    return t


def rotate_tensor(
    tensor: torch.Tensor,
    rotation_angles,
    rotation_pairs,
    channel_scales,
    permu=None,
    divide_scale=True,
):

    if tensor == None:
        return tensor
    assert rotation_pairs == None or tensor.size(-1) == rotation_pairs.size(-1)
    assert rotation_angles == None or tensor.size(-1) // 2 == rotation_angles.size(-1)
    assert tensor.size(-1) == channel_scales.size(-1)
    assert permu == None or tensor.size(-1) == permu.size(-1)
    tensor_orig_shape = tensor.shape
    tensor_orig_dtype = tensor.dtype
    tensor = tensor.cuda().float().view(-1, tensor.size(-1))

    KROT = rotation_pairs.size(0)
    GROUP_SIZE = 128

    if permu != None:
        permu = permu.to(dtype=torch.int32).cuda()
        tensor = tensor.index_select(-1, permu)

    if channel_scales != None:
        channel_scales = channel_scales.to(dtype=torch.float).cuda()
        tensor = tensor.to(channel_scales.dtype)
        if divide_scale:
            tensor = tensor / channel_scales
        else:
            tensor = tensor * channel_scales

    if rotation_pairs != None and rotation_angles != None:
        rotation_pairs = rotation_pairs.to(dtype=torch.int32).cuda()
        rotation_angles = rotation_angles.to(dtype=torch.float).cuda()
        tensor = tensor.to(rotation_angles.dtype)
        for i in range(KROT):
            tensor = apply_rotation_step(
                tensor, rotation_pairs[i], rotation_angles[i], GROUP_SIZE
            )
    tensor = tensor.to(dtype=tensor_orig_dtype).view(tensor_orig_shape)
    torch.cuda.empty_cache()
    gc.collect()
    return tensor


@torch.no_grad()
def quantizer(linear: nn.Linear, w_bit, q_grpsize, qscales=None, qzeros=None):
    dev = linear.weight.device
    linear.cuda()
    orig_weight = linear.weight.data.detach().clone()
    assert (qscales == None) == (qzeros == None)
    if qscales == None:
        weight_data, qscales, qzeros = pseudo_quantize_tensor(
            linear.weight.data, n_bit=w_bit, get_scale_zp=True, q_group_size=q_grpsize
        )
        linear.weight.copy_(weight_data)
    q_linear = WQLinear.from_linear(linear, w_bit, q_grpsize, False, qscales, qzeros)
    linear.weight.copy_(orig_weight)
    linear.to(dev)
    torch.cuda.empty_cache()
    gc.collect()

    return q_linear
