"""Unfused NPU references for the published ParoQuant checkpoint format."""

from types import MethodType

import torch


def unpack(packed):
    shifts = [0, 16, 4, 20, 8, 24, 12, 28]
    return torch.stack([(packed >> shift) & 15 for shift in shifts], dim=-1).flatten(-2)


def dequantize(module):
    weight = unpack(module.qweight).float()
    zeros = unpack(module.qzeros).repeat_interleave(module.group_size, dim=0).float()
    scales = module.scales.repeat_interleave(module.group_size, dim=0).float()
    return ((weight - zeros) * scales).half()


def prepare_reference_rotation(module):
    """Precompute constant indices and angles for the unfused reference."""
    groups = module.in_features // module.group_size
    offsets = torch.arange(groups, device=module.pairs.device)[:, None] * module.group_size
    stages = []
    for r in range(module.theta.shape[0]):
        pairs = module.pairs[r].long().view(groups, module.group_size) + offsets
        first, second = pairs[:, 0::2].flatten(), pairs[:, 1::2].flatten()
        angle = module.theta[r].float()
        stages.append((first, second, angle.cos(), angle.sin()))
    return stages


def rotation(x, module):
    """Materialize each stage with the checkpoint's pair order and FP16 rounding."""
    if not hasattr(module, "_unfused_stages"):
        module._unfused_stages = prepare_reference_rotation(module)
    y = (x * module.channel_scales).half()
    for first, second, cosine, sine in module._unfused_stages:
        a, b = y[..., first].float(), y[..., second].float()
        out = torch.empty_like(y)
        out[..., first] = (a * cosine + b * sine).half()
        out[..., second] = (b * cosine - a * sine).half()
        y = out
    return y


@torch.no_grad()
def reference_forward(self, x):
    if self._reference_rotation is None:
        x = rotation(x, self)
    else:
        from paroquant.kernels.ascend import rotate

        x = rotate(x, self.channel_scales, self._reference_rotation)
    weight = self._reference_weight if self._reference_weight is not None else dequantize(self)
    out = torch.matmul(x, weight)
    return out + self.bias if self.bias is not None else out


def install_reference(model, cache_weights=False, fuse_rotation=False):
    from paroquant.inference.backends.transformers.modules import RotateQuantizedLinear
    from paroquant.kernels.ascend import prepare_rotation

    for module in model.modules():
        if isinstance(module, RotateQuantizedLinear):
            weight = dequantize(module) if cache_weights else None
            module.register_buffer("_reference_weight", weight, persistent=False)
            if cache_weights:
                module.qweight = None
                module.qzeros = None
                module.scales = None
            if not fuse_rotation:
                module._unfused_stages = prepare_reference_rotation(module)
            module._reference_rotation = prepare_rotation(module.pairs, module.theta) if fuse_rotation else None
            module.forward = MethodType(reference_forward, module)
