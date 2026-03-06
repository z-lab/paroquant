# Copyright (c) 2025, Haisheng Chen.

import torch


class RotateTensorFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx_ij, theta, scale=None, group_size=128):
        ctx.orig_shape = x.shape
        ctx.orig_dtype = x.dtype
        ctx.has_scale = scale is not None
        ctx.group_size = group_size

        y = torch.ops.rotation.rotate(x, idx_ij, theta, scale, group_size)
        saved = (x, idx_ij, theta, y, scale) if ctx.has_scale else (x, idx_ij, theta, y)
        ctx.save_for_backward(*saved)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, idx_ij, theta, y = ctx.saved_tensors[:4]
        scale = ctx.saved_tensors[4] if ctx.has_scale else None
        gs = ctx.group_size

        KROT, H = idx_ij.shape
        num_groups = H // gs
        half_gs = gs // 2
        B = y.numel() // H
        t = y.view(B, H)
        g = grad_out.view(-1, H)
        grad_theta = torch.zeros_like(theta)
        offsets = torch.arange(num_groups, device=idx_ij.device, dtype=torch.int32).unsqueeze(1) * gs

        for i in range(KROT - 1, -1, -1):
            neg_theta = -theta[[i]]
            idx = idx_ij[[i]]
            t = torch.ops.rotation.rotate(t, idx, neg_theta, None, gs)
            g = torch.ops.rotation.rotate(g, idx, neg_theta, None, gs)

            idx = idx.view(num_groups, gs)
            di = (idx[:, 0::2] + offsets).reshape(-1)
            dj = (idx[:, 1::2] + offsets).reshape(-1)

            a = t[:, di].view(B, num_groups, half_gs)
            b = t[:, dj].view(B, num_groups, half_gs)
            ga = g[:, di].view(B, num_groups, half_gs)
            gb = g[:, dj].view(B, num_groups, half_gs)

            sin_t, cos_t = theta[i].view(num_groups, half_gs).sin(), theta[i].view(num_groups, half_gs).cos()
            grad_theta[i] = (
                ((ga * b - gb * a).sum(0) * cos_t - (ga * a + gb * b).sum(0) * sin_t).reshape(-1).to(theta.dtype)
            )

        if ctx.has_scale:
            grad_x = (g * scale.unsqueeze(0)).view(ctx.orig_shape).to(ctx.orig_dtype)
            grad_scale = (x.view(B, H) * g).sum(0).to(dtype=scale.dtype, device=scale.device)
        else:
            grad_x = g.view(ctx.orig_shape).to(ctx.orig_dtype)
            grad_scale = None

        return grad_x, None, grad_theta, grad_scale, None


def scaled_pairwise_rotation(x, idx_ij, theta, scales=None, group_size=128):
    return RotateTensorFunc.apply(x, idx_ij, theta, scales, group_size)
