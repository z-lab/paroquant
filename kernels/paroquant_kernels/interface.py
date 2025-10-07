import torch


class RotateTensorFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, idx_ij, theta, scale=None):
        """
        x:      (B, ..., H)
        idx_ij: (KROT, H)
        theta:  (KROT, H//2)
        scale:  (H,) or None
        """

        ctx.orig_shape, ctx.orig_dtype = x.shape, x.dtype
        ctx.has_scale = scale is not None

        y = torch.ops.rotation.rotate(x, idx_ij, theta, scale)
        saved = (x, idx_ij, theta, y)
        if ctx.has_scale:
            saved += (scale,)
        ctx.save_for_backward(*saved)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        saved = list(ctx.saved_tensors)
        x, idx_ij, theta, y = saved[:4]
        scale = saved[4] if ctx.has_scale else None
        GROUP_SIZE = 128
        g = grad_out.view(-1, x.size(-1))

        KROT, H = idx_ij.shape
        num_groups = H // GROUP_SIZE
        B = y.numel() // H
        t = y.view(B, H)
        grad_theta = torch.zeros_like(theta)

        for i in range(KROT - 1, -1, -1):
            theta_neg = -theta[[i]]
            idx_chunk = idx_ij[[i]]
            t_prev = torch.ops.rotation.rotate(t, idx_chunk, theta_neg, None)
            g_prev = torch.ops.rotation.rotate(g, idx_chunk, theta_neg, None)
            # t_prev = apply_rotation_step(t, idx_chunk, theta_neg, GROUP_SIZE)
            # g_prev = apply_rotation_step(g, idx_chunk, theta_neg, GROUP_SIZE)
            idx_chunk = idx_chunk.view(num_groups, GROUP_SIZE)
            group_offsets = (
                torch.arange(num_groups, device=idx_ij.device, dtype=torch.int32)
                * GROUP_SIZE
            ).unsqueeze(1)
            di = idx_chunk[:, 0:GROUP_SIZE:2] + group_offsets
            dj = idx_chunk[:, 1:GROUP_SIZE:2] + group_offsets

            a = t_prev[:, di.reshape(-1)].view(
                t_prev.size(0), num_groups, GROUP_SIZE // 2
            )  # (B, G, GS/2)
            b = t_prev[:, dj.reshape(-1)].view(
                t_prev.size(0), num_groups, GROUP_SIZE // 2
            )
            ga = g[:, di.reshape(-1)].view(g.size(0), num_groups, GROUP_SIZE // 2)
            gb = g[:, dj.reshape(-1)].view(g.size(0), num_groups, GROUP_SIZE // 2)

            grad_s = (ga * b - gb * a).sum(dim=0)  # (G, GS/2)
            grad_c = (ga * a + gb * b).sum(dim=0)  # (G, GS/2)

            theta_chunk = theta[i]  # shape (H//2,)
            theta_groups = theta_chunk.view(num_groups, GROUP_SIZE // 2)  # (G, GS/2)
            s_vals = theta_groups.sin()
            c_vals = theta_groups.cos()

            grad_theta_chunk = grad_s * c_vals - grad_c * s_vals
            grad_theta[i] = grad_theta_chunk.reshape(-1).to(theta.dtype)
            t, g = t_prev, g_prev

        if ctx.has_scale:
            s = scale.unsqueeze(0)  # (1, H)
            grad_x_flat = g * s
            grad_scale = (x.view(B, H) * g).sum(dim=0).to(scale.dtype).to(scale.device)
        else:
            grad_x_flat, grad_scale = g, None

        grad_x = grad_x_flat.view(ctx.orig_shape).to(ctx.orig_dtype)

        return grad_x, None, grad_theta, grad_scale


def fast_givens_transform(x, idx_ij, theta, scales=None):
    return RotateTensorFunc.apply(x, idx_ij, theta, scales)
