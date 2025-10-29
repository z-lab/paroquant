import torch


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def pseudo_quantize_tensor(
    w,
    n_bit=8,
    zero_point=True,
    q_group_size=-1,
    inplace=False,
    get_scale_zp=False,
    qzeros=None,
    qscales=None,
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    assert (qscales == None) == (qzeros == None)
    if qscales == None:
        if zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**n_bit - 1
            min_int = 0
            qscales = (max_val - min_val).clamp(min=1e-5) / max_int
            qzeros = (-torch.round(min_val / qscales)).clamp_(min_int, max_int)
        else:  # we actually never used this
            assert min_val is None
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))
            qscales = max_val / max_int
            qzeros = 0

    assert torch.isnan(qscales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(qscales).round_().add_(qzeros))
            .clamp_(min_int, max_int)
            .sub_(qzeros)
        ).mul_(qscales)
    else:
        max_int = 2**n_bit - 1
        min_int = 0
        w = (
            torch.clamp(torch.round(w / qscales) + qzeros, min_int, max_int) - qzeros
        ) * qscales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, qscales.view(w.shape[0], -1), qzeros.view(w.shape[0], -1)
    else:
        return w
