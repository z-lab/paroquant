"""Quantization primitives: STE rounding/clamping and pseudo-quantization."""

from __future__ import annotations

import torch


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for rounding."""
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
    """Straight-through estimator for clamping."""
    return (x.clamp(min, max) - x).detach() + x


def pseudo_quantize_tensor(
    w: torch.Tensor,
    n_bit: int = 4,
    q_group_size: int = -1,
    get_scale_zp: bool = False,
    qzeros: torch.Tensor | None = None,
    qscales: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric uniform quantization with STE.

    If qscales/qzeros are provided, uses them directly. Otherwise computes from data.
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    max_int = 2**n_bit - 1
    min_int = 0

    if qscales is None:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        qscales = (max_val - min_val).clamp(min=1e-5) / max_int
        qzeros = (-torch.round(min_val / qscales)).clamp_(min_int, max_int)

    w = (torch.clamp(torch.round(w / qscales) + qzeros, min_int, max_int) - qzeros) * qscales
    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, qscales.view(w.shape[0], -1), qzeros.view(w.shape[0], -1)
    return w
