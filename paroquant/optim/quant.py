"""Quantization primitives: STE rounding and clamping."""

from __future__ import annotations

import torch


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for rounding."""
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
    """Straight-through estimator for clamping."""
    return (x.clamp(min, max) - x).detach() + x
