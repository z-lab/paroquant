import torch
import torch.nn as nn
from typing import Optional, Union
from torch.utils.checkpoint import checkpoint

from .util import clamp_ste, round_ste


def _calc_scales_and_zero_points(
    weight: torch.Tensor, group_size: int, qmin: int, qmax: int
) -> None:
    assert weight.dim() == 2, weight.shape
    if weight.dtype != torch.float32:
        weight = weight.float()
    x = weight.reshape(-1, group_size)
    min_val = x.amin(dim=1, keepdim=True)
    max_val = x.amax(dim=1, keepdim=True)
    scale = clamp_ste(max_val - min_val, min=1e-5) / qmax
    zero_point = min_val / scale

    assert torch.isnan(zero_point).sum() == 0, zero_point

    return scale, zero_point


# Adapted from EfficientQAT
class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_bits: int,
        group_size: int,
    ):
        super().__init__()
        # self.n_bits = n_bits
        if not isinstance(n_bits, torch.Tensor):
            n_bits = torch.tensor(n_bits)
        if not isinstance(group_size, torch.Tensor):
            group_size = torch.tensor(group_size)
        self.register_buffer("n_bits", n_bits)
        self.register_buffer("group_size", group_size)
        assert weight.shape[-1] % group_size == 0

        scale, zero_point_float = _calc_scales_and_zero_points(
            weight, group_size, self.qmin, self.qmax
        )
        self.scale = nn.Parameter(scale)
        self.zero_point_float = nn.Parameter(zero_point_float)

        self.enable_checkpoint = False

    @property
    def qmin(self) -> int:
        return 0

    @property
    def qmax(self) -> torch.Tensor:
        return 2**self.n_bits - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.enable_checkpoint:
            return checkpoint(
                self.pseudo_quantize,
                x,
                self.n_bits,
                self.group_size,
                self.scale,
                self.zero_point_float,
                use_reentrant=False,
            )
        else:
            return self.pseudo_quantize(
                x,
                self.n_bits,
                self.group_size,
                self.scale,
                self.zero_point_float,
            )

    def optim_params(self) -> list[nn.Parameter]:
        return [self.scale, self.zero_point_float]

    def set_optim_enabled(self, enabled: bool):
        for param in self.optim_params():
            param.requires_grad = enabled

    @staticmethod
    def pseudo_quantize(
        x: torch.Tensor,
        n_bits: Union[int, torch.Tensor],
        group_size: Union[int, torch.Tensor],
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()
        assert torch.isnan(x).sum() == 0, x

        qmin, qmax = 0, 2**n_bits - 1
        if scale is None or zero_point is None:
            scale, zero_point = _calc_scales_and_zero_points(x, group_size, qmin, qmax)

        scale = clamp_ste(scale, min=1e-5, max=1e5)
        round_zero_point = clamp_ste(-round_ste(zero_point), qmin, qmax)
        dim1, dim2 = x.shape
        x = x.reshape(-1, group_size)
        x_int = round_ste(x / scale)
        assert torch.isnan(x_int).sum() == 0, (x_int, scale.min(), scale.max())
        x_int = x_int + round_zero_point
        x_int = clamp_ste(x_int, qmin, qmax)
        x_dequant = x_int
        x_dequant = x_dequant - round_zero_point
        x_dequant = x_dequant * scale
        x_dequant = x_dequant.reshape(dim1, dim2)
        assert torch.isnan(x_dequant).sum() == 0, x_dequant
        assert torch.isinf(x_dequant).sum() == 0, x_dequant
        return x_dequant.to(dtype)
