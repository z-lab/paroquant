"""ParoQuant modules for PyTorch: W4A16Linear, Rotation, RotateLinearW4A16."""

from __future__ import annotations

import torch
import torch.nn as nn


class W4A16Linear(nn.Module):
    """Weight-only INT4 quantized linear with FP16 activations (AutoAWQ int32 format).

    Dequantizes weights to FP16 on first forward pass and caches for reuse.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 bits: int = 4, group_size: int = 128, device: str = "cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        pack = 32 // bits

        self.register_buffer("qweight", torch.zeros(in_features, out_features // pack, dtype=torch.int32, device=device))
        self.register_buffer("qzeros", torch.zeros(in_features // group_size, out_features // pack, dtype=torch.int32, device=device))
        self.register_buffer("scales", torch.zeros(in_features // group_size, out_features, dtype=torch.float16, device=device))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16, device=device))
        else:
            self.bias = None
        self._weight_fp: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_fp is None:
            self._weight_fp = self._dequantize(x.dtype)
        out = x @ self._weight_fp
        if self.bias is not None:
            out = out + self.bias
        return out

    def _dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        mask = (1 << self.bits) - 1
        shifts = torch.arange(0, 32, self.bits, device=self.qweight.device)

        qw = (self.qweight.unsqueeze(-1) >> shifts) & mask
        qw = qw.reshape(self.in_features, self.out_features)

        qz = (self.qzeros.unsqueeze(-1) >> shifts) & mask
        qz = qz.reshape(-1, self.out_features)

        scales = self.scales.repeat_interleave(self.group_size, dim=0)
        zeros = qz.repeat_interleave(self.group_size, dim=0)

        return ((qw.float() - zeros.float()) * scales.float()).to(dtype)


class Rotation(nn.Module):
    """Pairwise Givens rotation via the CUDA rotation kernel."""

    def __init__(self, dim: int, krot: int = 8, group_size: int = 128,
                 rotation_angles: torch.Tensor | None = None,
                 rotation_pairs: torch.Tensor | None = None,
                 channel_scales: torch.Tensor | None = None,
                 device: str = "cuda"):
        super().__init__()
        num_groups = dim // group_size

        if rotation_angles is None:
            rotation_angles = torch.zeros(krot, dim // 2, device=device, dtype=torch.float16)
        if rotation_pairs is None:
            single = torch.randperm(group_size, dtype=torch.int, device=device)
            rotation_pairs = single.repeat(num_groups).unsqueeze(0).expand(krot, -1)
        if channel_scales is None:
            channel_scales = torch.ones(1, dim, dtype=torch.half, device=device)

        self.register_buffer("theta", rotation_angles.clone().contiguous())
        self.register_buffer("pairs", rotation_pairs.clone().short().contiguous())
        self.register_buffer("channel_scales", channel_scales.clone().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.rotation.rotate(x, self.pairs, self.theta, self.channel_scales)


class RotateLinearW4A16(nn.Module):
    """Rotation + W4A16 quantized linear: pairwise Givens rotation then dequantized matmul."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 128, bits: int = 4,
                 rotation_angles: torch.Tensor | None = None,
                 rotation_pairs: torch.Tensor | None = None,
                 channel_scales: torch.Tensor | None = None,
                 device: str = "cuda"):
        super().__init__()
        self.rotation = Rotation(
            in_features, rotation_angles=rotation_angles,
            rotation_pairs=rotation_pairs, channel_scales=channel_scales,
            device=device,
        )
        self.qlinear = W4A16Linear(in_features, out_features, bias,
                                    bits=bits, group_size=group_size, device=device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlinear(self.rotation(x))
