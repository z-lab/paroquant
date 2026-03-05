"""ParoQuant modules for PyTorch: Rotation and RotateQuantizedLinear."""

from __future__ import annotations

import torch
import torch.nn as nn

# AutoAWQ is deprecated but its GEMM kernel still works.
# Patch the missing symbol that was removed in transformers >=4.55.
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUActivation

from awq.modules.linear.gemm import WQLinear_GEMM as _AWQLinear


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


class RotateQuantizedLinear(nn.Module):
    """Pairwise Givens rotation + quantized matmul (AWQ GEMM kernel)."""

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
        self.qlinear = _AWQLinear(bits, group_size, in_features, out_features, bias, dev=device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlinear(self.rotation(x))
