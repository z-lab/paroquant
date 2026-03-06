from __future__ import annotations

import torch
import torch.nn as nn

# AutoAWQ's GEMM kernel imports PytorchGELUTanh from transformers.activations,
# but it was removed in transformers >=4.55. Provide a stub until AutoAWQ is updated.
import transformers.activations as _act

if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUActivation

from awq.modules.linear.gemm import WQLinearMMFunction


class RotateQuantizedLinear(nn.Module):
    """Pairwise Givens rotation + INT4 quantized matmul (AWQ GEMM kernel).

    All parameters are stored flat (no submodules), so state dict keys like
    ``gate_proj.theta`` and ``gate_proj.qweight`` match checkpoint naming directly.
    This enables native HuggingFace weight loading via ``from_pretrained``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        bits: int = 4,
        krot: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = bits
        self.group_size = group_size

        pack = 32 // bits
        n_groups = in_features // group_size

        # Rotation buffers
        self.register_buffer("theta", torch.zeros(krot, in_features // 2, dtype=torch.float16))
        self.register_buffer("pairs", torch.zeros(krot, in_features, dtype=torch.int16))
        self.register_buffer("channel_scales", torch.ones(1, in_features, dtype=torch.float16))

        # AWQ quantized weight buffers
        self.register_buffer("qweight", torch.zeros(in_features, out_features // pack, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros(n_groups, out_features // pack, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(n_groups, out_features, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float16, f"Expected float16 input, got {x.dtype}"
        x = torch.ops.rotation.rotate(x, self.pairs, self.theta, self.channel_scales)
        y = WQLinearMMFunction.apply(
            x,
            self.qweight,
            self.qzeros,
            self.scales,
            self.w_bit,
            self.group_size,
            self.bias,
            self.out_features,
        )
        return y.reshape(*x.shape[:-1], self.out_features)
