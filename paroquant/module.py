import torch
from torch import nn
from typing import Optional
from torch.utils.checkpoint import checkpoint

from .util import get_named_linears
from .quantizer import UniformAffineQuantizer

from paroquant_kernels import scaled_pairwise_rotation


class PseudoQuantizedLinear(nn.Module):
    """Pseudo-quantized linear layer using custom kernels for fine-tuning."""

    def __init__(
        self,
        linear: nn.Linear,
        rotation_pairs: Optional[list[torch.Tensor]],
        channel_scales: Optional[torch.Tensor],
        *,
        group_size: int,
        n_bits: int,
        num_rotations: int,
    ) -> None:
        super().__init__()
        self.enable_checkpoint = False
        self.weight = nn.Parameter(linear.weight.clone())
        self.in_feat = self.weight.shape[1]
        self.out_feat = self.weight.shape[0]
        num_groups = self.in_feat // group_size
        assert self.in_feat % group_size == 0
        assert self.weight.dtype == torch.float16 or self.weight.dtype == torch.float32
        if rotation_pairs is not None:
            pairs_grouped, angles_grouped, mask = rotation_pairs
            assert pairs_grouped.size(0) == num_rotations
            assert angles_grouped.size(0) == num_rotations
            assert mask.shape == angles_grouped.shape

        num_rotations = (
            torch.tensor(num_rotations, device="cuda")
            if not isinstance(num_rotations, torch.Tensor)
            else num_rotations
        )
        self.register_buffer("num_rotations", num_rotations)

        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.clone())
        if rotation_pairs is None:
            # generate dummy pairs & angles if rotation_pairs is None
            angles_grouped = torch.zeros(
                num_rotations, self.in_feat // 2, device="cuda", dtype=self.weight.dtype
            )
            single_group_idx_ij = torch.randperm(
                group_size, device="cuda", dtype=torch.short
            )
            one_krot_layer_idx_ij = single_group_idx_ij.repeat(num_groups)
            pairs_grouped = (
                one_krot_layer_idx_ij.unsqueeze(0)
                .expand(num_rotations, -1)
                .contiguous()
            )
            mask = torch.zeros_like(angles_grouped, device="cuda", dtype=torch.bool)
        else:
            pairs_grouped, angles_grouped, mask = rotation_pairs
        self.pairs_grouped = nn.Parameter(pairs_grouped.clone(), requires_grad=False)
        self.angles_grouped = nn.Parameter(angles_grouped.clone())

        self.register_buffer("mask", mask)
        if channel_scales is None:
            # generate dummy channel scales
            channel_scales = torch.ones(
                self.in_feat, device="cuda", dtype=self.weight.dtype
            )
        self.channel_scales = nn.Parameter(channel_scales)
        assert self.pairs_grouped.dtype == torch.short, self.pairs_grouped.dtype
        assert self.channel_scales.dtype == self.weight.dtype, (
            self.channel_scales.dtype,
            self.weight.dtype,
        )
        assert self.mask.dtype == torch.bool, self.mask.dtype

        self.quantizer: Optional[UniformAffineQuantizer] = None
        self.register_buffer("quantizer_optim_enabled", torch.tensor(False))
        n_bits = n_bits if isinstance(n_bits, torch.Tensor) else torch.tensor(n_bits)
        group_size = (
            group_size
            if isinstance(group_size, torch.Tensor)
            else torch.tensor(group_size)
        )
        self.register_buffer("n_bits", n_bits)
        self.register_buffer("group_size", group_size)

    def forward(
        self, x: torch.Tensor, weight_update: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        weight = self.weight
        if weight_update is not None:
            weight = weight + weight_update
        weight = self._pseudo_quantize(weight)
        x = self.checkpointed(torch.matmul, x, weight.T)
        if getattr(self, "bias", None) is not None:
            x = x + self.bias
        return x

    def _pseudo_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        weight = weight * self.channel_scales
        weight = self.checkpointed(
            scaled_pairwise_rotation, weight, self.pairs_grouped, self.angles_grouped
        )
        if self.quantizer_optim_enabled or self.quantizer is not None:
            assert self.quantizer is not None, "Quantizer should be initialized."
            self.quantizer.enable_checkpoint = self.enable_checkpoint
            weight = self.quantizer(weight)
        else:
            weight = self.checkpointed(
                UniformAffineQuantizer.pseudo_quantize,
                weight,
                self.n_bits,
                self.group_size,
            )
        if self.pairs_grouped is not None:
            flipped_pairs = torch.flip(self.pairs_grouped, dims=[0])
            flipped_angles = torch.flip(self.angles_grouped, dims=[0])
            weight = self.checkpointed(
                scaled_pairwise_rotation, weight, flipped_pairs, -flipped_angles
            )
        if self.channel_scales is not None:
            weight = weight / self.channel_scales.view(1, -1)
        return weight

    def set_optim_enabled(
        self,
        *,
        weight: bool = False,
        bias: bool = False,
        angles: bool = False,
        channel_scales: bool = False,
        quantizer: bool = False,
    ) -> None:
        self.weight.requires_grad = weight
        if self.bias is not None:
            self.bias.requires_grad = bias
        if self.angles_grouped is not None:
            self.angles_grouped.requires_grad = angles
        if self.channel_scales is not None:
            self.channel_scales.requires_grad = channel_scales

        if quantizer and self.quantizer is None:
            # Initialize the quantizer with the current rotated weight
            weight = self.weight
            if self.channel_scales is not None:
                weight = weight * self.channel_scales.view(1, -1)
            if self.pairs_grouped is not None:
                weight = scaled_pairwise_rotation(
                    weight, self.pairs_grouped, self.angles_grouped
                )

            self.quantizer = UniformAffineQuantizer(
                weight,
                n_bits=self.n_bits,
                group_size=self.group_size,
            )

        if self.quantizer:
            self.quantizer.set_optim_enabled(quantizer)
        self.quantizer_optim_enabled = torch.tensor(quantizer)

    def get_optim_params(self, name: str) -> list[nn.Parameter]:
        if name == "weight":
            return [self.weight]
        elif name == "bias":
            return [self.bias]
        elif name == "angles":
            if self.angles_grouped is None:
                return []
            return [self.angles_grouped]
        elif name == "channel_scales":
            if self.channel_scales is None:
                return []
            return [self.channel_scales]
        elif name == "quantizer":
            return self.quantizer.optim_params()
        else:
            raise ValueError(f"Unknown parameter group: {name}")

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "PseudoQuantizedLinear":
        weight: torch.Tensor = state_dict["weight"]
        has_bias: bool = "bias" in state_dict and state_dict["bias"] is not None
        linear = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=has_bias,
            device=weight.device,
            dtype=weight.dtype,
        )
        linear.weight.data.copy_(weight)
        if has_bias:
            linear.bias.data.copy_(state_dict["bias"])

        group_size = state_dict["group_size"]
        n_bits = state_dict["n_bits"]
        channel_scales = state_dict["channel_scales"]
        num_rotations = state_dict["num_rotations"]

        pairs_grouped = state_dict["pairs_grouped"]
        angles_grouped = state_dict["angles_grouped"]
        mask = state_dict.get(
            "mask", torch.zeros_like(angles_grouped, dtype=torch.bool)
        )

        qlinear = PseudoQuantizedLinear(
            linear,
            [pairs_grouped, angles_grouped, mask],
            channel_scales,
            group_size=group_size,
            n_bits=n_bits,
            num_rotations=num_rotations,
        )

        # Initialize the quantizer
        if "quantizer.scale" in state_dict:
            qlinear.set_optim_enabled(quantizer=True)

        qlinear.load_state_dict(state_dict)
        return qlinear

    @torch.no_grad()
    def reset_angles_by_mask(self):
        self.angles_grouped.data[self.mask] = 0

    def checkpointed(self, fn, *args) -> torch.Tensor:
        if torch.is_grad_enabled() and self.enable_checkpoint:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    @torch.no_grad()
    def pseudo_weight(self) -> torch.Tensor:
        return self._pseudo_quantize(self.weight).detach()


@torch.no_grad()
def reset_angles_by_mask(module: nn.Module) -> None:
    linears = get_named_linears(module, PseudoQuantizedLinear)
    for linear in linears.values():
        linear.reset_angles_by_mask()
