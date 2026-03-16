from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

from paroquant.kernels.cuda import scaled_pairwise_rotation

from .quantizer import UniformAffineQuantizer


def get_named_qwen3_5_moe_experts(module: nn.Module) -> dict[str, Qwen3_5MoeExperts]:
    return {name: m for name, m in module.named_modules() if isinstance(m, Qwen3_5MoeExperts)}


class PseudoQuantizedQwen3_5MoeExperts(nn.Module):
    """Pseudo-quantized fused MoE experts for Qwen3.5-MoE."""

    def __init__(
        self,
        experts: Qwen3_5MoeExperts,
        gate_up_rotation_pairs: Optional[list[torch.Tensor]],
        down_rotation_pairs: Optional[list[torch.Tensor]],
        gate_up_channel_scales: Optional[torch.Tensor],
        down_channel_scales: Optional[torch.Tensor],
        *,
        group_size: int,
        n_bits: int,
        num_rotations: int,
    ) -> None:
        super().__init__()
        self.enable_checkpoint = False
        self.config = experts.config
        self.num_experts = experts.num_experts
        self.hidden_dim = experts.hidden_dim
        self.intermediate_dim = experts.intermediate_dim
        self.act_fn = ACT2FN[self.config.hidden_act]

        # Keep compatibility with current transformers grouped expert interfaces.
        self.has_gate = getattr(experts, "has_gate", False)
        self.has_bias = False
        self.is_transposed = False

        self.gate_up_weight = nn.Parameter(experts.gate_up_proj.clone())
        self.down_weight = nn.Parameter(experts.down_proj.clone())
        assert self.gate_up_weight.ndim == 3
        assert self.down_weight.ndim == 3
        assert self.gate_up_weight.shape[-1] % group_size == 0
        assert self.down_weight.shape[-1] % group_size == 0

        n_bits = (
            n_bits.to(device=self.gate_up_weight.device)
            if isinstance(n_bits, torch.Tensor)
            else torch.tensor(n_bits, device=self.gate_up_weight.device)
        )
        group_size = (
            group_size.to(device=self.gate_up_weight.device)
            if isinstance(group_size, torch.Tensor)
            else torch.tensor(group_size, device=self.gate_up_weight.device)
        )
        num_rotations = (
            num_rotations.to(device=self.gate_up_weight.device)
            if isinstance(num_rotations, torch.Tensor)
            else torch.tensor(num_rotations, device=self.gate_up_weight.device)
        )
        self.register_buffer("n_bits", n_bits)
        self.register_buffer("group_size", group_size)
        self.register_buffer("num_rotations", num_rotations)

        gate_up_pairs, gate_up_angles, gate_up_mask = self._init_rotation(self.gate_up_weight, gate_up_rotation_pairs)
        down_pairs, down_angles, down_mask = self._init_rotation(self.down_weight, down_rotation_pairs)

        self.gate_up_pairs_grouped = nn.Parameter(gate_up_pairs, requires_grad=False)
        self.gate_up_angles_grouped = nn.Parameter(gate_up_angles)
        self.register_buffer("gate_up_mask", gate_up_mask)

        self.down_pairs_grouped = nn.Parameter(down_pairs, requires_grad=False)
        self.down_angles_grouped = nn.Parameter(down_angles)
        self.register_buffer("down_mask", down_mask)

        if gate_up_channel_scales is None:
            gate_up_channel_scales = torch.ones(
                1,
                self.gate_up_weight.shape[-1],
                dtype=self.gate_up_weight.dtype,
                device=self.gate_up_weight.device,
            )
        if down_channel_scales is None:
            down_channel_scales = torch.ones(
                1,
                self.down_weight.shape[-1],
                dtype=self.down_weight.dtype,
                device=self.down_weight.device,
            )
        gate_up_channel_scales = gate_up_channel_scales.to(
            device=self.gate_up_weight.device,
            dtype=self.gate_up_weight.dtype,
        )
        down_channel_scales = down_channel_scales.to(
            device=self.down_weight.device,
            dtype=self.down_weight.dtype,
        )
        self.gate_up_channel_scales = nn.Parameter(gate_up_channel_scales)
        self.down_channel_scales = nn.Parameter(down_channel_scales)

        self.gate_up_quantizer: Optional[UniformAffineQuantizer] = None
        self.down_quantizer: Optional[UniformAffineQuantizer] = None
        self.register_buffer("gate_up_quantizer_optim_enabled", torch.tensor(False))
        self.register_buffer("down_quantizer_optim_enabled", torch.tensor(False))

    def _init_rotation(
        self,
        weight: torch.Tensor,
        rotation_pairs: Optional[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        in_feat = weight.shape[-1]
        group_size = int(self.group_size.item())
        num_groups = in_feat // group_size
        assert in_feat % group_size == 0

        if rotation_pairs is not None:
            pairs_grouped, angles_grouped, mask = rotation_pairs
            pairs_grouped = pairs_grouped.to(device=weight.device, dtype=torch.short).contiguous()
            angles_grouped = angles_grouped.to(device=weight.device, dtype=weight.dtype).contiguous()
            mask = mask.to(device=weight.device, dtype=torch.bool)
            assert pairs_grouped.size(0) == self.num_rotations
            assert angles_grouped.size(0) == self.num_rotations
            assert mask.shape == angles_grouped.shape
            return pairs_grouped.clone(), angles_grouped.clone(), mask

        device = weight.device
        dtype = weight.dtype
        angles_grouped = torch.zeros(
            self.num_rotations,
            in_feat // 2,
            device=device,
            dtype=dtype,
        )
        single_group_idx_ij = torch.randperm(group_size, device=device, dtype=torch.short)
        one_krot_layer_idx_ij = single_group_idx_ij.repeat(num_groups)
        pairs_grouped = one_krot_layer_idx_ij.unsqueeze(0).expand(self.num_rotations, -1).contiguous()
        mask = torch.zeros_like(angles_grouped, device=device, dtype=torch.bool)
        return pairs_grouped, angles_grouped, mask

    def checkpointed(self, fn, *args) -> torch.Tensor:
        if torch.is_grad_enabled() and self.enable_checkpoint:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _pseudo_quantize_weight(
        self,
        weight_3d: torch.Tensor,
        pairs_grouped: torch.Tensor,
        angles_grouped: torch.Tensor,
        channel_scales: torch.Tensor,
        quantizer: Optional[UniformAffineQuantizer],
        quantizer_optim_enabled: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = weight_3d.shape
        weight = weight_3d.reshape(-1, original_shape[-1])

        weight = weight * channel_scales
        weight = self.checkpointed(
            scaled_pairwise_rotation,
            weight,
            pairs_grouped,
            angles_grouped,
            None,
            self.group_size,
        )

        if bool(quantizer_optim_enabled.item()) or quantizer is not None:
            assert quantizer is not None
            quantizer.enable_checkpoint = self.enable_checkpoint
            weight = quantizer(weight)
        else:
            weight = self.checkpointed(
                UniformAffineQuantizer.pseudo_quantize,
                weight,
                self.n_bits,
                self.group_size,
            )

        flipped_pairs = torch.flip(pairs_grouped, dims=[0])
        flipped_angles = torch.flip(angles_grouped, dims=[0])
        weight = self.checkpointed(
            scaled_pairwise_rotation,
            weight,
            flipped_pairs,
            -flipped_angles,
            None,
            self.group_size,
        )
        weight = weight / channel_scales
        return weight.view(*original_shape)

    @property
    def gate_up_proj(self) -> torch.Tensor:
        return self._pseudo_quantize_weight(
            self.gate_up_weight,
            self.gate_up_pairs_grouped,
            self.gate_up_angles_grouped,
            self.gate_up_channel_scales,
            self.gate_up_quantizer,
            self.gate_up_quantizer_optim_enabled,
        )

    @property
    def down_proj(self) -> torch.Tensor:
        return self._pseudo_quantize_weight(
            self.down_weight,
            self.down_pairs_grouped,
            self.down_angles_grouped,
            self.down_channel_scales,
            self.down_quantizer,
            self.down_quantizer_optim_enabled,
        )

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        gate_up_proj = self.gate_up_proj
        down_proj = self.down_proj

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        experts_forward = ALL_EXPERTS_FUNCTIONS.get_interface(
            self.config._experts_implementation,
            self._eager_forward,
        )
        return experts_forward(self, hidden_states, top_k_index, top_k_weights)

    def set_optim_enabled(
        self,
        *,
        weight: bool = False,
        angles: bool = False,
        channel_scales: bool = False,
        quantizer: bool = False,
    ) -> None:
        self.gate_up_weight.requires_grad = weight
        self.down_weight.requires_grad = weight

        self.gate_up_angles_grouped.requires_grad = angles
        self.down_angles_grouped.requires_grad = angles

        self.gate_up_channel_scales.requires_grad = channel_scales
        self.down_channel_scales.requires_grad = channel_scales

        if quantizer and self.gate_up_quantizer is None:
            gate_up_weight = self.gate_up_weight.reshape(-1, self.gate_up_weight.shape[-1])
            gate_up_weight = gate_up_weight * self.gate_up_channel_scales
            gate_up_weight = scaled_pairwise_rotation(
                gate_up_weight,
                self.gate_up_pairs_grouped,
                self.gate_up_angles_grouped,
                None,
                self.group_size,
            )
            self.gate_up_quantizer = UniformAffineQuantizer(
                gate_up_weight,
                n_bits=self.n_bits,
                group_size=self.group_size,
            )

        if quantizer and self.down_quantizer is None:
            down_weight = self.down_weight.reshape(-1, self.down_weight.shape[-1])
            down_weight = down_weight * self.down_channel_scales
            down_weight = scaled_pairwise_rotation(
                down_weight,
                self.down_pairs_grouped,
                self.down_angles_grouped,
                None,
                self.group_size,
            )
            self.down_quantizer = UniformAffineQuantizer(
                down_weight,
                n_bits=self.n_bits,
                group_size=self.group_size,
            )

        if self.gate_up_quantizer is not None:
            self.gate_up_quantizer.set_optim_enabled(quantizer)
        if self.down_quantizer is not None:
            self.down_quantizer.set_optim_enabled(quantizer)

        self.gate_up_quantizer_optim_enabled.fill_(quantizer)
        self.down_quantizer_optim_enabled.fill_(quantizer)

    def get_optim_params(self, name: str) -> list[nn.Parameter]:
        if name == "weight":
            return [self.gate_up_weight, self.down_weight]
        if name == "angles":
            return [self.gate_up_angles_grouped, self.down_angles_grouped]
        if name == "channel_scales":
            return [self.gate_up_channel_scales, self.down_channel_scales]
        if name == "quantizer":
            params: list[nn.Parameter] = []
            if self.gate_up_quantizer is not None:
                params.extend(self.gate_up_quantizer.optim_params())
            if self.down_quantizer is not None:
                params.extend(self.down_quantizer.optim_params())
            return params
        if name == "bias":
            return []
        raise ValueError(f"Unknown parameter group: {name}")

    @torch.no_grad()
    def reset_angles_by_mask(self) -> None:
        self.gate_up_angles_grouped.data[self.gate_up_mask] = 0
        self.down_angles_grouped.data[self.down_mask] = 0

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        template: Qwen3_5MoeExperts,
        device: torch.device | str,
    ) -> "PseudoQuantizedQwen3_5MoeExperts":
        target_device = torch.device(device)
        target_dtype = template.gate_up_proj.dtype

        gate_up_angles = state_dict["gate_up_angles_grouped"].to(device=target_device, dtype=target_dtype)
        down_angles = state_dict["down_angles_grouped"].to(device=target_device, dtype=target_dtype)
        gate_up_pairs = state_dict["gate_up_pairs_grouped"].to(device=target_device, dtype=torch.short)
        down_pairs = state_dict["down_pairs_grouped"].to(device=target_device, dtype=torch.short)
        gate_up_mask = state_dict.get("gate_up_mask", torch.zeros_like(gate_up_angles, dtype=torch.bool)).to(
            device=target_device,
            dtype=torch.bool,
        )
        down_mask = state_dict.get("down_mask", torch.zeros_like(down_angles, dtype=torch.bool)).to(
            device=target_device,
            dtype=torch.bool,
        )
        gate_up_channel_scales = state_dict["gate_up_channel_scales"].to(device=target_device, dtype=target_dtype)
        down_channel_scales = state_dict["down_channel_scales"].to(device=target_device, dtype=target_dtype)
        group_size = state_dict["group_size"].to(device=target_device)
        n_bits = state_dict["n_bits"].to(device=target_device)
        num_rotations = state_dict["num_rotations"].to(device=target_device)

        module = cls(
            template,
            [gate_up_pairs, gate_up_angles, gate_up_mask],
            [down_pairs, down_angles, down_mask],
            gate_up_channel_scales,
            down_channel_scales,
            group_size=group_size,
            n_bits=n_bits,
            num_rotations=num_rotations,
        )

        if "gate_up_quantizer.scale" in state_dict or "down_quantizer.scale" in state_dict:
            module.gate_up_weight.data = module.gate_up_weight.data.to(device=target_device, dtype=target_dtype)
            module.down_weight.data = module.down_weight.data.to(device=target_device, dtype=target_dtype)
            module.gate_up_pairs_grouped.data = module.gate_up_pairs_grouped.data.to(
                device=target_device, dtype=torch.short
            )
            module.down_pairs_grouped.data = module.down_pairs_grouped.data.to(device=target_device, dtype=torch.short)
            module.gate_up_angles_grouped.data = module.gate_up_angles_grouped.data.to(
                device=target_device,
                dtype=target_dtype,
            )
            module.down_angles_grouped.data = module.down_angles_grouped.data.to(
                device=target_device, dtype=target_dtype
            )
            module.gate_up_channel_scales.data = module.gate_up_channel_scales.data.to(
                device=target_device,
                dtype=target_dtype,
            )
            module.down_channel_scales.data = module.down_channel_scales.data.to(
                device=target_device, dtype=target_dtype
            )
            module.group_size = module.group_size.to(device=target_device)
            module.n_bits = module.n_bits.to(device=target_device)
            module.num_rotations = module.num_rotations.to(device=target_device)
            module.set_optim_enabled(quantizer=True)

        module.load_state_dict(state_dict)
        return module


@torch.no_grad()
def reset_moe_angles_by_mask(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, PseudoQuantizedQwen3_5MoeExperts):
            submodule.reset_angles_by_mask()
