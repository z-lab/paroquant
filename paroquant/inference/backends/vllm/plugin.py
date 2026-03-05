"""vLLM quantization plugin for ParoQuant.

Registers ``paroquant`` as a quantization method so any vLLM-supported
architecture can be served with rotation + INT4-AWQ-Marlin — no fork needed.

Checkpoint format (AutoAWQ int32, flat naming):
    qweight        (in, out // 8)   int32
    qzeros         (groups, out // 8) int32
    scales         (groups, out)     fp16
    theta          (krot, in // 2)   fp16
    pairs          (krot, in)        int16
    channel_scales (1, in)           fp16
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_scales,
    verify_marlin_supports_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import GroupQuantScaleParameter, ModelWeightParameter, PackedvLLMParameter
from vllm.scalar_type import scalar_types

_QKV_SHARD_MAP = {"q": 0, "k": 1, "v": 2}
_kernels_loaded = False


def _rotation_weight_loader(
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | tuple | None = None,
) -> None:
    """Map shard_id to partition index(es) for per-projection rotation params."""
    if loaded_shard_id is None:
        if param.data.dim() > loaded_weight.dim():
            param.data[0].copy_(loaded_weight)
        else:
            param.data.copy_(loaded_weight)
        return

    if isinstance(loaded_shard_id, tuple):
        for idx in loaded_shard_id:
            param.data[idx].copy_(loaded_weight)
    else:
        param.data[_QKV_SHARD_MAP.get(loaded_shard_id, loaded_shard_id)].copy_(loaded_weight)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@register_quantization_config("paroquant")
class ParoQuantConfig(QuantizationConfig):

    def __init__(self, bits: int, group_size: int, krot: int) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.krot = krot
        self.quant_type = scalar_types.uint4

    def __repr__(self) -> str:
        return f"ParoQuantConfig(bits={self.bits}, gs={self.group_size}, krot={self.krot})"

    @classmethod
    def get_name(cls) -> str:
        return "paroquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ParoQuantConfig:
        return cls(
            bits=cls.get_from_keys_or(config, ["bits"], 4),
            group_size=cls.get_from_keys_or(config, ["group_size"], 128),
            krot=cls.get_from_keys_or(config, ["krot"], 8),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(prefix, ["visual"], self.packed_modules_mapping, skip_with_substr=True):
            return UnquantizedLinearMethod()
        return ParoQuantLinearMethod(self)


# ---------------------------------------------------------------------------
# Linear method
# ---------------------------------------------------------------------------

class ParoQuantLinearMethod(LinearMethodBase):
    """Per-projection rotation + AWQ-Marlin INT4 matmul."""

    def __init__(self, quant_config: ParoQuantConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        out_total = sum(output_partition_sizes)
        bits = self.quant_config.bits
        group_size = self.quant_config.group_size
        pack = 32 // bits
        wl = extra_weight_attrs.get("weight_loader")
        n_parts = len(output_partition_sizes)

        # Fall back to fp16 for layers too small for Marlin
        try:
            verify_marlin_supports_shape(
                output_size_per_partition=out_total,
                input_size_per_partition=input_size_per_partition,
                input_size=input_size,
                group_size=group_size,
            )
            ok = input_size_per_partition % group_size == 0
        except ValueError:
            ok = False

        if not ok:
            layer._paro_unquantized = True
            layer.register_parameter("weight", ModelWeightParameter(
                data=torch.empty(out_total, input_size_per_partition, dtype=params_dtype),
                input_dim=1, output_dim=0, weight_loader=wl,
            ))
            return

        layer._paro_unquantized = False
        n_groups = input_size_per_partition // group_size

        layer.register_parameter("qweight", PackedvLLMParameter(
            data=torch.empty(input_size_per_partition, out_total // pack, dtype=torch.int32),
            input_dim=0, output_dim=1, packed_dim=1, packed_factor=pack, weight_loader=wl,
        ))
        layer.register_parameter("qzeros", PackedvLLMParameter(
            data=torch.empty(n_groups, out_total // pack, dtype=torch.int32),
            input_dim=0, output_dim=1, packed_dim=1, packed_factor=pack, weight_loader=wl,
        ))
        layer.register_parameter("scales", GroupQuantScaleParameter(
            data=torch.empty(n_groups, out_total, dtype=params_dtype),
            input_dim=0, output_dim=1, weight_loader=wl,
        ))

        dim = input_size_per_partition
        krot = self.quant_config.krot
        for name, shape, dtype in [
            ("theta", (n_parts, krot, dim // 2), torch.float16),
            ("pairs", (n_parts, krot, dim), torch.int16),
            ("channel_scales", (n_parts, 1, dim), torch.float16),
        ]:
            init_fn = torch.ones if name == "channel_scales" else torch.zeros
            p = Parameter(init_fn(shape, dtype=dtype), requires_grad=False)
            p.weight_loader = _rotation_weight_loader
            layer.register_parameter(name, p)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = out_total
        layer.num_groups = n_groups
        layer.num_partitions = n_parts
        layer.output_partition_sizes = output_partition_sizes

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer._paro_unquantized:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)
            return

        device = layer.qweight.device
        bits = self.quant_config.bits
        gs = self.quant_config.group_size
        pack = 32 // bits
        n = layer.num_partitions
        sizes = layer.output_partition_sizes
        k = layer.input_size_per_partition

        if n > 1:
            qw = layer.qweight.data.split([s // pack for s in sizes], dim=1)
            sc = layer.scales.data.split(sizes, dim=1)
            qz = layer.qzeros.data.split([s // pack for s in sizes], dim=1)
        else:
            qw, sc, qz = [layer.qweight.data], [layer.scales.data], [layer.qzeros.data]

        marlin_qweight, marlin_scales, marlin_zp = [], [], []
        for i in range(n):
            out_n = sizes[i]
            marlin_qweight.append(ops.awq_marlin_repack(qw[i].contiguous(), size_k=k, size_n=out_n, num_bits=bits))
            marlin_scales.append(marlin_permute_scales(sc[i].contiguous(), size_k=k, size_n=out_n, group_size=gs))
            marlin_zp.append(awq_to_marlin_zero_points(qz[i].contiguous(), size_k=layer.num_groups, size_n=out_n, num_bits=bits))

        layer.marlin_qweight = marlin_qweight
        layer.marlin_scales = marlin_scales
        layer.marlin_zp = marlin_zp
        layer.workspace = marlin_make_workspace_new(device)
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        layer.rotation_theta = layer.theta.data.to(device)
        layer.rotation_pairs = layer.pairs.data.to(device)
        layer.rotation_channel_scales = layer.channel_scales.data.to(device)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer._paro_unquantized:
            return F.linear(x, layer.weight, bias)

        global _kernels_loaded
        if not _kernels_loaded:
            import paroquant.kernels.cuda  # noqa: F401
            _kernels_loaded = True

        outputs = []
        for i in range(layer.num_partitions):
            x_rot = torch.ops.rotation.rotate(
                x, layer.rotation_pairs[i], layer.rotation_theta[i], layer.rotation_channel_scales[i],
            )
            outputs.append(apply_awq_marlin_linear(
                input=x_rot, weight=layer.marlin_qweight[i], weight_scale=layer.marlin_scales[i], weight_zp=layer.marlin_zp[i],
                g_idx=layer.g_idx, g_idx_sort_indices=layer.g_idx_sort_indices, workspace=layer.workspace,
                quant_type=self.quant_config.quant_type,
                output_size_per_partition=layer.output_partition_sizes[i],
                input_size_per_partition=layer.input_size_per_partition,
                bias=bias if layer.num_partitions == 1 else None,
            ))

        result = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None and layer.num_partitions > 1:
            result = result + bias
        return result
