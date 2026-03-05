from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Parameter
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    check_marlin_supports_layer,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata

import paroquant.kernels.cuda  # noqa: F401 — registers torch.ops.rotation.rotate

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

_SHARD_INDEX = {"q": 0, "k": 1, "v": 2}
_QUANT_TYPE = {4: scalar_types.uint4}


def _rotation_weight_loader(
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | tuple | None = None,
) -> None:
    """Load per-projection rotation params into the partitioned param tensor.

    vLLM calls this with different shard_id types depending on the merge:
      None         → single projection, copy directly
      "q"/"k"/"v"  → QKV merge, map to partition index 0/1/2
      int          → gate/up merge, use as partition index
      tuple        → fused projections (e.g. Qwen3.5), copy to each index
    """
    if loaded_shard_id is None:
        target = param.data[0] if param.data.dim() > loaded_weight.dim() else param.data
        target.copy_(loaded_weight)
        return

    indices = (
        loaded_shard_id if isinstance(loaded_shard_id, tuple) else (_SHARD_INDEX.get(loaded_shard_id, loaded_shard_id),)
    )
    for idx in indices:
        param.data[idx].copy_(loaded_weight)


@register_quantization_config("paroquant")
class ParoQuantConfig(QuantizationConfig):
    def __init__(self, bits: int, group_size: int, krot: int, modules_to_not_convert: list[str] | None = None) -> None:
        super().__init__()
        if bits not in _QUANT_TYPE:
            raise ValueError(f"Unsupported bits={bits}. Supported: {list(_QUANT_TYPE)}")
        self.bits = bits
        self.group_size = group_size
        self.krot = krot
        self.pack_factor = 32 // bits
        self.quant_type = _QUANT_TYPE[bits]
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return f"ParoQuantConfig(bits={self.bits}, group_size={self.group_size}, krot={self.krot})"

    @classmethod
    def get_name(cls) -> "QuantizationMethods":
        return "paroquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

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
            modules_to_not_convert=config.get("modules_to_not_convert"),
        )

    def maybe_update_config(self, model_name: str, revision: str | None = None):
        """Auto-detect unquantized layers from safetensors metadata."""
        if self.modules_to_not_convert:
            return

        from safetensors.torch import _TYPES as _SF_DTYPES

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        all_layers = {k.rsplit(".", 1)[0] for k in metadata}
        quant_layers: set[str] = {
            k.rsplit(".", 1)[0]
            for k, info in metadata.items()
            if (dt := info.get("dtype")) and _SF_DTYPES[dt] not in unquant_dtypes
        }
        # Strip "model." prefix so names match vLLM's internal module prefixes
        # (safetensors keys use "model.X" but vLLM's get_quant_method receives "X").
        self.modules_to_not_convert = [k.removeprefix("model.") for k in (all_layers - quant_layers)]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(prefix, self.modules_to_not_convert, self.packed_modules_mapping, skip_with_substr=True):
            return UnquantizedLinearMethod()
        if not check_marlin_supports_layer(layer, self.group_size):
            logger.warning_once(
                "Layer '%s' is not supported by Marlin. Falling back to unquantized.",
                prefix,
            )
            return UnquantizedLinearMethod()
        return ParoQuantLinearMethod(self)


class ParoQuantLinearMethod(LinearMethodBase):
    """Per-projection rotation followed by AWQ-Marlin INT4 matmul."""

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
        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        pack = self.quant_config.pack_factor
        group_size = self.quant_config.group_size
        weight_loader = extra_weight_attrs.get("weight_loader")
        n_parts = len(output_partition_sizes)
        n_groups = input_size_per_partition // group_size

        layer.register_parameter(
            "qweight",
            PackedvLLMParameter(
                data=torch.empty(input_size_per_partition, output_size_per_partition // pack, dtype=torch.int32),
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=pack,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "qzeros",
            PackedvLLMParameter(
                data=torch.empty(n_groups, output_size_per_partition // pack, dtype=torch.int32),
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=pack,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "scales",
            GroupQuantScaleParameter(
                data=torch.empty(n_groups, output_size_per_partition, dtype=params_dtype),
                input_dim=0,
                output_dim=1,
                weight_loader=weight_loader,
            ),
        )

        krot = self.quant_config.krot
        for name, shape, dtype in [
            ("theta", (n_parts, krot, input_size_per_partition // 2), torch.float16),
            ("pairs", (n_parts, krot, input_size_per_partition), torch.int16),
            ("channel_scales", (n_parts, 1, input_size_per_partition), torch.float16),
        ]:
            init_fn = torch.ones if name == "channel_scales" else torch.zeros
            p = Parameter(init_fn(shape, dtype=dtype), requires_grad=False)
            p.weight_loader = _rotation_weight_loader
            layer.register_parameter(name, p)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = n_groups
        layer.num_partitions = n_parts
        layer.output_partition_sizes = output_partition_sizes

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device
        bits = self.quant_config.bits
        gs = self.quant_config.group_size
        pack = self.quant_config.pack_factor
        n = layer.num_partitions
        sizes = layer.output_partition_sizes
        k = layer.input_size_per_partition

        if n > 1:
            qw = layer.qweight.data.split([s // pack for s in sizes], dim=1)
            sc = layer.scales.data.split(sizes, dim=1)
            qz = layer.qzeros.data.split([s // pack for s in sizes], dim=1)
        else:
            qw, sc, qz = [layer.qweight.data], [layer.scales.data], [layer.qzeros.data]

        marlin_qw, marlin_sc, marlin_zp = [], [], []
        for i in range(n):
            out_n = sizes[i]
            marlin_qw.append(ops.awq_marlin_repack(qw[i].contiguous(), size_k=k, size_n=out_n, num_bits=bits))
            marlin_sc.append(marlin_permute_scales(sc[i].contiguous(), size_k=k, size_n=out_n, group_size=gs))
            marlin_zp.append(
                awq_to_marlin_zero_points(qz[i].contiguous(), size_k=layer.num_groups, size_n=out_n, num_bits=bits)
            )

        del layer.qweight, layer.scales, layer.qzeros
        layer.marlin_qweight = marlin_qw
        layer.marlin_scales = marlin_sc
        layer.marlin_qzeros = marlin_zp
        layer.workspace = marlin_make_workspace_new(device)
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        layer.rot_theta = layer.theta.data.to(device)
        layer.rot_pairs = layer.pairs.data.to(device)
        layer.rot_scales = layer.channel_scales.data.to(device)
        del layer.theta, layer.pairs, layer.channel_scales

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        outputs = []
        for i in range(layer.num_partitions):
            x_rot = torch.ops.rotation.rotate(x, layer.rot_pairs[i], layer.rot_theta[i], layer.rot_scales[i])
            outputs.append(
                apply_awq_marlin_linear(
                    input=x_rot,
                    weight=layer.marlin_qweight[i],
                    weight_scale=layer.marlin_scales[i],
                    weight_zp=layer.marlin_qzeros[i],
                    g_idx=layer.g_idx,
                    g_idx_sort_indices=layer.g_idx_sort_indices,
                    workspace=layer.workspace,
                    quant_type=self.quant_config.quant_type,
                    output_size_per_partition=layer.output_partition_sizes[i],
                    input_size_per_partition=layer.input_size_per_partition,
                    bias=bias if layer.num_partitions == 1 else None,
                )
            )

        result = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None and layer.num_partitions > 1:
            result = result + bias
        return result
