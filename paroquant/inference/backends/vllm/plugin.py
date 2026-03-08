from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_awq_marlin_linear,
    check_marlin_supports_layer,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata

import paroquant.kernels.cuda  # noqa: F401 — registers torch.ops.rotation.rotate

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

_SHARD_INDEX = {"q": 0, "k": 1, "v": 2}
_QUANT_TYPE = {4: scalar_types.uint4}
_MARLIN_TILE_N = 64


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

        # Only consider leaf modules (those with ".weight"), not containers
        # that have scalar FP16 params (e.g. A_log) which would false-match.
        leaf_modules = {k.rsplit(".", 1)[0] for k in metadata if k.endswith(".weight")}
        quant_modules: set[str] = {
            k.rsplit(".", 1)[0]
            for k, info in metadata.items()
            if (dt := info.get("dtype")) and _SF_DTYPES[dt] not in unquant_dtypes
        }
        # Strip to "layers.N..." suffix so vLLM's substr matching works
        # regardless of model nesting depth (safetensors may store
        # "model.language_model.layers.0.X" while vLLM uses
        # "language_model.model.layers.0.X").
        def _strip(name: str) -> str:
            name = name.removeprefix("model.")
            i = name.find("layers.")
            return name[i:] if i >= 0 else name

        self.modules_to_not_convert = [_strip(k) for k in leaf_modules - quant_modules]

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


class ParoQuantLinearMethod(AWQMarlinLinearMethod):
    """Per-projection rotation followed by AWQ-Marlin INT4 matmul."""

    def __init__(self, quant_config: ParoQuantConfig) -> None:
        super().__init__(quant_config)

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
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        n_parts = len(output_partition_sizes)
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

        layer.num_partitions = n_parts
        layer.output_partition_sizes = output_partition_sizes

    def _convert_partition(self, qw, sc, qz, k, out_n, num_groups):
        """AWQ→Marlin conversion for a single partition via the parent method."""
        # Pad output dim to Marlin tile boundary when needed.
        if out_n % _MARLIN_TILE_N != 0:
            pad = _MARLIN_TILE_N - out_n % _MARLIN_TILE_N
            pack = self.quant_config.pack_factor
            qw = torch.nn.functional.pad(qw, (0, pad // pack))
            sc = torch.nn.functional.pad(sc, (0, pad))
            qz = torch.nn.functional.pad(qz, (0, pad // pack))
            out_n += pad

        proxy = torch.nn.Module()
        proxy.register_parameter("qweight", Parameter(qw.contiguous(), requires_grad=False))
        proxy.register_parameter("scales", Parameter(sc.contiguous(), requires_grad=False))
        proxy.register_parameter("qzeros", Parameter(qz.contiguous(), requires_grad=False))
        proxy.input_size_per_partition = k
        proxy.output_size_per_partition = out_n
        proxy.num_groups = num_groups
        super().process_weights_after_loading(proxy)
        return proxy

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        n = layer.num_partitions

        if n == 1:
            super().process_weights_after_loading(layer)
        else:
            pack = self.quant_config.pack_factor
            sizes = layer.output_partition_sizes
            k = layer.input_size_per_partition

            qw = layer.qweight.data.split([s // pack for s in sizes], dim=1)
            sc = layer.scales.data.split(sizes, dim=1)
            qz = layer.qzeros.data.split([s // pack for s in sizes], dim=1)

            proxies = [self._convert_partition(qw[i], sc[i], qz[i], k, sizes[i], layer.num_groups) for i in range(n)]

            del layer.qweight, layer.scales, layer.qzeros
            layer.marlin_qweight = [p.qweight for p in proxies]
            layer.marlin_scales = [p.scales for p in proxies]
            layer.marlin_qzeros = [p.qzeros for p in proxies]
            layer.workspace = proxies[0].workspace
            layer.g_idx = proxies[0].g_idx
            layer.g_idx_sort_indices = proxies[0].g_idx_sort_indices
            layer.padded_partition_sizes = [p.output_size_per_partition for p in proxies]

        layer.rot_theta = layer.theta.data
        layer.rot_pairs = layer.pairs.data
        layer.rot_scales = layer.channel_scales.data
        del layer.theta, layer.pairs, layer.channel_scales

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        n = layer.num_partitions

        if n == 1:
            x = torch.ops.rotation.rotate(x, layer.rot_pairs[0], layer.rot_theta[0], layer.rot_scales[0])
            return super().apply(layer, x, bias)

        outputs = []
        for i in range(n):
            x_rot = torch.ops.rotation.rotate(x, layer.rot_pairs[i], layer.rot_theta[i], layer.rot_scales[i])
            out = apply_awq_marlin_linear(
                input=x_rot,
                weight=layer.marlin_qweight[i],
                weight_scale=layer.marlin_scales[i],
                weight_zp=layer.marlin_qzeros[i],
                g_idx=layer.g_idx,
                g_idx_sort_indices=layer.g_idx_sort_indices,
                workspace=layer.workspace,
                quant_type=self.quant_config.quant_type,
                output_size_per_partition=layer.padded_partition_sizes[i],
                input_size_per_partition=layer.input_size_per_partition,
                bias=None,
            )
            if layer.padded_partition_sizes[i] != layer.output_partition_sizes[i]:
                out = out[..., :layer.output_partition_sizes[i]]
            outputs.append(out)

        result = torch.cat(outputs, dim=-1)
        if bias is not None:
            result = result + bias
        return result
