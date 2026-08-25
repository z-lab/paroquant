from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.auto_awq import (
    AutoAWQMarlinLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supports_layer,
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_SHARD_INDEX = {"q": 0, "k": 1, "v": 2}
_QUANT_TYPE = {4: scalar_types.uint4}
_ROTATION_GROUP_SIZES = {64, 128}
_ROTATION_COUNTS = {1, 8}


class _PartitionLayerView:
    """Expose one partition's root-owned kernel state under canonical names."""

    def __init__(
        self,
        layer: torch.nn.Module,
        tensor_names: dict[str, str],
        values: dict[str, Any],
    ) -> None:
        self._layer = layer
        self._tensor_names = tensor_names
        self._values = values

    def __getattr__(self, name: str) -> Any:
        if name in self._tensor_names:
            return getattr(self._layer, self._tensor_names[name])
        if name in self._values:
            return self._values[name]
        raise AttributeError(name)

    def refresh_values(self, other: _PartitionLayerView) -> None:
        if self._tensor_names != other._tensor_names:
            raise RuntimeError("ParoQuant partition state changed during reload.")
        if self._values.keys() != other._values.keys():
            raise RuntimeError("ParoQuant partition metadata changed during reload.")
        for name, new_value in other._values.items():
            old_value = self._values[name]
            if type(old_value) is not type(new_value) or old_value != new_value:
                raise RuntimeError(
                    "ParoQuant partition metadata changed during reload: "
                    f"attribute {name!r}."
                )


def _maybe_shard_input(
    target: torch.Tensor, loaded_weight: torch.Tensor
) -> torch.Tensor:
    """Slice loaded_weight along its last (input) dim if param is sharded for TP.

    Rotation params live along the linear layer's input dim. For row-parallel
    layers the param is allocated with input_size_per_partition = full // tp,
    while the on-disk weight is full size — slice it by tp_rank.
    """
    if (
        target.ndim != loaded_weight.ndim
        or target.shape[:-1] != loaded_weight.shape[:-1]
    ):
        raise ValueError(
            "ParoQuant rotation loader: incompatible shapes "
            f"target={tuple(target.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    if target.shape[-1] == loaded_weight.shape[-1]:
        return loaded_weight
    if loaded_weight.shape[-1] % target.shape[-1] != 0:
        raise ValueError(
            f"ParoQuant rotation loader: incompatible shapes "
            f"target={tuple(target.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    shard = target.shape[-1]
    num_shards = loaded_weight.shape[-1] // shard
    if num_shards != tp_world_size:
        raise ValueError(
            "ParoQuant rotation loader: loaded tensor does not match the "
            f"tensor-parallel world size (tensor shards={num_shards}, "
            f"world size={tp_world_size})"
        )
    if tp_rank >= num_shards:
        raise ValueError(
            "ParoQuant rotation loader: tensor-parallel rank is outside the "
            f"loaded tensor shards (rank={tp_rank}, shards={num_shards})"
        )
    return loaded_weight.narrow(-1, tp_rank * shard, shard)


def _rotation_weight_loader(
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | tuple | None = None,
) -> None:
    """Load per-projection rotation params into the partitioned param tensor.

    vLLM calls this with different shard_id types depending on the merge:
      None         → single or pre-fused projection, copy to all targets
      "q"/"k"/"v"  → QKV merge, map to partition index 0/1/2
      int          → gate/up merge, use as partition index
      tuple        → fused projections (e.g. Qwen3.5), copy to each index
    """
    if loaded_shard_id is None:
        if param.data.dim() == loaded_weight.dim() + 1:
            for target in param.data:
                target.copy_(_maybe_shard_input(target, loaded_weight))
        else:
            param.data.copy_(_maybe_shard_input(param.data, loaded_weight))
        return

    indices = (
        loaded_shard_id
        if isinstance(loaded_shard_id, tuple)
        else (_SHARD_INDEX.get(loaded_shard_id, loaded_shard_id),)
    )
    for idx in indices:
        if not isinstance(idx, int) or not 0 <= idx < param.data.shape[0]:
            raise ValueError(
                f"ParoQuant rotation loader: invalid shard id {loaded_shard_id!r} "
                f"for {param.data.shape[0]} partitions"
            )
        target = param.data[idx]
        target.copy_(_maybe_shard_input(target, loaded_weight))


@register_quantization_config("paroquant")
class ParoQuantConfig(QuantizationConfig):
    def __init__(
        self,
        bits: int,
        group_size: int,
        krot: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        if bits not in _QUANT_TYPE:
            raise ValueError(f"Unsupported bits={bits}. Supported: {list(_QUANT_TYPE)}")
        if group_size not in _ROTATION_GROUP_SIZES:
            raise ValueError(
                f"Unsupported group_size={group_size}. "
                f"Supported: {sorted(_ROTATION_GROUP_SIZES)}"
            )
        if krot not in _ROTATION_COUNTS:
            raise ValueError(
                f"Unsupported krot={krot}. Supported: {sorted(_ROTATION_COUNTS)}"
            )
        self.bits = bits
        self.weight_bits = bits
        self.group_size = group_size
        self.krot = krot
        self.pack_factor = 32 // bits
        self.quant_type = _QUANT_TYPE[bits]
        self.modules_to_not_convert = modules_to_not_convert
        self.zero_point = zero_point

    def __repr__(self) -> str:
        return f"ParoQuantConfig(bits={self.bits}, group_size={self.group_size}, krot={self.krot}, zero_point={self.zero_point})"

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
            zero_point=cls.get_from_keys_or(config, ["zero_point"], True),
            modules_to_not_convert=cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            ),
        )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        """Auto-detect unquantized layers from safetensors metadata."""
        if self.modules_to_not_convert is not None:
            return

        metadata = get_safetensors_params_metadata(model_name, revision=revision)

        # Only consider leaf modules (those with ".weight"), not containers
        # that have scalar FP16 params (e.g. A_log) which would false-match.
        leaf_modules = {k.rsplit(".", 1)[0] for k in metadata if k.endswith(".weight")}
        quant_modules = {
            k.removesuffix(".qweight") for k in metadata if k.endswith(".qweight")
        }

        # Preserve checkpoint names here. vLLM applies the model's unstacked
        # WeightsMapper after metadata discovery, including multimodal prefixes.
        self.modules_to_not_convert = sorted(leaf_modules - quant_modules)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix,
            self.modules_to_not_convert or [],
            self.packed_modules_mapping,
            skip_with_substr=True,
        ):
            return UnquantizedLinearMethod()
        if not check_marlin_supports_layer(
            layer, self.group_size, allow_tile_padding=True
        ):
            raise ValueError(
                f"Quantized ParoQuant layer {prefix!r} cannot be sharded without "
                "splitting a quantization group across tensor-parallel ranks."
            )
        quant_method = ParoQuantLinearMethod(self)
        quant_method.input_dtype = get_marlin_input_dtype(prefix)
        return quant_method


@register_weight_loader_v2_supported_method
class ParoQuantLinearMethod(AutoAWQMarlinLinearMethod):
    """Per-projection rotation followed by AWQ-Marlin INT4 matmul."""

    def __init__(self, quant_config: ParoQuantConfig) -> None:
        import paroquant.kernels.cuda  # noqa: F401

        super().__init__(quant_config)
        self._partition_kernels: list[Any] = []
        self._partition_views: list[_PartitionLayerView] = []

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
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "ParoQuant rotation requires input_size_per_partition to be "
                f"divisible by group_size, got {input_size_per_partition} and "
                f"{self.quant_config.group_size}."
            )
        if input_size_per_partition % 2 != 0:
            raise ValueError(
                "ParoQuant rotation requires an even input dimension, got "
                f"{input_size_per_partition}."
            )
        if any(size % self.quant_config.pack_factor for size in output_partition_sizes):
            raise ValueError(
                "ParoQuant output partitions must be divisible by the packing "
                f"factor {self.quant_config.pack_factor}: {output_partition_sizes}."
            )

        full_partition_sizes = getattr(layer, "output_sizes", None)
        if full_partition_sizes is None:
            if n_parts != 1:
                raise ValueError(
                    "Multipart ParoQuant layers must expose their full logical "
                    "output sizes through layer.output_sizes."
                )
            full_partition_sizes = [output_size]
        if len(full_partition_sizes) != n_parts:
            raise ValueError(
                "ParoQuant logical partition metadata does not match local "
                f"partitions: full={full_partition_sizes}, local={output_partition_sizes}."
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

        layer.paro_num_partitions = n_parts
        layer.paro_output_partition_sizes = list(output_partition_sizes)
        self._input_size = input_size
        self._input_size_per_partition = input_size_per_partition
        self._full_partition_sizes = list(full_partition_sizes)
        self._params_dtype = params_dtype

    def _process_partition(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        local_output_size: int,
        full_output_size: int,
    ) -> tuple[torch.nn.Module, Any]:
        proxy = torch.nn.Module()
        method = AutoAWQMarlinLinearMethod(self.quant_config)
        method.input_dtype = self.input_dtype

        with torch.device(qweight.device):
            method.create_weights(
                proxy,
                input_size_per_partition=self._input_size_per_partition,
                output_partition_sizes=[local_output_size],
                input_size=self._input_size,
                output_size=full_output_size,
                params_dtype=self._params_dtype,
            )

        with torch.no_grad():
            proxy.qweight.copy_(qweight)
            proxy.scales.copy_(scales)
            proxy.qzeros.copy_(qzeros)

        method.process_weights_after_loading(proxy)
        return proxy, method.kernel

    @staticmethod
    def _partition_state_name(partition: int, name: str) -> str:
        return f"_paro_partition_{partition}_{name}"

    def _install_partition_state(
        self,
        layer: torch.nn.Module,
        partition: int,
        proxy: torch.nn.Module,
    ) -> _PartitionLayerView:
        tensor_names: dict[str, str] = {}
        values: dict[str, Any] = {}

        for name, param in proxy._parameters.items():
            if param is None:
                values[name] = None
                continue
            root_name = self._partition_state_name(partition, name)
            layer.register_parameter(root_name, param)
            tensor_names[name] = root_name

        for name, buffer in proxy._buffers.items():
            if buffer is None:
                values[name] = None
                continue
            root_name = self._partition_state_name(partition, name)
            layer.register_buffer(root_name, buffer)
            tensor_names[name] = root_name

        for name, value in vars(proxy).items():
            if name.startswith("_") or name in tensor_names or name in values:
                continue
            if isinstance(value, torch.Tensor):
                root_name = self._partition_state_name(partition, name)
                layer.register_buffer(root_name, value)
                tensor_names[name] = root_name
            else:
                values[name] = value

        return _PartitionLayerView(layer, tensor_names, values)

    @staticmethod
    def _refresh_kernel_state(active: Any, reloaded: Any) -> None:
        if type(active) is not type(reloaded):
            raise RuntimeError(
                "ParoQuant selected a different linear kernel during reload: "
                f"{type(active).__name__} -> {type(reloaded).__name__}."
            )

        if vars(active).keys() != vars(reloaded).keys():
            raise RuntimeError("ParoQuant kernel state changed during reload.")

        tensor_state: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name, new_value in vars(reloaded).items():
            old_value = getattr(active, name)
            if isinstance(old_value, torch.Tensor) or isinstance(
                new_value, torch.Tensor
            ):
                if not (
                    isinstance(old_value, torch.Tensor)
                    and isinstance(new_value, torch.Tensor)
                    and old_value.shape == new_value.shape
                    and old_value.dtype == new_value.dtype
                    and old_value.device == new_value.device
                    and old_value.layout == new_value.layout
                    and old_value.stride() == new_value.stride()
                ):
                    raise RuntimeError(
                        "ParoQuant kernel tensor state changed during reload: "
                        f"attribute {name!r}."
                    )
                tensor_state.append((old_value, new_value))
            elif old_value != new_value:
                raise RuntimeError(
                    "ParoQuant kernel configuration changed during reload: "
                    f"attribute {name!r}."
                )

        for old_value, new_value in tensor_state:
            old_value.copy_(new_value)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        n = layer.paro_num_partitions

        if n == 1:
            super().process_weights_after_loading(layer)
        else:
            from vllm.model_executor.offloader import NoopOffloader, get_offloader

            if not isinstance(get_offloader(), NoopOffloader):
                raise NotImplementedError(
                    "Multipart ParoQuant weights cannot be registered after vLLM "
                    "has configured model-weight offloading. Disable model-weight "
                    "offloading for this checkpoint."
                )

            pack = self.quant_config.pack_factor
            sizes = layer.paro_output_partition_sizes

            qw = layer.qweight.data.split([s // pack for s in sizes], dim=1)
            sc = layer.scales.data.split(sizes, dim=1)
            qz = layer.qzeros.data.split([s // pack for s in sizes], dim=1)

            partitions: list[torch.nn.Module] = []
            kernels: list[Any] = []
            for i in range(n):
                partition, kernel = self._process_partition(
                    qw[i],
                    sc[i],
                    qz[i],
                    sizes[i],
                    self._full_partition_sizes[i],
                )
                partitions.append(partition)
                kernels.append(kernel)

            if self._partition_kernels:
                if len(self._partition_kernels) != len(kernels):
                    raise RuntimeError(
                        "ParoQuant partition count changed during weight reload."
                    )
                for active, reloaded in zip(
                    self._partition_kernels, kernels, strict=True
                ):
                    self._refresh_kernel_state(active, reloaded)

            state_names = {
                self._partition_state_name(i, name)
                for i, partition in enumerate(partitions)
                for name, value in (
                    list(partition._parameters.items())
                    + list(partition._buffers.items())
                    + [
                        (name, value)
                        for name, value in vars(partition).items()
                        if not name.startswith("_") and isinstance(value, torch.Tensor)
                    ]
                )
                if value is not None
            }
            collisions = sorted(name for name in state_names if hasattr(layer, name))
            if collisions:
                raise RuntimeError(
                    "ParoQuant partition state already exists before processing: "
                    f"{collisions}."
                )

            del layer.qweight, layer.scales, layer.qzeros
            views = [
                self._install_partition_state(layer, i, partition)
                for i, partition in enumerate(partitions)
            ]
            if self._partition_views:
                for active, reloaded in zip(self._partition_views, views, strict=True):
                    active.refresh_values(reloaded)
            else:
                self._partition_views = views
            if not self._partition_kernels:
                self._partition_kernels = kernels

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        n = layer.paro_num_partitions

        if n == 1:
            x = torch.ops.rotation.rotate(
                x,
                layer.pairs[0],
                layer.theta[0],
                layer.channel_scales[0],
                self.quant_config.group_size,
            )
            return super().apply(layer, x, bias)

        outputs = []
        for i, (partition, kernel) in enumerate(
            zip(self._partition_views, self._partition_kernels, strict=True)
        ):
            x_rot = torch.ops.rotation.rotate(
                x,
                layer.pairs[i],
                layer.theta[i],
                layer.channel_scales[i],
                self.quant_config.group_size,
            )
            outputs.append(kernel.apply_weights(partition, x_rot, None))

        result = torch.cat(outputs, dim=-1)
        if bias is not None:
            result = result + bias
        return result
