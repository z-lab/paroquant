from __future__ import annotations

import glob
import json
import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers.quantizers.auto import (
    register_quantization_config,
    register_quantizer,
)
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

import paroquant.kernels.cuda  # noqa: F401 — registers torch.ops.rotation.rotate

from .modules import RotateQuantizedLinear

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def _find_quantized_modules(model_path: str) -> set[str]:
    """Scan checkpoint safetensors to find modules that have ``.qweight`` keys."""
    local_dir = model_path if os.path.isdir(model_path) else snapshot_download(model_path)

    index_file = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            keys = json.load(f).get("weight_map", {}).keys()
    else:
        keys = []
        for sf in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
            with safe_open(sf, framework="pt") as st:
                keys.extend(st.keys())

    return {k.rsplit(".", 1)[0] for k in keys if k.endswith(".qweight")}


@register_quantization_config("paroquant")
class ParoQuantConfig(QuantizationConfigMixin):
    """Quantization config for ParoQuant checkpoints."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        krot: int = 8,
        modules_to_not_convert: list[str] | None = None,
        **kwargs,
    ):
        self.quant_method = "paroquant"
        self.bits = bits
        self.group_size = group_size
        self.krot = krot
        self.modules_to_not_convert = modules_to_not_convert
        if hasattr(self, "post_init"):
            self.post_init()


@register_quantizer("paroquant")
class ParoQuantHfQuantizer(HfQuantizer):
    """Replaces nn.Linear with RotateQuantizedLinear for quantized layers.

    Only modules with ``.qweight`` in the checkpoint are replaced — visual
    encoders and other unquantized layers are left untouched automatically.
    """

    requires_calibration = True

    def validate_environment(self, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("ParoQuant requires CUDA.")

    def update_dtype(self, dtype):
        if dtype != torch.float16:
            logger.warning("ParoQuant requires float16. Overriding dtype=%s → float16.", dtype)
            return torch.float16
        return dtype

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        qcfg = self.quantization_config
        quantized_modules = _find_quantized_modules(model.config._name_or_path)
        if qcfg.modules_to_not_convert:
            quantized_modules -= set(qcfg.modules_to_not_convert)
        logger.info("Found %d quantized modules in checkpoint.", len(quantized_modules))

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name not in quantized_modules:
                continue

            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model

            setattr(
                parent,
                attr,
                RotateQuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    group_size=qcfg.group_size,
                    bits=qcfg.bits,
                    krot=qcfg.krot,
                ),
            )

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self) -> bool:
        return True
