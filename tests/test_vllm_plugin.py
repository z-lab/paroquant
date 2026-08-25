from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch.nn import Parameter

from paroquant.inference.backends.vllm import plugin
from paroquant.inference.backends.vllm.plugin import (
    ParoQuantConfig,
    _rotation_weight_loader,
)


def test_general_plugin_registration_keeps_cuda_extension_lazy() -> None:
    code = """
import importlib.metadata
import os
import sys

os.environ["VLLM_PLUGINS"] = "paroquant"
entry_points = [
    ep
    for ep in importlib.metadata.entry_points(group="vllm.general_plugins")
    if ep.name == "paroquant"
]
assert len(entry_points) == 1
assert "paroquant.inference.backends.vllm.generator" not in sys.modules
assert "paroquant.inference.backends.vllm.plugin" not in sys.modules
assert "paroquant.kernels.cuda" not in sys.modules

import vllm.plugins

vllm.plugins.plugins_loaded = False
vllm.plugins.load_general_plugins()
assert "paroquant.inference.backends.vllm.plugin" in sys.modules
assert "paroquant.kernels.cuda" not in sys.modules

from vllm.model_executor.layers.quantization import get_quantization_config
from paroquant.inference.backends.vllm.plugin import ParoQuantConfig

assert get_quantization_config("paroquant") is ParoQuantConfig
vllm.plugins.load_general_plugins()
assert "paroquant.kernels.cuda" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bits": 8}, "Unsupported bits"),
        ({"group_size": 32}, "Unsupported group_size"),
        ({"krot": 2}, "Unsupported krot"),
    ],
)
def test_config_rejects_values_not_compiled_by_rotation_kernel(
    kwargs: dict[str, int], match: str
) -> None:
    config = {"bits": 4, "group_size": 128, "krot": 8, "zero_point": True}
    config.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ParoQuantConfig(**config)


def test_config_accepts_current_vllm_update_signature_and_scans_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def metadata(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "model.layers.0.float_proj.weight": {"dtype": "F16"},
            "model.layers.0.float_proj.index_buffer": {"dtype": "I32"},
            "model.layers.0.quant_proj.qweight": {"dtype": "I32"},
            "model.layers.0.quant_proj.pairs": {"dtype": "I16"},
            "model.layers.0.container.A_log": {"dtype": "F16"},
        }

    monkeypatch.setattr(plugin, "get_safetensors_params_metadata", metadata)
    config = ParoQuantConfig.from_config({"bits": 4, "group_size": 128, "krot": 8})

    config.maybe_update_config("model", hf_config=object(), revision="revision")
    assert config.modules_to_not_convert == ["model.layers.0.float_proj"]
    assert calls == [(("model",), {"revision": "revision"})]

    config.maybe_update_config("model", hf_config=object(), revision="revision")
    assert len(calls) == 1


def test_explicit_empty_skip_list_is_not_rescanned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_scan(*args, **kwargs):
        raise AssertionError("explicit modules_to_not_convert must not be rescanned")

    monkeypatch.setattr(plugin, "get_safetensors_params_metadata", unexpected_scan)
    config = ParoQuantConfig(
        bits=4,
        group_size=128,
        krot=8,
        zero_point=True,
        modules_to_not_convert=[],
    )

    config.maybe_update_config("model", hf_config=object())
    assert config.modules_to_not_convert == []


def test_config_maps_explicit_skip_list() -> None:
    class Mapper:
        def apply_list(self, values):
            return [f"mapped.{value}" for value in values]

    config = ParoQuantConfig(
        bits=4,
        group_size=128,
        krot=8,
        zero_point=True,
        modules_to_not_convert=["layers.0.float_proj"],
    )

    config.apply_vllm_mapper(Mapper())

    assert config.modules_to_not_convert == ["mapped.layers.0.float_proj"]


def test_paro_method_uses_current_weight_loader_and_marlin_input_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.linear import (
        WEIGHT_LOADER_V2_SUPPORTED,
        LinearBase,
    )

    class FakeMethod:
        def __init__(self, quant_config):
            self.quant_config = quant_config
            self.input_dtype = None

    monkeypatch.setattr(plugin, "ParoQuantLinearMethod", FakeMethod)
    monkeypatch.setattr(
        plugin, "check_marlin_supports_layer", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(plugin, "get_marlin_input_dtype", lambda prefix: torch.int8)

    layer = LinearBase(input_size=128, output_size=64, disable_tp=True)
    config = ParoQuantConfig(
        bits=4,
        group_size=128,
        krot=8,
        zero_point=True,
        modules_to_not_convert=[],
    )
    method = config.get_quant_method(layer, "layers.0.proj")

    assert "ParoQuantLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED
    assert isinstance(method, FakeMethod)
    assert method.input_dtype is torch.int8


@pytest.mark.parametrize(
    ("shard_id", "indices"),
    [
        ("q", [0]),
        (1, [1]),
        ((0, 2), [0, 2]),
    ],
)
def test_rotation_loader_routes_fused_shards(
    shard_id: int | str | tuple[int, ...], indices: list[int]
) -> None:
    param = Parameter(torch.zeros(3, 2, 8), requires_grad=False)
    loaded = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    _rotation_weight_loader(param, loaded, shard_id)

    for idx in range(3):
        expected = loaded if idx in indices else torch.zeros_like(loaded)
        torch.testing.assert_close(param[idx], expected)


def test_rotation_loader_replicates_unsharded_fused_weight() -> None:
    param = Parameter(torch.zeros(4, 2, 8), requires_grad=False)
    loaded = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    _rotation_weight_loader(param, loaded)

    for target in param:
        torch.testing.assert_close(target, loaded)


def test_rotation_loader_slices_row_parallel_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2
    )
    param = Parameter(torch.zeros(1, 2, 4), requires_grad=False)
    loaded = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    _rotation_weight_loader(param, loaded)

    torch.testing.assert_close(param[0], loaded[:, 4:])


def test_rotation_loader_rejects_tp_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 4
    )
    param = Parameter(torch.zeros(1, 2, 4), requires_grad=False)
    loaded = torch.zeros(2, 8)

    with pytest.raises(ValueError, match="tensor-parallel world size"):
        _rotation_weight_loader(param, loaded)


@pytest.mark.parametrize("shard_id", ["invalid", -1, 3, (0, 4)])
def test_rotation_loader_rejects_invalid_shard_ids(shard_id) -> None:
    param = Parameter(torch.zeros(3, 2, 8), requires_grad=False)
    loaded = torch.zeros(2, 8)

    with pytest.raises(ValueError, match="invalid shard id"):
        _rotation_weight_loader(param, loaded, shard_id)
