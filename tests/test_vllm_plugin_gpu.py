from __future__ import annotations

import contextlib
import os
import tempfile

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_current_awq_kernel_single_and_multipart_differential() -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from paroquant.inference.backends.vllm.plugin import ParoQuantConfig

    if torch.cuda.get_device_capability() < (7, 5):
        pytest.skip("ParoQuant requires CUDA capability 7.5 or newer")

    fd, store_path = tempfile.mkstemp()
    os.close(fd)
    torch.cuda.set_device(0)
    torch.manual_seed(123)

    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{store_path}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)

            for parts in (
                [64],
                [64, 72],
                [64, 72, 80],
                [64, 72, 80, 88],
            ):
                _run_multipart_case(parts, ParoQuantConfig)
            _run_layerwise_reload_case(ParoQuantConfig)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(OSError):
            os.unlink(store_path)


def _run_multipart_case(parts: list[int], config_type: type) -> None:
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        ReplicatedLinear,
    )

    input_size = 128
    config = config_type(
        bits=4,
        group_size=128,
        krot=1,
        zero_point=True,
        modules_to_not_convert=[],
    )
    with torch.device("cuda"):
        fused = MergedColumnParallelLinear(
            input_size=input_size,
            output_sizes=parts,
            bias=True,
            params_dtype=torch.float16,
            quant_config=config,
            prefix="fused",
            disable_tp=True,
        )
        singles = [
            ReplicatedLinear(
                input_size=input_size,
                output_size=part_size,
                bias=True,
                params_dtype=torch.float16,
                quant_config=config,
                prefix=f"single_{index}",
                disable_tp=True,
            )
            for index, part_size in enumerate(parts)
        ]

    with torch.no_grad():
        fused.qweight.random_(-(2**31), 2**31 - 1)
        fused.qzeros.zero_()
        fused.scales.uniform_(0.001, 0.02)
        fused.theta.zero_()
        fused.channel_scales.fill_(1)
        fused.pairs.copy_(
            torch.arange(input_size, dtype=torch.int16, device="cuda")
            .reshape(1, 1, input_size)
            .expand(len(parts), -1, -1)
        )
        fused.bias.uniform_(-0.2, 0.2)
        original_fused_bias = fused.bias.clone()

        output_offset = 0
        for index, (single, part_size) in enumerate(zip(singles, parts, strict=True)):
            packed_offset = output_offset // config.pack_factor
            packed_size = part_size // config.pack_factor
            single.qweight.copy_(
                fused.qweight[:, packed_offset : packed_offset + packed_size]
            )
            single.qzeros.copy_(
                fused.qzeros[:, packed_offset : packed_offset + packed_size]
            )
            single.scales.copy_(
                fused.scales[:, output_offset : output_offset + part_size]
            )
            single.theta.copy_(fused.theta[index : index + 1])
            single.pairs.copy_(fused.pairs[index : index + 1])
            single.channel_scales.copy_(fused.channel_scales[index : index + 1])
            single.bias.copy_(fused.bias[output_offset : output_offset + part_size])
            single.quant_method.process_weights_after_loading(single)
            output_offset += part_size

        fused.quant_method.process_weights_after_loading(fused)

    x = torch.randn(7, input_size, device="cuda", dtype=torch.float16)
    actual_without_bias = fused.quant_method.apply(fused, x, None)
    expected_without_bias = torch.cat(
        [single.quant_method.apply(single, x, None) for single in singles],
        dim=-1,
    )
    actual = fused.quant_method.apply(fused, x, fused.bias)
    expected = torch.cat(
        [single.quant_method.apply(single, x, single.bias) for single in singles],
        dim=-1,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_without_bias, expected_without_bias, rtol=0, atol=0
    )
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
    assert actual.shape == (7, sum(parts))
    assert torch.isfinite(actual).all()

    if len(parts) > 1:
        torch.testing.assert_close(fused.bias, original_fused_bias, rtol=0, atol=0)
        torch.testing.assert_close(
            actual,
            actual_without_bias + original_fused_bias,
            rtol=0,
            atol=0,
        )
        assert not hasattr(fused, "paro_partitions")
        assert len(fused.quant_method._partition_views) == len(parts)
        assert len(
            {id(kernel) for kernel in fused.quant_method._partition_kernels}
        ) == len(parts)
        registered = dict(fused.named_parameters())
        for index in range(len(parts)):
            assert f"_paro_partition_{index}_qweight" in registered
            assert f"_paro_partition_{index}_qzeros" in registered
            assert f"_paro_partition_{index}_scales" in registered
        assert "theta" in registered
        assert "pairs" in registered
        assert "channel_scales" in registered


def _run_layerwise_reload_case(config_type: type) -> None:
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        ReplicatedLinear,
    )
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    input_size = 128
    parts = [64, 72]
    config = config_type(
        bits=4,
        group_size=128,
        krot=1,
        zero_point=True,
        modules_to_not_convert=[],
    )
    with torch.device("cuda"):
        fused = MergedColumnParallelLinear(
            input_size=input_size,
            output_sizes=parts,
            bias=True,
            params_dtype=torch.float16,
            quant_config=config,
            prefix="reload_fused",
            disable_tp=True,
        )
        model = torch.nn.Module()
        model.add_module("fused", fused)
        record_metadata_for_reloading(model)

    with torch.no_grad():
        fused.qweight.random_(-(2**31), 2**31 - 1)
        fused.qzeros.zero_()
        fused.scales.uniform_(0.001, 0.02)
        fused.theta.zero_()
        fused.channel_scales.fill_(1)
        fused.pairs.copy_(
            torch.arange(input_size, dtype=torch.int16, device="cuda")
            .reshape(1, 1, input_size)
            .expand(len(parts), -1, -1)
        )
        fused.bias.uniform_(-0.2, 0.2)
        reload_weights = {
            name: param.detach().clone()
            for name, param in fused.named_parameters(recurse=False)
        }
        reload_weights["qweight"].random_(-(2**31), 2**31 - 1)
        reload_weights["scales"].mul_(0.75)
        fused.quant_method.process_weights_after_loading(fused)

    parameter_ptrs = {
        name: param.data_ptr() for name, param in fused.named_parameters(recurse=False)
    }
    buffer_ptrs = {
        name: buffer.data_ptr() for name, buffer in fused.named_buffers(recurse=False)
    }
    partition_views = list(fused.quant_method._partition_views)
    partition_value_ids = [id(view._values) for view in partition_views]
    partition_kernels = list(fused.quant_method._partition_kernels)
    kernel_config_ids = [id(kernel.config) for kernel in partition_kernels]
    workspace_ptrs = [kernel.workspace.data_ptr() for kernel in partition_kernels]

    static_x = torch.randn(5, input_size, device="cuda", dtype=torch.float16)
    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    compiled_apply = torch.compile(
        lambda value: fused.quant_method.apply(fused, value, fused.bias),
        backend=counting_backend,
        fullgraph=True,
    )
    compiled_output = compiled_apply(static_x)
    assert compile_count == 1

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(2):
            fused.quant_method.apply(fused, static_x, fused.bias)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fused.quant_method.apply(fused, static_x, fused.bias)
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph.replay()
    torch.cuda.synchronize()
    original_graph_output = graph_output.clone()

    def reload_and_check_pointers() -> None:
        initialize_layerwise_reload(model)
        restored_names = {name for name, _ in fused.named_parameters(recurse=False)}
        assert restored_names == set(reload_weights)
        for name, param in list(fused.named_parameters(recurse=False)):
            param.weight_loader(param, reload_weights[name])
        finalize_layerwise_reload(model, None)

        assert parameter_ptrs == {
            name: param.data_ptr()
            for name, param in fused.named_parameters(recurse=False)
        }
        assert buffer_ptrs == {
            name: buffer.data_ptr()
            for name, buffer in fused.named_buffers(recurse=False)
        }
        assert all(
            active is original
            for active, original in zip(
                fused.quant_method._partition_views, partition_views, strict=True
            )
        )
        assert all(
            active is original
            for active, original in zip(
                fused.quant_method._partition_kernels,
                partition_kernels,
                strict=True,
            )
        )
        assert workspace_ptrs == [
            kernel.workspace.data_ptr() for kernel in partition_kernels
        ]
        assert partition_value_ids == [
            id(view._values) for view in fused.quant_method._partition_views
        ]
        assert kernel_config_ids == [
            id(kernel.config) for kernel in fused.quant_method._partition_kernels
        ]

    reload_and_check_pointers()
    torch.testing.assert_close(
        compiled_apply(static_x),
        fused.quant_method.apply(fused, static_x, fused.bias),
        rtol=0,
        atol=0,
    )
    assert compile_count == 1
    graph.replay()
    torch.cuda.synchronize()
    first_reload_graph_output = graph_output.clone()

    reload_weights["scales"].mul_(0.8)
    reload_and_check_pointers()
    torch.testing.assert_close(
        compiled_apply(static_x),
        fused.quant_method.apply(fused, static_x, fused.bias),
        rtol=0,
        atol=0,
    )
    assert compile_count == 1

    with torch.device("cuda"):
        references = [
            ReplicatedLinear(
                input_size=input_size,
                output_size=part_size,
                bias=True,
                params_dtype=torch.float16,
                quant_config=config,
                prefix=f"reload_reference_{index}",
                disable_tp=True,
            )
            for index, part_size in enumerate(parts)
        ]

    with torch.no_grad():
        output_offset = 0
        for index, (reference, part_size) in enumerate(
            zip(references, parts, strict=True)
        ):
            packed_offset = output_offset // config.pack_factor
            packed_size = part_size // config.pack_factor
            reference.qweight.copy_(
                reload_weights["qweight"][
                    :, packed_offset : packed_offset + packed_size
                ]
            )
            reference.qzeros.copy_(
                reload_weights["qzeros"][:, packed_offset : packed_offset + packed_size]
            )
            reference.scales.copy_(
                reload_weights["scales"][:, output_offset : output_offset + part_size]
            )
            reference.theta.copy_(reload_weights["theta"][index : index + 1])
            reference.pairs.copy_(reload_weights["pairs"][index : index + 1])
            reference.channel_scales.copy_(
                reload_weights["channel_scales"][index : index + 1]
            )
            reference.bias.copy_(
                reload_weights["bias"][output_offset : output_offset + part_size]
            )
            reference.quant_method.process_weights_after_loading(reference)
            output_offset += part_size

    actual_without_bias = fused.quant_method.apply(fused, static_x, None)
    expected_without_bias = torch.cat(
        [
            reference.quant_method.apply(reference, static_x, None)
            for reference in references
        ],
        dim=-1,
    )
    eager_output = fused.quant_method.apply(fused, static_x, fused.bias)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_without_bias, expected_without_bias, rtol=0, atol=0
    )
    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
    assert not torch.equal(compiled_output, eager_output)
    assert not torch.equal(original_graph_output, graph_output)
    assert not torch.equal(first_reload_graph_output, graph_output)
