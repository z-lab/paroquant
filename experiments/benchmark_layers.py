"""Compare actual checkpoint linears, including rotation and layout costs."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch_npu
from safetensors import safe_open
from paroquant.inference.backends.transformers.modules import RotateQuantizedLinear
from ascend_reference import dequantize, reference_forward, rotation
from paroquant.kernels.ascend import linear, prepare_rotation, rotate
from paroquant.kernels.ascend.native import prepare_native, native_linear


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-4B-PARO"))
    parser.add_argument(
        "--modules", nargs="+", default=["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"]
    )
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    records = []
    index = args.model / "model.safetensors.index.json"
    shards = json.loads(index.read_text())["weight_map"] if index.exists() else None

    def tensor(key):
        path = args.model / (shards[key] if shards else "model.safetensors")
        with safe_open(path, framework="pt", device="cpu") as file:
            return file.get_tensor(key)

    for name in args.modules:
        tensors = {
            suffix: tensor(name + "." + suffix)
            for suffix in ["qweight", "qzeros", "scales", "pairs", "theta", "channel_scales"]
        }
        k, n = tensors["qweight"].shape[0], tensors["scales"].shape[1]
        module = RotateQuantizedLinear(k, n).npu()
        module.load_state_dict(tensors)
        module._reference_weight = None
        module._reference_rotation = None
        torch.npu.synchronize()
        before_rotation = time.perf_counter()
        prepared_rotation = prepare_rotation(module.pairs, module.theta)
        torch.npu.synchronize()
        rotation_prepare_seconds = time.perf_counter() - before_rotation
        triton_function = lambda x: linear(
            rotate(x, module.channel_scales, prepared_rotation), module.qweight, module.qzeros, module.scales
        )
        for m in args.rows:
            x = torch.randn(m, k, dtype=torch.float16, device="npu")
            torch.npu.synchronize()
            start = time.perf_counter()
            actual = triton_function(x)
            torch.npu.synchronize()
            first_seconds = time.perf_counter() - start
            expected = reference_forward(module, x)
            difference = actual.float() - expected.float()
            relative_l2 = float(difference.square().sum() / expected.float().square().sum()) ** 0.5
            if relative_l2 > 0.01:
                raise AssertionError(f"Checkpoint layer mismatch: {name} M={m}, relative L2={relative_l2}")
            for mode in ("triton", "naive", "dense", "dense_fused", "native_unfused", "native"):
                before_layout = time.perf_counter()
                if mode in ("dense", "dense_fused"):
                    module._reference_weight = dequantize(module)
                    if mode == "dense_fused":
                        module._reference_rotation = prepared_rotation
                if mode.startswith("native"):
                    prepared_weight = prepare_native(module.qweight, module.qzeros)
                torch.npu.synchronize()
                layout_seconds = time.perf_counter() - before_layout
                if mode.startswith("native"):
                    rotation_function = (
                        (lambda x: rotation(x, module))
                        if mode == "native_unfused"
                        else (lambda x: rotate(x, module.channel_scales, prepared_rotation))
                    )
                    function = lambda x: native_linear(rotation_function(x), prepared_weight, module.scales)
                else:
                    function = triton_function if mode == "triton" else lambda x: reference_forward(module, x)
                mode_output = function(x)
                mode_error = mode_output.float() - expected.float()
                mode_relative_l2 = float(mode_error.square().sum() / expected.float().square().sum()) ** 0.5
                mode_max_error = float(mode_error.abs().max())
                if mode_relative_l2 > 0.01:
                    raise AssertionError(f"{name} M={m} {mode} relative L2 error: {mode_relative_l2}")
                del mode_output, mode_error
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                before = time.perf_counter()
                for _ in range(args.repeats):
                    function(x)
                torch.npu.synchronize()
                record = {
                    "module": name,
                    "shape": [m, k, n],
                    "mode": mode,
                    "seconds": (time.perf_counter() - before) / args.repeats,
                    "peak_bytes": torch.npu.max_memory_allocated(),
                    "relative_l2": mode_relative_l2,
                    "max_error": mode_max_error,
                    "first_triton_seconds": first_seconds,
                }
                record.update(weight_layout_seconds=layout_seconds, rotation_prepare_seconds=rotation_prepare_seconds)
                records.append(record)
                print(json.dumps(record), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"model": str(args.model), "seed": args.seed, "records": records}, indent=2)
                )
                module._reference_weight = None
                module._reference_rotation = None
                if mode.startswith("native"):
                    function = None
                    del prepared_weight
        del module


if __name__ == "__main__":
    main()
