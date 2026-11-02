"""Independent checks of rotation direction, rounding and AWQ nibble layout."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch_npu
from paroquant.kernels.ascend import linear, prepare_rotation, rotate
from paroquant.kernels.ascend.native import native_linear, prepare_native


def pack(values):
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    v = values.reshape(*values.shape[:-1], -1, 8).long()
    result = torch.zeros(v.shape[:-1], dtype=torch.int64)
    for lane, source in enumerate(order):
        result |= v[..., source] << (lane * 4)
    return result.int()


def reference_rotate(x, pairs, theta, scales):
    y = (x * scales).half()
    groups = x.shape[-1] // 128
    for r in range(pairs.shape[0]):
        out = y.clone()
        for group in range(groups):
            p = pairs[r, group * 128 : (group + 1) * 128].long() + group * 128
            first, second = p[0::2], p[1::2]
            angles = theta[r, group * 64 : (group + 1) * 64].double()
            a, b = y[:, first].double(), y[:, second].double()
            out[:, first] = (a * angles.cos() + b * angles.sin()).half()
            out[:, second] = (b * angles.cos() - a * angles.sin()).half()
        y = out
    return y


def main():
    torch.manual_seed(893)
    records = []
    for m, k, n in ((1, 128, 64), (3, 256, 128), (17, 256, 128), (64, 512, 256)):
        pairs = torch.stack([torch.cat([torch.randperm(128) for _ in range(k // 128)]) for _ in range(8)]).short()
        angles = (torch.randn(8, k // 2) * 0.2).half()
        channel_scales = (torch.rand(1, k) + 0.5).half()
        x = torch.randn(m, k).half()
        prepared = prepare_rotation(pairs.npu(), angles.npu())
        rotated = rotate(x.npu(), channel_scales.npu(), prepared).cpu()
        expected_rotation = reference_rotate(x, pairs, angles, channel_scales)
        torch.testing.assert_close(rotated, expected_rotation, rtol=0.004, atol=0.004)
        integers = torch.randint(0, 16, (k, n))
        zeros = torch.randint(0, 16, (k // 128, n))
        scales = (0.03 + torch.rand(k // 128, n) * 0.08).half()
        weight = (
            (integers.float() - zeros.repeat_interleave(128, 0).float()) * scales.repeat_interleave(128, 0).float()
        ).half()
        got = linear(x.npu(), pack(integers).npu(), pack(zeros).npu(), scales.npu()).cpu()
        native = native_linear(x.npu(), prepare_native(pack(integers).npu(), pack(zeros).npu()), scales.npu()).cpu()
        expected = (x.double() @ weight.double()).half()
        torch.testing.assert_close(got, expected, rtol=0.002, atol=0.004)
        native_relative_l2 = ((native.double() - expected.double()).norm() / expected.double().norm()).item()
        if native_relative_l2 > 0.002:
            raise AssertionError(f"Native W4A16 relative L2 error: {native_relative_l2}")
        records.append(
            {
                "shape": [m, k, n],
                "rotation_max_error": (rotated - expected_rotation).abs().max().item(),
                "linear_max_error": (got - expected).abs().max().item(),
                "native_max_error": (native - expected).abs().max().item(),
                "native_relative_l2": native_relative_l2,
            }
        )
        print(json.dumps(records[-1]), flush=True)
    print(json.dumps({"status": "pass", "cases": records}, indent=2))


if __name__ == "__main__":
    main()
