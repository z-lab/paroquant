from __future__ import annotations

import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from paroquant.export.options import ExportOptions
from paroquant.export.pipeline import run_export

AWQ_REORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_awq(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int32).view(values.shape[0], -1, 8)[:, :, AWQ_REORDER]
    out = torch.zeros(values.shape[0], values.shape[1], dtype=torch.int32)
    for i in range(8):
        out |= (values[:, :, i] & 0xF) << (4 * i)
    return out


def _write_fixture_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "quantization_config": {
            "quant_method": "paroquant",
            "bits": 4,
            "group_size": 4,
            "krot": 1,
        },
        "vision_config": {"dummy": True},
        "image_token_id": 1,
        "video_token_id": 2,
        "vision_start_token_id": 3,
        "vision_end_token_id": 4,
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    in_features = 8
    out_features = 16
    group_size = 4
    n_groups = in_features // group_size
    prefix = "model.layers.0.mlp.up_proj"

    iweight = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    izeros = torch.zeros((n_groups, out_features), dtype=torch.int32)
    scales = torch.ones((n_groups, out_features), dtype=torch.float16)

    state = {
        f"{prefix}.qweight": _pack_awq(iweight),
        f"{prefix}.qzeros": _pack_awq(izeros),
        f"{prefix}.scales": scales,
        f"{prefix}.theta": torch.zeros((1, in_features // 2), dtype=torch.float16),
        f"{prefix}.pairs": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.int16),
        f"{prefix}.channel_scales": torch.ones((1, in_features), dtype=torch.float16),
        "model.embed_tokens.weight": torch.randn((32, in_features), dtype=torch.float16),
    }
    safetensors_torch.save_file(state, str(model_dir / "model.safetensors"))


def test_run_export_hf_only(tmp_path: Path) -> None:
    source = tmp_path / "source-model"
    _write_fixture_model(source)

    output_dir = tmp_path / "out"
    opts = ExportOptions(model=str(source), output_dir=output_dir, targets=("hf",))

    result = run_export(opts)
    assert result["status"] == "completed"
    assert result["converted_layers"] == 1

    hf_out = output_dir / "hf-fp16"
    assert (hf_out / "config.json").exists()
    assert (output_dir / "manifest.json").exists()

    cfg = json.loads((hf_out / "config.json").read_text())
    assert "quantization_config" not in cfg
    assert "vision_config" not in cfg


def test_cli_smoke_hf_only(tmp_path: Path) -> None:
    source = tmp_path / "source-model"
    _write_fixture_model(source)

    out = tmp_path / "cli-out"
    cmd = [
        "python3",
        "-m",
        "paroquant.cli.export",
        "--model",
        str(source),
        "--output-dir",
        str(out),
        "--targets",
        "hf",
    ]

    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "completed"
    assert (out / "manifest.json").exists()
