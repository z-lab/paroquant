from __future__ import annotations

from pathlib import Path

from paroquant.export.targets import build_lmstudio, build_ollama, write_manifest


def test_build_ollama_and_lmstudio_and_manifest(tmp_path: Path) -> None:
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    q4 = gguf_dir / "model-Q4_K_M.gguf"
    q8 = gguf_dir / "model-Q8_0.gguf"
    q4.write_bytes(b"q4")
    q8.write_bytes(b"q8")

    gguf_outputs = {"Q4_K_M": q4, "Q8_0": q8, "f16": gguf_dir / "model-f16.gguf"}
    gguf_outputs["f16"].write_bytes(b"f16")

    ollama_file = build_ollama(tmp_path / "ollama", gguf_outputs, default_quant="Q4_K_M")
    assert ollama_file.exists()
    assert "FROM ../gguf/model-Q4_K_M.gguf" in ollama_file.read_text()

    model_yaml = build_lmstudio(
        lmstudio_dir=tmp_path / "lmstudio",
        model_id="z-lab/Qwen3.5-9B-PARO",
        mlx_dir=tmp_path / "mlx",
        gguf_outputs=gguf_outputs,
    )
    assert model_yaml.exists()
    text = model_yaml.read_text()
    assert "type: gguf" in text
    assert "model-Q4_K_M.gguf" in text

    payload = {
        "status": "completed",
        "artifacts": {
            "ollama": str(ollama_file),
            "gguf": {k: str(v) for k, v in gguf_outputs.items()},
        },
    }
    manifest = write_manifest(tmp_path, payload)
    assert manifest.exists()
    assert '"hashes"' in manifest.read_text()
