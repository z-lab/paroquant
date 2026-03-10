from __future__ import annotations

from .hf import convert_paro_to_hf
from .options import ExportOptions
from .targets import (
    build_gguf,
    build_lmstudio,
    build_mlx,
    build_ollama,
    write_manifest,
)


def run_export(opts: ExportOptions) -> dict:
    """Run ParoQuant compatibility export pipeline.

    This scaffold intentionally prepares deterministic output layout.
    Conversion and runtime builders are implemented in dedicated modules.
    """
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create target directories for predictable outputs.
    target_dirs = {
        "hf": opts.output_dir / "hf-fp16",
        "mlx": opts.output_dir / "mlx",
        "gguf": opts.output_dir / "gguf",
        "ollama": opts.output_dir / "ollama",
        "lmstudio": opts.output_dir / "lmstudio",
    }

    requires_hf = any(t in opts.targets for t in ("hf", "mlx", "gguf", "ollama", "lmstudio"))
    hf_result = None
    artifacts: dict[str, str | dict[str, str]] = {}

    if requires_hf:
        target_dirs["hf"].mkdir(parents=True, exist_ok=True)
        hf_result = convert_paro_to_hf(
            model=opts.model,
            output_dir=target_dirs["hf"],
            text_only=opts.text_only,
        )
        artifacts["hf"] = str(hf_result.output_dir)

    mlx_output = None
    if "mlx" in opts.targets:
        mlx_output = build_mlx(
            hf_dir=target_dirs["hf"],
            mlx_dir=target_dirs["mlx"],
            q_bits=opts.mlx_q_bits,
            q_group_size=opts.mlx_q_group_size,
        )
        artifacts["mlx"] = str(mlx_output)

    gguf_outputs = None
    requires_gguf = any(t in opts.targets for t in ("gguf", "ollama", "lmstudio"))
    if requires_gguf:
        if opts.llama_cpp_dir is None:
            raise ValueError("--llama-cpp-dir is required for gguf, ollama, or lmstudio targets.")
        gguf_outputs = build_gguf(
            hf_dir=target_dirs["hf"],
            gguf_dir=target_dirs["gguf"],
            llama_cpp_dir=opts.llama_cpp_dir,
            quants=opts.gguf_quants,
        )
        artifacts["gguf"] = {k: str(v) for k, v in gguf_outputs.items()}

    if "ollama" in opts.targets:
        if gguf_outputs is None:
            raise RuntimeError("Ollama target requested but GGUF outputs are missing.")
        default_quant = opts.gguf_quants[0]
        modelfile = build_ollama(target_dirs["ollama"], gguf_outputs, default_quant=default_quant)
        artifacts["ollama"] = str(modelfile)

    if "lmstudio" in opts.targets:
        model_yaml = build_lmstudio(
            lmstudio_dir=target_dirs["lmstudio"],
            model_id=opts.model,
            mlx_dir=mlx_output,
            gguf_outputs=gguf_outputs,
        )
        artifacts["lmstudio"] = str(model_yaml)

    result = {
        "status": "completed",
        "model": opts.model,
        "output_dir": str(opts.output_dir),
        "targets": list(opts.targets),
        "converted_layers": hf_result.num_converted_layers if hf_result else 0,
        "artifacts": artifacts,
    }

    manifest = write_manifest(opts.output_dir, result)
    result["manifest"] = str(manifest)

    return result
