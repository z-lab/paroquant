from __future__ import annotations

from pathlib import Path

from .hf import convert_paro_to_hf
from .options import ExportOptions


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

    for target in opts.targets:
        if target in target_dirs:
            target_dirs[target].mkdir(parents=True, exist_ok=True)

    requires_hf = any(t in opts.targets for t in ("hf", "mlx", "gguf", "ollama", "lmstudio"))
    hf_result = None

    if requires_hf:
        hf_result = convert_paro_to_hf(
            model=opts.model,
            output_dir=target_dirs["hf"],
            text_only=opts.text_only,
        )

    unimplemented = [t for t in opts.targets if t in {"mlx", "gguf", "ollama", "lmstudio"}]
    if unimplemented:
        raise NotImplementedError(
            "HF conversion completed, but target builders are not implemented yet: " + ", ".join(unimplemented)
        )

    return {
        "model": opts.model,
        "output_dir": str(opts.output_dir),
        "targets": list(opts.targets),
        "status": "hf_converted",
        "hf_output": str(hf_result.output_dir) if hf_result else None,
        "converted_layers": hf_result.num_converted_layers if hf_result else 0,
    }
