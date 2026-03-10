from __future__ import annotations

from pathlib import Path

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

    return {
        "model": opts.model,
        "output_dir": str(opts.output_dir),
        "targets": list(opts.targets),
        "status": "scaffold_ready",
    }
