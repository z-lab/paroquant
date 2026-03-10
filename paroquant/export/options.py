from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGETS = ("hf", "mlx", "gguf", "ollama", "lmstudio")
DEFAULT_GGUF_QUANTS = ("Q4_K_M", "Q8_0")
VALID_TARGETS = frozenset(DEFAULT_TARGETS)


@dataclass(frozen=True)
class ExportOptions:
    model: str
    output_dir: Path
    targets: tuple[str, ...] = DEFAULT_TARGETS
    text_only: bool = True
    gguf_quants: tuple[str, ...] = DEFAULT_GGUF_QUANTS
    llama_cpp_dir: Path | None = None
    hf_publish_layout: bool = True
    mlx_q_bits: int = 4
    mlx_q_group_size: int = 128
