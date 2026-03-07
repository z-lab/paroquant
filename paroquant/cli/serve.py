"""Thin wrapper around ``vllm serve`` that auto-registers the ParoQuant quantization plugin."""

from __future__ import annotations

import sys

import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers quantization config
from vllm.entrypoints.openai.run_server import run_server


def main():
    run_server(sys.argv[1:])


if __name__ == "__main__":
    main()
