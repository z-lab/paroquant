"""Thin wrapper around ``vllm serve`` that auto-registers the ParoQuant quantization plugin."""

from __future__ import annotations

import asyncio
import sys

import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers quantization config
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    make_arg_parser,
    run_server,
)


def main():
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(sys.argv[1:])
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
