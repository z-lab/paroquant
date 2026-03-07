from __future__ import annotations

import asyncio
import sys

from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    make_arg_parser,
    run_server,
)

import paroquant.inference.backends.vllm.plugin  # noqa: F401 — registers quantization config


def main():
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(sys.argv[1:])
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
