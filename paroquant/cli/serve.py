from __future__ import annotations

from paroquant.inference.base import detect_backend


def _serve_vllm():
    import asyncio
    import sys

    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        make_arg_parser,
        run_server,
    )

    import paroquant.inference.backends.vllm.plugin  # noqa: F401

    args = make_arg_parser(FlexibleArgumentParser()).parse_args(sys.argv[1:])
    asyncio.run(run_server(args))


def _serve_mlx():
    import mlx_lm.server
    from mlx_lm.utils import load_tokenizer

    from paroquant.inference.backends.mlx.load import load as paro_load

    def _patched_load(path_or_hf_repo, tokenizer_config=None, adapter_path=None, **kwargs):
        model, _, _ = paro_load(path_or_hf_repo, force_text=True)
        tokenizer = load_tokenizer(path_or_hf_repo, tokenizer_config_extra=tokenizer_config)
        tokenizer._tool_call_start = None
        tokenizer._tool_call_end = None
        return model, tokenizer

    mlx_lm.server.load = _patched_load
    mlx_lm.server.main()


def main():
    backend = detect_backend()
    if backend in ("vllm", "transformers"):
        _serve_vllm()
    elif backend == "mlx":
        _serve_mlx()
    else:
        raise RuntimeError(f"Serve requires vllm or mlx. Detected: {backend}")


if __name__ == "__main__":
    main()
