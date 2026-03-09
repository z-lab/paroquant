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
    import os
    import sys

    from paroquant.inference.backends.mlx.load import load as paro_load

    original_argv = list(sys.argv)
    model_arg = None
    llm_only = False
    stripped_argv = [original_argv[0]]
    i = 1
    while i < len(original_argv):
        arg = original_argv[i]
        if arg == "--model":
            if i + 1 >= len(original_argv):
                raise ValueError("--model expects a value")
            model_arg = original_argv[i + 1]
            i += 2
            continue
        if arg.startswith("--model="):
            model_arg = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--llm-only":
            llm_only = True
            i += 1
            continue
        stripped_argv.append(arg)
        i += 1

    if not model_arg:
        model_arg = os.environ.get("MODEL")
    if not model_arg:
        raise ValueError("Model path is required (use --model or MODEL environment variable).")

    model, processor, is_vlm = paro_load(model_arg, force_text=llm_only)

    if is_vlm:
        import mlx_vlm.server as mlx_server

        os.environ["MODEL"] = model_arg
        sys.argv = stripped_argv

        def _patched_load(path_or_hf_repo, *args, **kwargs):
            return model, processor

        _uvicorn_run = mlx_server.uvicorn.run

        def _run_no_reload(*args, **kwargs):
            kwargs["reload"] = False
            return _uvicorn_run(*args, **kwargs)

        mlx_server.uvicorn.run = _run_no_reload
    else:
        import mlx_lm.server as mlx_server

        tokenizer = getattr(processor, "tokenizer", processor)
        if hasattr(tokenizer, "_tool_call_start"):
            tokenizer._tool_call_start = None
        if hasattr(tokenizer, "_tool_call_end"):
            tokenizer._tool_call_end = None
        sys.argv = stripped_argv

        def _patched_load(path_or_hf_repo, tokenizer_config=None, adapter_path=None, **kwargs):
            return model, tokenizer

    mlx_server.load = _patched_load
    mlx_server.main()


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
