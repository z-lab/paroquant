from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import os
import warnings
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

from paroquant.inference import GenerationParams, create_generator


@dataclass
class ChatAppConfig:
    model: str
    backend: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    gpu_memory_utilization: float = 0.8
    enable_thinking: bool = False
    debug: bool = False


@contextlib.contextmanager
def _silence_stderr():
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        yield


async def run_chat_app(config: ChatAppConfig):
    if not config.debug:
        warnings.filterwarnings("ignore")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        try:
            from huggingface_hub import disable_progress_bars

            disable_progress_bars()
        except Exception:
            pass
        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
        except Exception:
            pass

    console = Console(
        theme=Theme(
            {
                "user": "bold cyan",
                "assistant": "bold blue",
                "hint": "dim",
            }
        )
    )

    from paroquant.inference.base import _detect_backend

    backend = _detect_backend() if config.backend == "auto" else config.backend

    kwargs = {"enable_thinking": config.enable_thinking}
    if backend == "vllm":
        kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization

    console.print(f"[hint]Loading model ({backend})...[/hint]")
    generator = create_generator(backend, config.model, **kwargs)

    banner = (
        f"[bold]ParoQuant Chat[/bold]\n"
        f"Backend: [bold]{backend}[/bold]\n"
        f"Model: [bold]{config.model}[/bold]\n\n"
        f"Type [bold]/quit[/bold] to exit, [bold]/clear[/bold] to reset history."
    )
    console.print(Panel.fit(banner, border_style="bright_blue"))

    history: list[dict[str, str]] = []

    try:
        while True:
            try:
                user_prompt = Prompt.ask("[user]You[/user]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[hint]Session closed.[/hint]")
                break

            if not user_prompt:
                continue
            if user_prompt.lower() in {"/quit", "quit", "/exit", "exit"}:
                break
            if user_prompt.lower() == "/clear":
                history.clear()
                console.clear()
                console.print("[hint]Conversation history cleared.[/hint]")
                continue

            history.append({"role": "user", "content": user_prompt})

            params = GenerationParams(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            console.print("[assistant]Assistant[/assistant]: ", end="")
            generation_ctx = contextlib.nullcontext() if config.debug else _silence_stderr()
            with generation_ctx:
                result = await generator.generate(
                    history,
                    params,
                    on_text=lambda text: console.print(
                        text,
                        end="",
                        highlight=False,
                        soft_wrap=True,
                    ),
                )
            console.print()

            history.append({"role": "assistant", "content": result.output_text})

            s = result.stats
            ttft = f"{s.ttft * 1000:.2f}ms" if s.ttft is not None else "n/a"
            metrics = f"tokens={s.num_tokens} | latency={s.latency:.2f}s" f" | ttft={ttft} | tps={s.tps:.2f}"
            console.print(f"Metrics: {metrics}", style="hint", highlight=False)
            console.print()
    finally:
        await generator.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive terminal chat for ParoQuant models")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint or HF model id")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers", "mlx"],
        help="Generation backend (auto: mlx on Apple Silicon, vllm on NVIDIA GPU)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=32, help="Top-k sampling")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization ratio",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_false",
        dest="enable_thinking",
        help="Pass enable_thinking=False to chat template when supported",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable suppression and show backend logs/warnings",
    )
    return parser


async def run_from_args(args: argparse.Namespace):
    config = ChatAppConfig(
        model=args.model,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_thinking=args.enable_thinking,
        debug=args.debug,
    )
    await run_chat_app(config)


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    asyncio.run(run_from_args(cli_args))
