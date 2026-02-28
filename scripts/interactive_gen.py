import argparse
import asyncio
import sys
from pathlib import Path
import io
import contextlib
import warnings
import traceback
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

sys.path.append(str(Path(__file__).parents[1]))

from inference_engine.generation import create_generator, GenerationParams


@dataclass
class ChatAppConfig:
    model: str
    backend: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    compile_decode: bool = False
    gpu_memory_utilization: float = 0.8
    enable_thinking: bool = False
    debug: bool = False


@contextlib.contextmanager
def _silence_stderr():
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        yield


@contextlib.contextmanager
def _suppress_output_unless_error(console: Console, phase: str):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            yield
    except Exception:
        captured_stdout = stdout_buffer.getvalue().strip()
        captured_stderr = stderr_buffer.getvalue().strip()
        console.print(f"[red]{phase} failed.[/red]")
        if captured_stdout:
            console.print(captured_stdout)
        if captured_stderr:
            console.print(captured_stderr)
        console.print(traceback.format_exc())
        raise


async def run_chat_app(config: ChatAppConfig):
    if not config.debug:
        warnings.filterwarnings("ignore")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if config.backend == "vllm":
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
                "assistant": "bold green",
                "hint": "dim",
            }
        )
    )

    kwargs = {"enable_thinking": config.enable_thinking}
    if config.backend == "transformers":
        kwargs["compile_decode"] = config.compile_decode
    if config.backend == "vllm":
        kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization

    console.print("[hint]Loading model...[/hint]")
    loading_ctx = (
        contextlib.nullcontext()
        if (config.debug or config.backend == "vllm")
        else _suppress_output_unless_error(console, "Model loading")
    )
    with loading_ctx:
        generator = create_generator(config.backend, config.model, **kwargs)

    if config.backend == "transformers" and config.compile_decode:
        warmup_params = GenerationParams(
            max_new_tokens=min(8, config.max_new_tokens),
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
        warmup_ctx = (
            contextlib.nullcontext()
            if config.debug
            else _suppress_output_unless_error(console, "Warmup")
        )
        with warmup_ctx:
            await generator.generate(
                [{"role": "user", "content": "Hello"}],
                warmup_params,
                on_text=None,
            )

    console.print(
        Panel.fit(
            f"[bold]ParoQuant Chat[/bold]\nBackend: [bold]{config.backend}[/bold]\nModel: [bold]{config.model}[/bold]\n\nType [bold]/quit[/bold] to exit, [bold]/clear[/bold] to reset history.",
            border_style="bright_blue",
        )
    )

    history: List[Dict[str, str]] = []

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
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            console.print("[assistant]Assistant[/assistant]: ", end="")
            generation_ctx = (
                contextlib.nullcontext()
                if (config.debug or config.backend == "vllm")
                else _silence_stderr()
            )
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

            stats = result.stats
            ttft = f"{stats.ttft_s * 1000:.2f}ms" if stats.ttft_s is not None else "n/a"
            metric_str = f"tokens={stats.token_count} | time={stats.total_time_s:.2f}s | ttft={ttft} | tps={stats.tokens_per_second:.2f}"
            console.print(f"Metrics: {metric_str}", style="hint", highlight=False)
            console.print()
    finally:
        await generator.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat for ParoQuant models"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to checkpoint or HF model id"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["transformers", "vllm"],
        help="Generation backend",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16384,
        help="Maximum number of new tokens",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=32, help="Top-k sampling")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile decode path for transformers backend",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization ratio",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to chat template when supported",
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
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        compile_decode=not args.no_compile,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_thinking=args.enable_thinking,
        debug=args.debug,
    )
    await run_chat_app(config)


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.backend == "transformers" and cli_args.max_new_tokens > 1024:
        print(
            "Transformers backend suffers from performance degradation with long generations. Consider using vLLM backend for better performance."
        )
        cli_args.max_new_tokens = 1024
        print("Max new tokens set to 1024.")
    asyncio.run(run_from_args(cli_args))
