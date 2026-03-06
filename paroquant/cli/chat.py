from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import os
import re
import time
import warnings
from dataclasses import dataclass

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.theme import Theme

from paroquant.inference import GenerationParams, build_prompt, create_generator


@dataclass
class ChatAppConfig:
    model: str
    backend: str
    params: GenerationParams
    gpu_memory_utilization: float = 0.8
    enable_thinking: bool = True
    debug: bool = False


@contextlib.contextmanager
def _silence_stderr():
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        yield


_THINKING_LINES = 2
_SPECIAL_RE = re.compile(r"<\|[^|]+\|>")


class _ThinkingTracker:
    """Parses raw streamed text, shows thinking in a small live box, then folds it."""

    def __init__(self, console: Console, enable_thinking: bool):
        self.console = console
        self.enable_thinking = enable_thinking
        self.raw = ""
        self.in_thinking = enable_thinking
        self.thinking_tokens = 0
        self.thinking_text = ""
        self.response_started = False
        self.streamed_response_len = 0
        self._live: Live | None = None
        self._waiting: Live | None = None
        self._think_start = time.perf_counter()
        self.thinking_elapsed: float = 0.0

    def start_waiting(self):
        """Show a spinner while waiting for the first token (non-thinking mode only)."""
        if not self.enable_thinking:
            self._waiting = Live(
                Spinner("dots", text=" Generating...", style="dim"),
                console=self.console,
                refresh_per_second=10,
                transient=True,
            )
            self._waiting.start()

    def on_token(self, text: str):
        if self._waiting is not None:
            self._waiting.stop()
            self._waiting = None
        self.raw += text

        if self.in_thinking:
            self.thinking_tokens += 1
            self.thinking_text += text
            if "</think>" in self.raw:
                self.in_thinking = False
                self._fold_thinking()
                self.response_started = True
                self._emit_response()
            else:
                self._update_thinking_box()
        else:
            self.response_started = True
            self._emit_response()

    def _update_thinking_box(self):
        lines = self.thinking_text.strip().splitlines()
        tail = "\n".join(lines[-_THINKING_LINES:])
        elapsed = time.perf_counter() - self._think_start
        title = f"thinking ({elapsed:.1f}s)"
        panel = Panel(
            tail or "...",
            title=title,
            border_style="dim",
            width=min(self.console.width, 80),
            height=_THINKING_LINES + 2,
        )
        if self._live is None:
            self._live = Live(panel, console=self.console, refresh_per_second=8, transient=True)
            self._live.start()
        else:
            self._live.update(panel)

    def _fold_thinking(self):
        if self._live is not None:
            self._live.stop()
            self._live = None
        self.thinking_elapsed = time.perf_counter() - self._think_start

    def _emit_response(self):
        response = self._get_response()
        delta = response[self.streamed_response_len :]
        if delta:
            self.streamed_response_len = len(response)
            self.console.print(delta, end="", highlight=False, soft_wrap=True)

    def _get_response(self) -> str:
        if self.enable_thinking and "</think>" in self.raw:
            text = self.raw.split("</think>", 1)[1].lstrip("\n")
        elif self.in_thinking:
            text = ""
        else:
            text = self.raw
        return _SPECIAL_RE.sub("", text)

    def stop(self):
        if self._waiting is not None:
            self._waiting.stop()
            self._waiting = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    @property
    def output_text(self) -> str:
        return self._get_response().strip()


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

    console = Console(theme=Theme({"hint": "dim"}))

    from paroquant.inference.base import _detect_backend

    backend = _detect_backend() if config.backend == "auto" else config.backend

    kwargs = {}
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
                user_prompt = console.input(">>> ").strip()
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

            prompt = build_prompt(generator.tokenizer, history, config.enable_thinking)

            tracker = _ThinkingTracker(console, config.enable_thinking)
            tracker.start_waiting()

            generation_ctx = contextlib.nullcontext() if config.debug else _silence_stderr()
            with generation_ctx:
                result = await generator.generate(prompt, config.params, on_text=tracker.on_token)
            tracker.stop()

            if not tracker.response_started and result.output_text:
                console.print(result.output_text, highlight=False, soft_wrap=True)
            console.print()

            output = tracker.output_text if config.enable_thinking else result.output_text
            history.append({"role": "assistant", "content": output})

            s = result.stats
            parts = [f"{s.num_tokens} tokens"]
            if tracker.thinking_elapsed > 0:
                parts.append(f"thinking {tracker.thinking_elapsed:.1f}s")
            parts.append(f"{s.tps:.1f} tok/s")
            console.print(f"  {' · '.join(parts)}", style="hint", highlight=False)
            console.print()
    finally:
        await generator.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers", "mlx"],
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--thinking", action="store_true", dest="enable_thinking", default=True)
    parser.add_argument("--no-thinking", action="store_false", dest="enable_thinking")
    parser.add_argument("--debug", action="store_true")
    return parser


async def run_from_args(args: argparse.Namespace):
    config = ChatAppConfig(
        model=args.model,
        backend=args.backend,
        params=GenerationParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_thinking=args.enable_thinking,
        debug=args.debug,
    )
    await run_chat_app(config)


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    asyncio.run(run_from_args(cli_args))
