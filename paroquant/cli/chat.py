from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import io
import os
import re
import time
import warnings

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

from paroquant.inference import (
    GenerationParams,
    build_prompt,
    create_generator,
    detect_backend,
)


@contextlib.contextmanager
def _silence_stderr():
    with contextlib.redirect_stderr(io.StringIO()):
        yield


_THINKING_LINES = 4
_SPECIAL_RE = re.compile(r"<\|[^|]+\|>")


class _ThinkingTracker:
    """Parses raw streamed text, shows thinking in a small live box, then folds it."""

    _REFRESH_INTERVAL = 1.0 / 24

    def __init__(self, console: Console, enable_thinking: bool):
        self.console = console
        self.enable_thinking = enable_thinking
        self.raw = ""
        self.in_thinking = enable_thinking
        self._live: Live | None = None
        self._think_start = time.perf_counter()
        self._last_update = 0.0

    def on_token(self, text: str):
        self.raw += text

        if self.in_thinking:
            if "</think>" in self.raw:
                self.in_thinking = False
                self.stop()
                self._start_response_live()
            else:
                self._update_thinking_box()
        else:
            if self._live is None:
                self._start_response_live()
            else:
                self._update_response()

    def _update_thinking_box(self):
        now = time.perf_counter()
        if self._live is not None and now - self._last_update < self._REFRESH_INTERVAL:
            return
        self._last_update = now

        lines = self.raw.splitlines()
        tail = "\n".join(lines[-_THINKING_LINES:])
        panel = Panel(
            tail or "...",
            title=f"thinking ({now - self._think_start:.1f}s)",
            border_style="dim",
            width=min(self.console.width, 80),
            height=_THINKING_LINES + 2,
        )
        if self._live is None:
            self._live = Live(panel, console=self.console, transient=True)
            self._live.start()
        else:
            self._live.update(panel)

    def _start_response_live(self):
        self._live = Live(Markdown(""), console=self.console, vertical_overflow="visible")
        self._live.start()
        self._update_response()

    def _update_response(self):
        now = time.perf_counter()
        if now - self._last_update < self._REFRESH_INTERVAL:
            return
        self._last_update = now

        response = self._get_response()
        if response and self._live is not None:
            self._live.update(Markdown(response))

    def _get_response(self) -> str:
        if self.enable_thinking and "</think>" in self.raw:
            text = self.raw.split("</think>", 1)[1].lstrip("\n")
        elif self.in_thinking:
            text = ""
        else:
            text = self.raw
        return _SPECIAL_RE.sub("", text)

    def stop(self):
        if self._live is not None:
            response = self._get_response()
            if response:
                self._live.update(Markdown(response))
            self._live.stop()
            self._live = None

    @property
    def output_text(self) -> str:
        return self._get_response().strip()


def _suppress_library_noise():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    for mod, fn in [
        ("huggingface_hub", "disable_progress_bars"),
        ("transformers.utils.logging", "set_verbosity_error"),
    ]:
        try:
            getattr(importlib.import_module(mod), fn)()
        except Exception:
            pass


async def run_chat_app(model: str, backend: str, params: GenerationParams):
    _suppress_library_noise()

    console = Console(theme=Theme({"hint": "dim"}))

    backend = detect_backend() if backend == "auto" else backend

    console.print(f"[hint]Loading model ({backend})...[/hint]")
    generator = create_generator(backend, model)

    console.print("[hint]Warming up...[/hint]")
    warmup_prompt = build_prompt(generator.tokenizer, [{"role": "user", "content": "Hi"}], False)
    await generator.generate(warmup_prompt, GenerationParams(max_tokens=1, temperature=0.0))

    console.clear()

    enable_thinking = False

    banner = (
        f"[bold]ParoQuant Chat[/bold]\n"
        f"Backend: [bold]{backend}[/bold]\n"
        f"Model: [bold]{model}[/bold]\n\n"
        f"[bold]/think[/bold] · [bold]/clear[/bold] · [bold]/quit[/bold]"
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
            cmd = user_prompt.lower()
            if cmd in {"/quit", "quit", "/exit", "exit"}:
                break
            if cmd == "/clear":
                history.clear()
                console.clear()
                console.print("[hint]Conversation history cleared.[/hint]\n")
                continue
            if cmd == "/think":
                enable_thinking = not enable_thinking
                console.print(f"[hint]Thinking {'on' if enable_thinking else 'off'}.[/hint]\n")
                continue

            history.append({"role": "user", "content": user_prompt})

            prompt = build_prompt(generator.tokenizer, history, enable_thinking)

            tracker = _ThinkingTracker(console, enable_thinking)
            with _silence_stderr():
                result = await generator.generate(prompt, params, on_text=tracker.on_token)
            tracker.stop()

            history.append({"role": "assistant", "content": tracker.output_text})

            s = result.stats
            parts = []
            if s.ttft is not None:
                parts.append(f"ttft {s.ttft:.2f}s")
            parts += [f"{s.num_tokens} tokens", f"{s.tps:.1f} tok/s", f"{s.latency:.1f}s total"]
            console.print(f"  {' · '.join(parts)}", style="hint", highlight=False)
            console.print()
    finally:
        await generator.close()


def main():
    _suppress_library_noise()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "transformers", "mlx"])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    asyncio.run(
        run_chat_app(
            model=args.model,
            backend=args.backend,
            params=GenerationParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            ),
        )
    )


if __name__ == "__main__":
    main()
