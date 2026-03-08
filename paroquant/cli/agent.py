from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
import warnings

from qwen_agent.agents import Assistant
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

_SPECIAL_RE = re.compile(r"<\|[^|]+\|>")
_THINK_RE = re.compile(r"^.*?</think>\n?", re.DOTALL)
_THINKING_LINES = 4

SYSTEM_PROMPT = """\
You are ParoQuant Agent, a helpful AI assistant powered by a local quantized model.
You have access to tools that let you interact with the real world:
- Get the current time in any timezone
- Fetch and read web pages
- Read, write, and list files in the workspace directory
Use tools when the user's request requires real-world information or actions.
Be concise in your final answers."""


def _suppress_library_noise():
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)


def _make_agent(server: str, model: str, workspace: str, max_tokens: int = 8192) -> Assistant:
    llm_cfg = {
        "model": model,
        "model_type": "qwenvl_oai",
        "model_server": server,
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        "generate_cfg": {
            "max_tokens": max_tokens,
            "timeout": 120,
            "max_retries": 1,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    }

    mcp_servers = {
        "time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone=US/Pacific"]},
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", workspace]},
    }

    tools: list = []
    try:
        import mcp  # noqa: F401

        tools = [{"mcpServers": mcp_servers}]
    except ImportError:
        pass

    return Assistant(
        llm=llm_cfg,
        function_list=tools or None,
        name="ParoQuant Agent",
        description="Agent powered by a local ParoQuant quantized model with tool calling.",
        system_message=SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    return _THINK_RE.sub("", _SPECIAL_RE.sub("", text)).strip()


def _get_thinking(text: str) -> str | None:
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[1]
    return None


def _friendly_tool_name(name: str) -> str:
    parts = name.split("-", 1)
    return parts[1] if len(parts) > 1 else name


def _format_tool_args(args_json: str) -> str:
    try:
        return ", ".join(f"{k}={v}" for k, v in json.loads(args_json).items())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return args_json or ""


def _last_msg_with(msgs: list[dict], *, role: str = "", key: str = "") -> tuple[dict | None, int]:
    """Return the last message matching role/key and its index, or (None, -1)."""
    for i in reversed(range(len(msgs))):
        m = msgs[i]
        if not isinstance(m, dict):
            continue
        if role and m.get("role") != role:
            continue
        if key and not m.get(key):
            continue
        return m, i
    return None, -1


def _thinking_panel(width: int, text: str, elapsed: float) -> Panel:
    tail = "\n".join(text.splitlines()[-_THINKING_LINES:])
    return Panel(
        tail or "...",
        title=f"thinking ({elapsed:.1f}s)",
        border_style="dim",
        width=min(width, 80),
        height=_THINKING_LINES + 2,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    _suppress_library_noise()

    parser = argparse.ArgumentParser(description="ParoQuant Agent")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    workspace = args.workspace or tempfile.mkdtemp(prefix="paroquant-agent-")
    console = Console(theme=Theme({"hint": "dim", "tool": "cyan"}))

    console.print("[hint]Connecting to model server and initializing tools...[/hint]")
    agent = _make_agent(args.server, args.model, workspace, args.max_tokens)

    console.print("[hint]Warming up...[/hint]")
    warmup = _make_agent(args.server, args.model, workspace, max_tokens=16)
    for _ in warmup.run(messages=[{"role": "user", "content": "Hi"}]):
        pass
    del warmup

    console.clear()

    console.print(
        Panel.fit(
            f"[bold]ParoQuant Agent[/bold]\n"
            f"Model: [bold]{args.model}[/bold]\n"
            f"Server: [bold]{args.server}[/bold]\n\n"
            f"[bold]/clear[/bold] · [bold]/quit[/bold]",
            border_style="bright_blue",
        )
    )

    messages: list[dict] = []

    while True:
        try:
            user_input = console.input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[hint]Session closed.[/hint]")
            break
        if not user_input:
            continue
        if user_input.lower() in {"/quit", "quit", "/exit", "exit"}:
            break
        if user_input.lower() == "/clear":
            messages.clear()
            console.clear()
            console.print("[hint]Conversation history cleared.[/hint]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        last_response: list[dict] = []
        final_content = ""
        turn_start = time.perf_counter()
        live: Live | None = None
        phase = "idle"
        seen_tool_idx = seen_resp_idx = -1
        tool_time = 0.0
        tool_timer_start: float | None = None

        try:
            for response in agent.run(messages=messages):
                if not isinstance(response, list):
                    continue
                last_response = response
                elapsed = time.perf_counter() - turn_start

                # Detect new tool call
                tc_msg, tc_idx = _last_msg_with(response, key="function_call")
                if tc_idx > seen_tool_idx and tc_idx == len(response) - 1:
                    seen_tool_idx = tc_idx
                    tool_timer_start = time.perf_counter()
                    phase = "tool"
                    continue

                # Detect new tool response
                _, resp_idx = _last_msg_with(response, role="function")
                if resp_idx > seen_resp_idx and resp_idx == len(response) - 1:
                    seen_resp_idx = resp_idx
                    if tool_timer_start is not None:
                        tool_time += time.perf_counter() - tool_timer_start
                        tool_timer_start = None
                    continue

                # Tool call completed — print summary and move on
                if phase == "tool" and tc_msg is not None and tc_idx != len(response) - 1:
                    if live:
                        live.stop()
                        live = None
                    fc = tc_msg["function_call"]
                    name = _friendly_tool_name(fc.get("name", ""))
                    console.print(f"  [tool]> {name}({_format_tool_args(fc.get('arguments', ''))})[/tool]")
                    resp_msg, _ = _last_msg_with(response, role="function")
                    if resp_msg:
                        text = resp_msg.get("content", "").replace("\n", " ")
                        if len(text) > 100:
                            text = text[:100] + "..."
                        console.print(f"    [hint]{text}[/hint]")
                    phase = "generating"

                # Extract latest assistant content
                asst_msg, _ = _last_msg_with(response, role="assistant")
                raw = (asst_msg or {}).get("content", "")
                if not raw:
                    continue

                # Show thinking box
                thinking = _get_thinking(raw)
                if thinking is not None:
                    if phase != "thinking":
                        if live:
                            live.stop()
                        live = Live(console=console, transient=True)
                        live.start()
                        phase = "thinking"
                    live.update(_thinking_panel(console.width, thinking, elapsed))
                    continue

                # Stream answer
                content = _clean(raw)
                if not content or content == final_content:
                    continue
                if phase != "answering":
                    if live:
                        live.stop()
                    live = Live(Markdown(content), console=console, vertical_overflow="visible")
                    live.start()
                    phase = "answering"
                else:
                    live.update(Markdown(content))
                final_content = content

        except Exception as e:
            if live:
                live.stop()
            console.print(f"[red]Error: {e}[/red]\n")
            messages.pop()
            continue

        if live:
            if final_content:
                live.update(Markdown(final_content))
            live.stop()

        if not final_content:
            asst_msg, _ = _last_msg_with(last_response, role="assistant")
            final_content = _clean((asst_msg or {}).get("content", ""))
            if final_content:
                console.print(Markdown(final_content))

        if tool_timer_start is not None:
            tool_time += time.perf_counter() - tool_timer_start

        total = time.perf_counter() - turn_start
        stats = []
        if tool_time > 0:
            stats.append(f"tools {tool_time:.1f}s")
        stats.append(f"{total:.1f}s total")
        console.print(f"  {' · '.join(stats)}", style="hint", highlight=False)

        if last_response:
            messages.extend(last_response)
        console.print()


if __name__ == "__main__":
    main()
