"""Interactive agent demo using a local ParoQuant MLX model via qwen_agent.

Usage:
    # Terminal 1: start the server
    python -m paroquant.cli.serve_mlx --model z-lab/Qwen3.5-9B-PARO --port 8092 --max-tokens 4096

    # Terminal 2: run the agent
    python experiments/agent_demo.py
    python experiments/agent_demo.py --model-server http://127.0.0.1:8092/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from qwen_agent.agents import Assistant  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.theme import Theme  # noqa: E402

_SPECIAL_RE = re.compile(r"<\|[^|]+\|>")
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"^.*?</think>\n?", re.DOTALL)

SYSTEM_PROMPT = """\
You are ParoQuant Agent, a helpful AI assistant powered by a local quantized model.
You have access to tools that let you interact with the real world:
- Get the current time in any timezone
- Fetch and read web pages
- Read, write, and list files in the workspace directory
Use tools when the user's request requires real-world information or actions.
Think step by step before acting. Be concise in your final answers."""

_THINKING_LINES = 4


def make_agent(model_server: str, model: str, workspace: str) -> Assistant:
    llm_cfg = {
        "model": model,
        "model_type": "qwenvl_oai",
        "model_server": model_server,
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        "generate_cfg": {
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

    mcp_servers: dict = {}
    mcp_servers["time"] = {
        "command": "uvx",
        "args": ["mcp-server-time", "--local-timezone=US/Pacific"],
    }
    mcp_servers["fetch"] = {
        "command": "uvx",
        "args": ["mcp-server-fetch"],
    }
    mcp_servers["filesystem"] = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", workspace],
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


def _clean(text: str) -> str:
    text = _SPECIAL_RE.sub("", text)
    text = _TOOL_CALL_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    return text.strip()


def _get_thinking(text: str) -> str | None:
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[1] if "<think>" in text else text
    return None


def _extract_final_content(response_parts) -> str:
    if not response_parts:
        return ""
    items = response_parts if isinstance(response_parts, list) else [response_parts]
    for msg in reversed(items):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not content:
            continue
        cleaned = _clean(content)
        if cleaned:
            return cleaned
    return ""


def _extract_tool_info(response_parts) -> tuple[str | None, str | None]:
    """Extract (tool_name, tool_args_json) from response."""
    if not isinstance(response_parts, list):
        return None, None
    for msg in response_parts:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if "<tool_call>" in content:
            match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
            args_match = re.search(r'"arguments"\s*:\s*(\{[^}]*\})', content)
            name = match.group(1) if match else None
            args = args_match.group(1) if args_match else None
            return name, args
    return None, None


def _extract_tool_response(response_parts) -> tuple[str | None, str | None]:
    """Extract (tool_name, response_summary) from tool response."""
    if not isinstance(response_parts, list):
        return None, None
    for msg in response_parts:
        if isinstance(msg, dict) and msg.get("role") == "function":
            name = msg.get("name", "")
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            return name, content
    return None, None


def _render_thinking_box(console_width: int, text: str, elapsed: float) -> Panel:
    lines = text.splitlines()
    tail = "\n".join(lines[-_THINKING_LINES:])
    return Panel(
        tail or "...",
        title=f"thinking ({elapsed:.1f}s)",
        border_style="dim",
        width=min(console_width, 80),
        height=_THINKING_LINES + 2,
    )


def main():
    parser = argparse.ArgumentParser(description="ParoQuant Agent Demo")
    parser.add_argument("--model-server", type=str, default="http://127.0.0.1:8092/v1")
    parser.add_argument("--model", type=str, default="z-lab/Qwen3.5-9B-PARO")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--system-prompt", type=str, default=None)
    args = parser.parse_args()

    workspace = args.workspace or tempfile.mkdtemp(prefix="paroquant-agent-")

    console = Console(theme=Theme({"hint": "dim", "tool": "cyan", "stat": "dim"}))

    console.print("[hint]Connecting to model server and initializing tools...[/hint]")
    if args.system_prompt:
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = args.system_prompt
    agent = make_agent(args.model_server, args.model, workspace)

    banner = (
        f"[bold]ParoQuant Agent[/bold]\n"
        f"Model: [bold]{args.model}[/bold]\n"
        f"Server: [bold]{args.model_server}[/bold]\n"
        f"Workspace: [bold]{workspace}[/bold]\n\n"
        f"[bold]/clear[/bold] · [bold]/system[/bold] · [bold]/quit[/bold]"
    )
    console.print(Panel.fit(banner, border_style="bright_blue"))

    messages: list[dict] = []

    while True:
        try:
            user_input = console.input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[hint]Session closed.[/hint]")
            break

        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in {"/quit", "quit", "/exit", "exit"}:
            break
        if cmd == "/clear":
            messages.clear()
            console.clear()
            console.print("[hint]Conversation history cleared.[/hint]\n")
            continue
        if cmd == "/system":
            console.print(Panel(SYSTEM_PROMPT, title="System Prompt", border_style="dim"))
            continue

        messages.append({"role": "user", "content": user_input})

        last_response = []
        final_content = ""
        seen_tools: set[str] = set()
        seen_tool_responses: set[str] = set()
        turn_start = time.perf_counter()
        live: Live | None = None
        phase = "idle"

        debug_log = []
        try:
            for response in agent.run(messages=messages):
                last_response = response
                elapsed = time.perf_counter() - turn_start
                debug_log.append(repr(response)[:500])

                # Check for tool call
                tool_name, tool_args = _extract_tool_info(response)
                if tool_name and tool_name not in seen_tools:
                    seen_tools.add(tool_name)
                    if live:
                        live.stop()
                        live = None
                    args_str = ""
                    if tool_args:
                        try:
                            parsed = json.loads(tool_args)
                            args_str = " ".join(f"{k}={v}" for k, v in parsed.items())
                        except json.JSONDecodeError:
                            args_str = tool_args
                    console.print(f"  [tool]> {tool_name}[/tool]({args_str})", highlight=False)
                    phase = "tool"
                    continue

                # Check for tool response
                resp_name, resp_summary = _extract_tool_response(response)
                if resp_name and resp_name not in seen_tool_responses:
                    seen_tool_responses.add(resp_name)
                    console.print(
                        Panel(
                            resp_summary or "ok",
                            title=f"{resp_name} response",
                            border_style="green",
                            width=min(console.width, 80),
                            height=min(8, (resp_summary or "").count("\n") + 3),
                        )
                    )
                    phase = "tool_response"
                    continue

                # Get raw assistant content
                raw_content = ""
                if isinstance(response, list):
                    for msg in response:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            raw_content = msg.get("content", "")
                            break

                # Thinking phase
                thinking = _get_thinking(raw_content)
                if thinking is not None:
                    if phase != "thinking":
                        if live:
                            live.stop()
                        live = Live(console=console, transient=True)
                        live.start()
                        phase = "thinking"
                    live.update(_render_thinking_box(console.width, thinking, elapsed))
                    continue

                # Final answer
                content = _extract_final_content(response)
                if content and content != final_content:
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

        # Debug: dump last 3 responses if no content captured
        if not final_content:
            debug_path = os.path.join(workspace, "debug_last_run.txt")
            with open(debug_path, "w") as f:
                f.write(f"Total yields: {len(debug_log)}\n\n")
                for i, entry in enumerate(debug_log[-5:]):
                    f.write(f"--- yield {len(debug_log) - 5 + i} ---\n{entry}\n\n")
            console.print(f"[hint]Debug log written to {debug_path}[/hint]")

        # Fallback: if no final_content was captured, extract from last response
        if not final_content and last_response:
            items = last_response if isinstance(last_response, list) else [last_response]
            for msg in reversed(items):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    raw = msg.get("content", "")
                    cleaned = _clean(raw)
                    if cleaned:
                        final_content = cleaned
                        console.print(Markdown(final_content))
                        break

        # Stats
        total_time = time.perf_counter() - turn_start
        parts = []
        if seen_tools:
            parts.append(f"tools: {', '.join(seen_tools)}")
        parts.append(f"{total_time:.1f}s total")
        console.print(f"  {' · '.join(parts)}", style="stat", highlight=False)

        if last_response:
            messages.extend(last_response if isinstance(last_response, list) else [last_response])

        console.print()


if __name__ == "__main__":
    main()
