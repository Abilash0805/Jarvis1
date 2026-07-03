"""Terminal UI built on `rich`.

Handles the banner, live streaming of thinking + assistant text, tool-call
rendering, the task list, and the interactive permission prompt. Degrades to
plain output if a stream isn't a TTY.
"""
from __future__ import annotations

import sys
from typing import Any

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.text import Text

from . import BANNER


class Console:
    def __init__(self, *, quiet: bool = False, show_thinking: bool = True):
        self.rich = RichConsole()
        self.quiet = quiet
        self.show_thinking = show_thinking
        self._mode: str | None = None  # "thinking" | "text" | None

    # -- banner / structure -------------------------------------------------

    def banner(self, subtitle: str = "") -> None:
        self.rich.print(f"[bold cyan]{BANNER}[/bold cyan]")
        if subtitle:
            self.rich.print(f"[dim]{subtitle}[/dim]\n")

    def rule(self, label: str = "") -> None:
        self.rich.rule(f"[bold]{label}[/bold]" if label else "")

    def info(self, msg: str) -> None:
        self.rich.print(f"[dim]{msg}[/dim]")

    def warn(self, msg: str) -> None:
        self.rich.print(f"[yellow]⚠ {msg}[/yellow]")

    def error(self, msg: str) -> None:
        self.rich.print(f"[bold red]✗ {msg}[/bold red]")

    def user_prompt(self) -> str:
        return "[bold green]you ›[/bold green] "

    # -- live streaming -----------------------------------------------------

    def _switch(self, mode: str) -> None:
        if self._mode == mode:
            return
        self.end_stream()
        self._mode = mode
        if mode == "thinking":
            self.rich.print("[dim italic]· thinking ·[/dim italic]")
        elif mode == "text":
            self.rich.print("[bold cyan]jarvis ›[/bold cyan]")

    def stream_thinking(self, delta: str) -> None:
        if self.quiet or not self.show_thinking or not delta:
            return
        self._switch("thinking")
        self.rich.file.write(_dim(delta))
        self.rich.file.flush()

    def stream_text(self, delta: str) -> None:
        if not delta:
            return
        self._switch("text")
        self.rich.file.write(delta)
        self.rich.file.flush()

    def end_stream(self) -> None:
        if self._mode is not None:
            self.rich.file.write("\n")
            self.rich.file.flush()
            self._mode = None

    # -- tools --------------------------------------------------------------

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        self.end_stream()
        preview = _preview_args(name, args)
        self.rich.print(f"[magenta]⚙ {name}[/magenta] [dim]{preview}[/dim]")

    def tool_result(self, name: str, content: str, is_error: bool) -> None:
        if self.quiet:
            return
        snippet = content.strip().splitlines()
        head = "\n".join(snippet[:8])
        more = f"\n[dim]... (+{len(snippet) - 8} lines)[/dim]" if len(snippet) > 8 else ""
        color = "red" if is_error else "green"
        mark = "✗" if is_error else "✓"
        self.rich.print(
            Panel(
                Text(head + ("" if not more else ""), style="dim") if not more else Text(head, style="dim"),
                title=f"[{color}]{mark} {name}[/{color}]",
                title_align="left",
                border_style=color,
                expand=False,
            )
        )
        if more:
            self.rich.print(more)

    def render_tasks(self, tasks: list[dict[str, Any]]) -> None:
        self.end_stream()
        if not tasks:
            return
        icon = {"pending": "☐", "in_progress": "[cyan]▸[/cyan]", "done": "[green]✓[/green]", "blocked": "[red]✗[/red]"}
        lines = [f"{icon.get(t['status'], '?')} {t['title']}" for t in tasks]
        self.rich.print(Panel("\n".join(lines), title="[bold]plan[/bold]", title_align="left", border_style="blue", expand=False))

    # -- interaction --------------------------------------------------------

    def confirm(self, action: str) -> bool:
        self.end_stream()
        self.rich.print(f"[yellow]Allow:[/yellow] {action}")
        try:
            ans = input("  [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes")


def _dim(text: str) -> str:
    # Manual dim for raw writes (we're bypassing rich markup on the fast path).
    return f"\033[2m{text}\033[0m" if sys.stdout.isatty() else text


def _preview_args(name: str, args: dict[str, Any]) -> str:
    if not args:
        return ""
    key_order = ["path", "command", "pattern", "query", "objective", "code", "content"]
    for k in key_order:
        if k in args:
            v = str(args[k]).replace("\n", " ")
            return (v[:100] + "…") if len(v) > 100 else v
    # Fallback: first value
    k, v = next(iter(args.items()))
    v = str(v).replace("\n", " ")
    return f"{k}={(v[:80] + '…') if len(v) > 80 else v}"
