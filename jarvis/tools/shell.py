"""Sandboxed shell execution.

Runs commands inside the workspace with a timeout and output cap. A hard
blocklist rejects catastrophic commands in *every* mode; ordinary commands are
gated behind confirmation in interactive mode and run freely when autonomous.
"""
from __future__ import annotations

import re
import subprocess
from typing import Any

from .base import Tool, ToolContext, ToolResult

MAX_OUTPUT = 30_000
DEFAULT_TIMEOUT = 120

# Patterns that are never allowed to run, regardless of mode. Deliberately small
# and specific — this is a guard against catastrophe, not a general policy engine.
CATASTROPHIC = [
    re.compile(r"\brm\s+-rf?\s+(/|/\*|~|\$HOME)(\s|$)"),
    re.compile(r"\brm\s+-rf?\s+--no-preserve-root"),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),  # fork bomb
    re.compile(r"\bmkfs\.\w+\b"),
    re.compile(r"\bdd\b.*\bof=/dev/(sd|nvme|hd|disk)"),
    re.compile(r">\s*/dev/(sd|nvme|hd|disk)"),
    re.compile(r"\bchmod\s+-R\s+0*777\s+/(\s|$)"),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bmv\s+.+\s+/dev/null\b"),
]


def is_catastrophic(command: str) -> str | None:
    """Return a human-readable reason if the command is forbidden, else None."""
    for rx in CATASTROPHIC:
        if rx.search(command):
            return f"blocked by catastrophic-command guard (pattern: {rx.pattern})"
    return None


class Bash(Tool):
    name = "bash"
    description = (
        "Run a shell command inside the workspace. Returns combined stdout+stderr "
        "and the exit code. Use for builds, tests, git, and general shell work. "
        "Prefer the dedicated file tools for reading/editing files."
    )
    mutating = True
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
            "timeout": {"type": "integer", "description": f"Seconds (default {DEFAULT_TIMEOUT}, max 600)"},
        },
        "required": ["command"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args["command"]
        reason = is_catastrophic(command)
        if reason:
            return ToolResult.error(f"Refused: {reason}")
        gate = self._gate(ctx, f"run shell command: {command}")
        if gate:
            return gate
        timeout = min(int(args.get("timeout", DEFAULT_TIMEOUT)), 600)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(ctx.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error(f"Command timed out after {timeout}s.")
        except OSError as e:
            return ToolResult.error(f"Failed to run command: {e}")
        output = (proc.stdout or "") + (proc.stderr or "")
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + "\n... (output truncated)"
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        body = output.strip() or "(no output)"
        result = f"[{status}]\n{body}"
        # A non-zero exit is information, not a tool failure — let the model react.
        return ToolResult.ok(result)


SHELL_TOOLS = [Bash()]
