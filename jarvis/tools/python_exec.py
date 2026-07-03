"""Local Python execution.

Runs a snippet in a fresh subprocess (this interpreter) with a timeout, inside
the workspace. Useful for computation, data wrangling, and quick verification.
Distinct from Claude's server-side code execution — this runs on the host.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

from .base import Tool, ToolContext, ToolResult

MAX_OUTPUT = 30_000
DEFAULT_TIMEOUT = 120


class RunPython(Tool):
    name = "run_python"
    description = (
        "Execute a Python snippet in a fresh subprocess and return its stdout/stderr. "
        "The working directory is the workspace root. Use print() to return values."
    )
    mutating = True
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute"},
            "timeout": {"type": "integer", "description": f"Seconds (default {DEFAULT_TIMEOUT}, max 600)"},
        },
        "required": ["code"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        gate = self._gate(ctx, "execute a Python snippet")
        if gate:
            return gate
        timeout = min(int(args.get("timeout", DEFAULT_TIMEOUT)), 600)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", args["code"]],
                cwd=str(ctx.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error(f"Execution timed out after {timeout}s.")
        except OSError as e:
            return ToolResult.error(f"Failed to launch Python: {e}")
        output = (proc.stdout or "") + (proc.stderr or "")
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + "\n... (output truncated)"
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        return ToolResult.ok(f"[{status}]\n{output.strip() or '(no output)'}")


PYTHON_TOOLS = [RunPython()]
