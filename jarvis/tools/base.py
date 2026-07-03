"""Tool base classes and the shared execution context.

A :class:`Tool` is a name, a description, a JSON-Schema for its input, and a
``run`` method. Tools receive a :class:`ToolContext` giving them the workspace
root, config, memory, a permission gate, and (for delegation) the agent factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # avoid import cycles at runtime
    from ..config import Config
    from ..memory import Memory
    from ..ui import Console


@dataclass
class ToolResult:
    content: str
    is_error: bool = False

    @staticmethod
    def ok(content: str) -> "ToolResult":
        return ToolResult(content=content, is_error=False)

    @staticmethod
    def error(content: str) -> "ToolResult":
        return ToolResult(content=content, is_error=True)


@dataclass
class ToolContext:
    """Everything a tool may need, injected at run time."""

    config: "Config"
    memory: "Memory"
    console: "Console"
    workspace: Path
    autonomous: bool
    # Returns True if the action is permitted. Autonomous mode auto-approves.
    confirm: Callable[[str], bool]
    # Factory that runs a sub-agent objective and returns its final report.
    # Wired by the Agent so tools don't import it directly (avoids a cycle).
    delegate: Callable[[str, str], str] | None = None
    depth: int = 0
    # Per-session scratch shared across tool calls (e.g. the task list).
    state: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, path: str) -> Path:
        """Resolve a user/model-supplied path and confine it to the workspace.

        Rejects anything that escapes the workspace root (``..``, absolute paths
        outside the root, symlink traversal). Raises :class:`PathError`.
        """
        raw = Path(path)
        candidate = raw if raw.is_absolute() else (self.workspace / raw)
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:  # e.g. symlink loop
            raise PathError(f"cannot resolve path {path!r}: {exc}") from exc
        root = self.workspace.resolve()
        if resolved != root and root not in resolved.parents:
            raise PathError(
                f"path {path!r} escapes the workspace root ({root}); refused"
            )
        return resolved


class PathError(Exception):
    """Raised when a tool is asked to touch a path outside the workspace."""


class Tool:
    """Base class for a custom (client-executed) tool."""

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    # Actions that mutate state or reach outward are gated in interactive mode.
    mutating: bool = False

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description.strip(),
            "input_schema": self.input_schema,
        }

    # Convenience for subclasses.
    def _gate(self, ctx: ToolContext, action: str) -> ToolResult | None:
        """Return an error ToolResult if the user declines a mutating action."""
        if self.mutating and not ctx.confirm(action):
            return ToolResult.error(f"Permission denied by user for: {action}")
        return None
