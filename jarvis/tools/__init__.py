"""Tool registry — assembles the custom tool surface for an agent."""
from __future__ import annotations

from ..config import Config
from .base import PathError, Tool, ToolContext, ToolResult
from .filesystem import FILESYSTEM_TOOLS
from .memory_tool import MEMORY_TOOLS
from .python_exec import PYTHON_TOOLS
from .shell import SHELL_TOOLS
from .subagent import SUBAGENT_TOOLS
from .tasks import TASK_TOOLS, render_tasks
from .web import web_tool_declarations

__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "PathError",
    "ToolRegistry",
    "build_tools",
    "web_tool_declarations",
    "render_tasks",
]


def build_tools(config: Config, *, include_subagents: bool = True) -> list[Tool]:
    """The custom (client-executed) tools available to an agent."""
    tools: list[Tool] = []
    tools += FILESYSTEM_TOOLS
    tools += SHELL_TOOLS
    tools += PYTHON_TOOLS
    tools += MEMORY_TOOLS
    tools += TASK_TOOLS
    if include_subagents and config.enable_subagents:
        tools += SUBAGENT_TOOLS
    return tools


class ToolRegistry:
    """Name → tool lookup, plus API schema assembly."""

    def __init__(self, tools: list[Tool]):
        self._by_name: dict[str, Tool] = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return list(self._by_name)

    def custom_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._by_name.values()]

    def overview(self) -> str:
        lines = []
        for t in self._by_name.values():
            first = t.description.strip().splitlines()[0] if t.description.strip() else ""
            lines.append(f"- {t.name}: {first}")
        return "\n".join(lines)
