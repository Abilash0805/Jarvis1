"""The working task list.

A lightweight, in-session plan the agent maintains as it works. Kept in the
tool context's shared state and rendered live by the UI, so a long autonomous
run stays legible.
"""
from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolResult

STATES = ("pending", "in_progress", "done", "blocked")
_ICON = {"pending": "☐", "in_progress": "▸", "done": "✓", "blocked": "✗"}


def _tasks(ctx: ToolContext) -> list[dict[str, Any]]:
    return ctx.state.setdefault("tasks", [])


def render_tasks(ctx: ToolContext) -> str:
    tasks = _tasks(ctx)
    if not tasks:
        return "(no tasks)"
    return "\n".join(f"{_ICON.get(t['status'], '?')} [{t['id']}] {t['title']}" for t in tasks)


class UpdateTasks(Tool):
    name = "update_tasks"
    description = (
        "Create or update the working task list for a multi-step job. Replaces the "
        "whole list. Keep exactly one task in_progress at a time. Use this to plan "
        "before executing and to track progress as you go."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "status": {"type": "string", "enum": list(STATES)},
                    },
                    "required": ["title", "status"],
                },
            }
        },
        "required": ["tasks"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        incoming = args.get("tasks", [])
        rebuilt = []
        for i, t in enumerate(incoming, 1):
            status = t.get("status", "pending")
            if status not in STATES:
                status = "pending"
            title = str(t.get("title", "")).strip()
            if not title:
                continue
            rebuilt.append({"id": i, "title": title, "status": status})
        ctx.state["tasks"] = rebuilt
        ctx.console.render_tasks(rebuilt)
        done = sum(1 for t in rebuilt if t["status"] == "done")
        return ToolResult.ok(f"Task list updated ({done}/{len(rebuilt)} done).")


TASK_TOOLS = [UpdateTasks()]
