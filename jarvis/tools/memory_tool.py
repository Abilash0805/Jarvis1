"""Memory tools — the agent's window onto its persistent store."""
from __future__ import annotations

from typing import Any

from ..memory.store import VALID_KINDS
from .base import Tool, ToolContext, ToolResult


class Remember(Tool):
    name = "remember"
    description = (
        "Save a durable memory that persists across sessions. Use for project "
        "facts, confirmed approaches, user preferences, and hard-won lessons. "
        "One idea per entry; say why it matters."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": list(VALID_KINDS), "description": "Category (default note)"},
            "tags": {"type": "string", "description": "Optional space/comma separated tags"},
        },
        "required": ["content"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            mid = ctx.memory.add(
                args["content"], kind=args.get("kind", "note"), tags=args.get("tags", "")
            )
        except ValueError as e:
            return ToolResult.error(str(e))
        return ToolResult.ok(f"Remembered as #{mid}.")


class Recall(Tool):
    name = "recall"
    description = "Search persistent memory by keywords. Returns the most relevant entries."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "Max results (default 8)"},
        },
        "required": ["query"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        recs = ctx.memory.search(args["query"], limit=int(args.get("limit", 8)))
        if not recs:
            return ToolResult.ok("No matching memories.")
        return ToolResult.ok("\n".join(r.render() for r in recs))


class Forget(Tool):
    name = "forget"
    description = "Delete a memory by its id (e.g. after it turns out to be wrong or stale)."
    mutating = True
    input_schema = {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ok = ctx.memory.forget(int(args["id"]))
        return ToolResult.ok(f"Forgot #{args['id']}." if ok else f"No memory #{args['id']}.")


MEMORY_TOOLS = [Remember(), Recall(), Forget()]
