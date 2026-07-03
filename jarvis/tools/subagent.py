"""Sub-agent delegation.

Lets the orchestrator hand a self-contained objective to a focused sub-agent
that runs to completion and reports back. Depth is bounded to prevent runaway
recursion; the actual spawn is provided by the Agent via ``ctx.delegate``.
"""
from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolResult


class Delegate(Tool):
    name = "delegate"
    description = (
        "Delegate a self-contained subtask to a focused sub-agent that runs "
        "autonomously and returns a report. Use for independent workstreams: "
        "fanning out across files, parallel research, or a contained build. "
        "Give the sub-agent everything it needs — it does not share your context."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": "The complete, self-contained objective for the sub-agent.",
            },
            "context": {
                "type": "string",
                "description": "Relevant facts/context the sub-agent needs (it can't see your conversation).",
            },
        },
        "required": ["objective"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.delegate is None:
            return ToolResult.error("Delegation is not available in this context.")
        if ctx.depth >= ctx.config.subagent_max_depth:
            return ToolResult.error(
                f"Max delegation depth ({ctx.config.subagent_max_depth}) reached; "
                "do this subtask directly."
            )
        objective = args["objective"]
        extra = args.get("context", "")
        try:
            report = ctx.delegate(objective, extra)
        except Exception as e:  # a sub-agent failure shouldn't crash the parent
            return ToolResult.error(f"Sub-agent failed: {e}")
        return ToolResult.ok(f"Sub-agent report:\n{report}")


SUBAGENT_TOOLS = [Delegate()]
