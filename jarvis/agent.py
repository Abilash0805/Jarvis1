"""The agentic loop.

An :class:`Agent` runs an open-ended plan→act→verify loop: it calls the model,
renders thinking and text live, executes the tools the model asks for, feeds the
results back, and repeats until the model is done (or a safety cap trips). It
can delegate self-contained subtasks to sub-agents.
"""
from __future__ import annotations

from typing import Any

from .config import Config
from .llm import LLMClient, StreamCallbacks
from .memory import Memory
from .prompts import SUBAGENT_PROMPT, system_prompt
from .tools import (
    PathError,
    ToolContext,
    ToolRegistry,
    build_tools,
    web_tool_declarations,
)
from .ui import Console


class Agent:
    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        memory: Memory,
        console: Console,
        *,
        depth: int = 0,
    ):
        self.config = config
        self.llm = llm
        self.memory = memory
        self.console = console
        self.depth = depth
        include_sub = depth + 1 < config.subagent_max_depth
        self.registry = ToolRegistry(build_tools(config, include_subagents=include_sub))
        self.messages: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}
        self._system: str | None = None

    # -- context / tools ----------------------------------------------------

    def _ctx(self) -> ToolContext:
        return ToolContext(
            config=self.config,
            memory=self.memory,
            console=self.console,
            workspace=self.config.workspace,
            autonomous=self.config.autonomous,
            confirm=self._confirm,
            delegate=self._delegate if self.config.enable_subagents else None,
            depth=self.depth,
            state=self.state,
        )

    def _confirm(self, action: str) -> bool:
        return True if self.config.autonomous else self.console.confirm(action)

    def _api_tools(self) -> list[dict]:
        tools = self.registry.custom_schemas()
        if self.config.enable_web and self.config.caps.supports_web_tools:
            tools += web_tool_declarations()
        return tools

    def _build_system(self) -> str:
        if self._system is None:
            self._system = system_prompt(
                workspace=self.config.workspace,
                autonomous=self.config.autonomous,
                tools_overview=self.registry.overview(),
                memory_digest=self.memory.digest(),
            )
        return self._system

    # -- public entry points ------------------------------------------------

    def run(self, objective: str) -> str:
        """Fresh objective (one-shot). Returns the final text."""
        self.messages = [{"role": "user", "content": objective}]
        return self._loop(self._build_system())

    def send(self, user_message: str) -> str:
        """Continue an ongoing conversation (REPL)."""
        self.messages.append({"role": "user", "content": user_message})
        return self._loop(self._build_system())

    # -- the loop -----------------------------------------------------------

    def _loop(self, system: str) -> str:
        tools = self._api_tools()
        effort = self.config.subagent_effort if self.depth > 0 else self.config.effort
        max_tokens = 16000 if self.depth > 0 else self.config.max_tokens
        final_text = ""
        callbacks = StreamCallbacks(
            on_thinking=self.console.stream_thinking,
            on_text=self.console.stream_text,
        )

        for _ in range(self.config.max_iterations):
            msg = self.llm.complete(
                messages=self.messages,
                system=system,
                tools=tools,
                callbacks=callbacks,
                effort=effort,
                max_tokens=max_tokens,
            )
            self.console.end_stream()
            self.messages.append({"role": "assistant", "content": msg.content})

            text = _text_of(msg)
            if text:
                final_text = text

            stop = msg.stop_reason
            if stop == "refusal":
                detail = _refusal_detail(msg)
                self.console.error(f"Request refused by safety policy.{detail}")
                return final_text or "(request refused)"

            if stop == "tool_use":
                results = self._execute_tools(msg)
                self.messages.append({"role": "user", "content": results})
                continue

            if stop == "pause_turn":
                # Server-side tool (e.g. web search) hit its per-turn cap; resume.
                continue

            if stop == "max_tokens":
                self.messages.append(
                    {"role": "user", "content": "Your response was cut off. Continue from where you stopped."}
                )
                continue

            # end_turn / stop_sequence
            break
        else:
            self.console.warn(
                f"Reached the {self.config.max_iterations}-iteration safety cap."
            )
        return final_text

    def _execute_tools(self, msg: Any) -> list[dict[str, Any]]:
        ctx = self._ctx()
        results: list[dict[str, Any]] = []
        for block in msg.content:
            if getattr(block, "type", None) != "tool_use":
                continue  # server tools already ran server-side
            name = block.name
            args = block.input or {}
            self.console.tool_call(name, args)
            tool = self.registry.get(name)
            if tool is None:
                content, is_error = f"Unknown tool: {name}", True
            else:
                try:
                    res = tool.run(args, ctx)
                    content, is_error = res.content, res.is_error
                except PathError as e:
                    content, is_error = str(e), True
                except Exception as e:  # a tool bug must not crash the loop
                    content, is_error = f"Tool '{name}' raised: {e}", True
            self.console.tool_result(name, content, is_error)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        return results

    # -- delegation ---------------------------------------------------------

    def _delegate(self, objective: str, extra_context: str) -> str:
        self.console.rule(f"sub-agent (depth {self.depth + 1})")
        sub = Agent(self.config, self.llm, self.memory, self.console, depth=self.depth + 1)
        first = SUBAGENT_PROMPT.format(workspace=self.config.workspace, objective=objective)
        if extra_context.strip():
            first += f"\n\nContext from the orchestrator:\n{extra_context}"
        sub.messages = [{"role": "user", "content": first}]
        report = sub._loop(
            system_prompt(
                workspace=self.config.workspace,
                autonomous=self.config.autonomous,
                tools_overview=sub.registry.overview(),
                memory_digest="",
            )
        )
        self.console.rule("resume")
        return report or "(sub-agent produced no report)"


def _text_of(msg: Any) -> str:
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "\n".join(p for p in parts if p).strip()


def _refusal_detail(msg: Any) -> str:
    details = getattr(msg, "stop_details", None)
    if details is None:
        return ""
    cat = getattr(details, "category", None)
    return f" (category: {cat})" if cat else ""
