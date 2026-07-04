"""The agentic loop — provider-agnostic.

An :class:`Agent` drives a :class:`Backend` (any provider) through a plan → act →
verify loop: run a turn, render thinking + text live, execute the tools the model
asked for, feed the results back, and repeat until the model is done. It can
delegate self-contained subtasks to sub-agents.
"""
from __future__ import annotations

from typing import Callable

from .config import Config
from .memory import Memory
from .prompts import SUBAGENT_PROMPT, system_prompt
from .providers import AssistantTurn, Backend, StreamCallbacks, make_backend
from .providers.base import ToolResultMsg
from .tools import PathError, ToolContext, ToolRegistry, build_tools
from .ui import Console


class Agent:
    def __init__(
        self,
        config: Config,
        backend: Backend,
        memory: Memory,
        console: Console,
        *,
        depth: int = 0,
        backend_factory: Callable[[], Backend] | None = None,
    ):
        self.config = config
        self.backend = backend
        self.memory = memory
        self.console = console
        self.depth = depth
        self.backend_factory = backend_factory or (lambda: make_backend(config))
        include_sub = depth + 1 < config.subagent_max_depth
        self.registry = ToolRegistry(build_tools(config, include_subagents=include_sub))
        self.state: dict = {}
        self._configured = False

    # -- setup --------------------------------------------------------------

    def _configure(self, *, memory_digest: str = None) -> None:
        digest = self.memory.digest() if memory_digest is None else memory_digest
        system = system_prompt(
            workspace=self.config.workspace,
            autonomous=self.config.autonomous,
            tools_overview=self.registry.overview(),
            memory_digest=digest,
        )
        self.backend.configure(system, self.registry.schemas())
        self._configured = True

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

    # -- entry points -------------------------------------------------------

    def run(self, objective: str) -> str:
        if not self._configured:
            self._configure()
        self.backend.reset()
        self.backend.add_user_message(objective)
        return self._loop()

    def send(self, user_message: str) -> str:
        if not self._configured:
            self._configure()
        self.backend.add_user_message(user_message)
        return self._loop()

    def reset(self) -> None:
        self.backend.reset()
        self.state = {}

    # -- the loop -----------------------------------------------------------

    def _loop(self) -> str:
        callbacks = StreamCallbacks(
            on_thinking=self.console.stream_thinking,
            on_text=self.console.stream_text,
        )
        final_text = ""
        for _ in range(self.config.max_iterations):
            turn = self.backend.run(callbacks)
            self.console.end_stream()
            if turn.text:
                final_text = turn.text

            if turn.stop_reason == "refusal":
                cat = f" (category: {turn.refusal_category})" if turn.refusal_category else ""
                self.console.error(f"The model refused this request.{cat}")
                return final_text or "(request refused)"

            if turn.tool_calls:
                results = self._execute_tools(turn)
                self.backend.add_tool_results(results)
                continue

            if turn.stop_reason == "pause_turn":
                continue  # server-side tool (Anthropic) — resume

            if turn.stop_reason == "max_tokens":
                self.backend.add_user_message("Your response was cut off — continue where you stopped.")
                continue

            break
        else:
            self.console.warn(f"Reached the {self.config.max_iterations}-iteration safety cap.")
        return final_text

    def _execute_tools(self, turn: AssistantTurn) -> list[ToolResultMsg]:
        ctx = self._ctx()
        results: list[ToolResultMsg] = []
        for call in turn.tool_calls:
            self.console.tool_call(call.name, call.args)
            tool = self.registry.get(call.name)
            if tool is None:
                content, is_error = f"Unknown tool: {call.name}", True
            else:
                try:
                    res = tool.run(call.args, ctx)
                    content, is_error = res.content, res.is_error
                except PathError as e:
                    content, is_error = str(e), True
                except Exception as e:  # noqa: BLE001 — a tool bug must not crash the loop
                    content, is_error = f"Tool '{call.name}' raised: {e}", True
            self.console.tool_result(call.name, content, is_error)
            results.append(ToolResultMsg(id=call.id, name=call.name, content=content, is_error=is_error))
        return results

    # -- delegation ---------------------------------------------------------

    def _delegate(self, objective: str, extra_context: str) -> str:
        self.console.rule(f"sub-agent (depth {self.depth + 1})")
        sub = Agent(
            self.config,
            self.backend_factory(),
            self.memory,
            self.console,
            depth=self.depth + 1,
            backend_factory=self.backend_factory,
        )
        sub._configure(memory_digest="")
        first = SUBAGENT_PROMPT.format(workspace=self.config.workspace, objective=objective)
        if extra_context.strip():
            first += f"\n\nContext from the orchestrator:\n{extra_context}"
        report = sub.run(first)
        self.console.rule("resume")
        return report or "(sub-agent produced no report)"
