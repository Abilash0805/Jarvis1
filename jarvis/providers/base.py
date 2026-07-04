"""Provider-neutral types and the Backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


class ProviderError(Exception):
    """A backend could not complete a request (connection, config, server)."""


class ProviderAuthError(ProviderError):
    """The backend needs credentials that are missing or invalid."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class AssistantTurn:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Normalized: "end_turn" | "tool_use" | "max_tokens" | "refusal" | "pause_turn"
    stop_reason: str = "end_turn"
    refusal_category: str | None = None


@dataclass
class StreamCallbacks:
    """Live-rendering hooks; all optional."""

    on_thinking: Callable[[str], None] | None = None
    on_text: Callable[[str], None] | None = None
    on_tool_start: Callable[[str], None] | None = None

    def thinking(self, s: str) -> None:
        if self.on_thinking and s:
            self.on_thinking(s)

    def text(self, s: str) -> None:
        if self.on_text and s:
            self.on_text(s)

    def tool_start(self, name: str) -> None:
        if self.on_tool_start and name:
            self.on_tool_start(name)


@dataclass
class ToolResultMsg:
    id: str
    name: str
    content: str
    is_error: bool


class Backend(ABC):
    """A stateful conversation with one model provider.

    The backend keeps its own native message history (so provider-specific
    details like Anthropic thinking-block signatures are preserved) and exposes
    a neutral surface: configure → add messages → run.
    """

    label: str = "backend"
    model: str = ""

    @abstractmethod
    def configure(self, system: str, tools: list[dict]) -> None:
        """Set the system prompt and the (Anthropic-shaped) custom tool schemas."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the conversation history (keeps system + tools)."""

    @abstractmethod
    def add_user_message(self, text: str) -> None: ...

    @abstractmethod
    def add_tool_results(self, results: list[ToolResultMsg]) -> None: ...

    @abstractmethod
    def run(self, callbacks: StreamCallbacks) -> AssistantTurn:
        """Make one model call; append the assistant message to native history."""

    def health_check(self) -> str | None:
        """Return a human-readable problem string, or None if the backend looks ready."""
        return None
