"""Pluggable model backends.

A :class:`Backend` owns its own native message history and exposes a small,
provider-agnostic surface to the agent: configure once, add user/tool messages,
and run a turn that yields an :class:`AssistantTurn`.
"""
from __future__ import annotations

from ..config import Config
from .base import (
    AssistantTurn,
    Backend,
    ProviderAuthError,
    ProviderError,
    StreamCallbacks,
    ToolCall,
)

__all__ = [
    "AssistantTurn",
    "Backend",
    "ToolCall",
    "StreamCallbacks",
    "ProviderError",
    "ProviderAuthError",
    "make_backend",
]


def make_backend(config: Config) -> Backend:
    if config.kind == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(config)
    from .openai_compat import OpenAIBackend

    return OpenAIBackend(config)
