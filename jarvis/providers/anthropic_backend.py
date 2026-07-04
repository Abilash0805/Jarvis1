"""Anthropic (Claude) backend.

Optional — only used when ``--provider anthropic``. Preserves native content
blocks (including thinking-block signatures) in history for correct multi-turn
replay, and enables server-side refusal fallbacks for the Fable/Mythos family.
"""
from __future__ import annotations

import time
from typing import Any

from ..config import Config
from .base import (
    AssistantTurn,
    Backend,
    ProviderAuthError,
    ProviderError,
    StreamCallbacks,
    ToolCall,
    ToolResultMsg,
)

SERVER_FALLBACK_BETA = "server-side-fallback-2026-06-01"


class AnthropicBackend(Backend):
    def __init__(self, config: Config, client: Any = None):
        self.config = config
        self.caps = config.caps
        self.model = config.model
        self.label = config.label
        self._system = ""
        self._tools: list[dict] = []
        self.messages: list[dict[str, Any]] = []
        self._client = client or self._build_client()

    def _build_client(self) -> Any:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "The 'anthropic' package is required for --provider anthropic. Install it with:\n"
                "  pip install anthropic"
            ) from e
        return anthropic.Anthropic()

    # -- Backend interface --------------------------------------------------

    def configure(self, system: str, tools: list[dict]) -> None:
        self._system = system
        self._tools = list(tools)  # already {name, description, input_schema}

    def reset(self) -> None:
        self.messages = []

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResultMsg]) -> None:
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r.id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        self.messages.append({"role": "user", "content": blocks})

    def health_check(self) -> str | None:
        return None  # the SDK also resolves `ant auth login` profiles; check at call time

    # -- request assembly ---------------------------------------------------

    def _kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(self.config.max_tokens, 16000),
            "system": self._system,
            "messages": self.messages,
        }
        if self._tools:
            kw["tools"] = self._tools
        if self.caps.supports_thinking:
            kw["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.caps.supports_effort:
            kw["output_config"] = {"effort": self.config.effort}
        return kw

    def run(self, callbacks: StreamCallbacks) -> AssistantTurn:
        msg = self._stream_with_retry(self._kwargs(), callbacks)
        self.messages.append({"role": "assistant", "content": msg.content})
        return self._to_turn(msg)

    # -- streaming ----------------------------------------------------------

    def _stream_context(self, kwargs: dict[str, Any]):
        if self.caps.is_fable_family:
            return self._client.beta.messages.stream(
                **kwargs,
                betas=[SERVER_FALLBACK_BETA],
                fallbacks=[{"model": "claude-opus-4-8"}],
            )
        return self._client.messages.stream(**kwargs)

    def _run_stream(self, kwargs: dict[str, Any], cb: StreamCallbacks):
        with self._stream_context(kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) in ("tool_use", "server_tool_use"):
                        cb.tool_start(getattr(block, "name", "") or "")
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", None)
                    if dtype == "thinking_delta":
                        cb.thinking(getattr(delta, "thinking", "") or "")
                    elif dtype == "text_delta":
                        cb.text(getattr(delta, "text", "") or "")
            return stream.get_final_message()

    def _stream_with_retry(self, kwargs: dict[str, Any], cb: StreamCallbacks, attempt: int = 0):
        import anthropic

        try:
            return self._run_stream(kwargs, cb)
        except anthropic.AuthenticationError as e:
            raise ProviderAuthError("Anthropic authentication failed. Set ANTHROPIC_API_KEY.") from e
        except anthropic.BadRequestError as e:
            stripped = _strip_unsupported(kwargs, str(e))
            if stripped is not None and attempt == 0:
                return self._stream_with_retry(stripped, cb, attempt + 1)
            raise ProviderError(f"Anthropic rejected the request: {e}") from e
        except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            if attempt < 4:
                time.sleep(2 ** attempt)
                return self._stream_with_retry(kwargs, cb, attempt + 1)
            raise ProviderError(f"Anthropic request failed after retries: {e}") from e

    def _to_turn(self, msg: Any) -> AssistantTurn:
        text = "\n".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
        thinking = "\n".join(
            getattr(b, "thinking", "") for b in msg.content if getattr(b, "type", None) == "thinking"
        )
        tool_calls = [
            ToolCall(id=b.id, name=b.name, args=b.input or {})
            for b in msg.content
            if getattr(b, "type", None) == "tool_use"
        ]
        stop = msg.stop_reason or "end_turn"
        category = None
        details = getattr(msg, "stop_details", None)
        if details is not None:
            category = getattr(details, "category", None)
        return AssistantTurn(
            text=text, thinking=thinking, tool_calls=tool_calls,
            stop_reason=stop, refusal_category=category,
        )


def _strip_unsupported(kwargs: dict[str, Any], message: str) -> dict[str, Any] | None:
    msg = message.lower()
    out = dict(kwargs)
    changed = False
    if "thinking" in msg and "thinking" in out:
        out.pop("thinking")
        changed = True
    if ("effort" in msg or "output_config" in msg) and "output_config" in out:
        out.pop("output_config")
        changed = True
    return out if changed else None
