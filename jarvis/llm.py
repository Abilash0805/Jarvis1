"""Anthropic client wrapper.

Centralizes every model-shape decision so the agent loop never has to think
about thinking config, effort, streaming, or Fable-5 refusal fallbacks. One
method — :meth:`LLMClient.complete` — streams a turn and returns the final
message, invoking optional callbacks for live rendering.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import anthropic

from .config import FALLBACK_MODEL, Config

# Beta flag that enables the server-side `fallbacks` parameter (Fable/Mythos).
SERVER_FALLBACK_BETA = "server-side-fallback-2026-06-01"


@dataclass
class StreamCallbacks:
    """Live-rendering hooks. All optional; default to no-ops."""

    on_thinking: Callable[[str], None] | None = None
    on_text: Callable[[str], None] | None = None
    on_tool_start: Callable[[str], None] | None = None  # tool/server-tool name

    def thinking(self, s: str) -> None:
        if self.on_thinking:
            self.on_thinking(s)

    def text(self, s: str) -> None:
        if self.on_text:
            self.on_text(s)

    def tool_start(self, name: str) -> None:
        if self.on_tool_start:
            self.on_tool_start(name)


class LLMClient:
    """Thin, resilient wrapper over ``anthropic.Anthropic``."""

    def __init__(self, config: Config, client: anthropic.Anthropic | None = None):
        self.config = config
        self.caps = config.caps
        self.client = client or anthropic.Anthropic()

    # -- request assembly ---------------------------------------------------

    def _base_kwargs(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
        max_tokens: int,
        effort: str,
    ) -> dict[str, Any]:
        caps = self.caps
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if caps.supports_thinking:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if caps.supports_effort:
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: Sequence[dict[str, Any]] | None = None,
        callbacks: StreamCallbacks | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> anthropic.types.Message:
        """Stream one assistant turn and return the final :class:`Message`.

        Streaming (rather than a blocking create) keeps long, high-``max_tokens``
        turns from tripping HTTP timeouts and lets us render thinking + text live.
        """
        cb = callbacks or StreamCallbacks()
        kwargs = self._base_kwargs(
            model=self.config.model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens or self.config.max_tokens,
            effort=effort or self.config.effort,
        )
        return self._stream_with_retry(kwargs, cb)

    # -- internals ----------------------------------------------------------

    def _stream_context(self, kwargs: dict[str, Any]):
        """Pick the right streaming endpoint; add Fable-5 refusal fallbacks."""
        if self.caps.is_fable_family:
            return self.client.beta.messages.stream(
                **kwargs,
                betas=[SERVER_FALLBACK_BETA],
                fallbacks=[{"model": FALLBACK_MODEL}],
            )
        return self.client.messages.stream(**kwargs)

    def _run_stream(self, kwargs: dict[str, Any], cb: StreamCallbacks):
        with self._stream_context(kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = getattr(event, "content_block", None)
                    btype = getattr(block, "type", None)
                    if btype in ("tool_use", "server_tool_use", "mcp_tool_use"):
                        cb.tool_start(getattr(block, "name", "") or "")
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", None)
                    if dtype == "thinking_delta":
                        cb.thinking(getattr(delta, "thinking", "") or "")
                    elif dtype == "text_delta":
                        cb.text(getattr(delta, "text", "") or "")
            return stream.get_final_message()

    def _stream_with_retry(
        self, kwargs: dict[str, Any], cb: StreamCallbacks, _attempt: int = 0
    ) -> anthropic.types.Message:
        try:
            return self._run_stream(kwargs, cb)
        except anthropic.BadRequestError as exc:
            # Degrade gracefully if this model/SDK rejects an optional param
            # (thinking / effort). Strip and retry once so we still get an answer.
            stripped = _strip_unsupported(kwargs, str(exc))
            if stripped is not None and _attempt == 0:
                return self._stream_with_retry(stripped, cb, _attempt + 1)
            raise
        except (anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
            if _attempt < 4:
                time.sleep(2 ** _attempt)
                return self._stream_with_retry(kwargs, cb, _attempt + 1)
            raise RuntimeError(f"LLM request failed after retries: {exc}") from exc


def _strip_unsupported(kwargs: dict[str, Any], message: str) -> dict[str, Any] | None:
    """Return a copy of kwargs with an offending optional param removed, or None."""
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
