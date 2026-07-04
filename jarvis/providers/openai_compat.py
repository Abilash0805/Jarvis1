"""OpenAI-compatible backend.

Works with any provider that speaks the OpenAI Chat Completions API: Groq,
NVIDIA NIM, Cerebras, OpenRouter, Mistral, Together, a local Ollama, LM Studio,
llama.cpp — and OpenAI itself. Handles streaming, native tool calling, and
reasoning output (both a dedicated ``reasoning_content`` field and inline
``<think>`` tags).
"""
from __future__ import annotations

import json
import urllib.request
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

# Where to get a free key, surfaced in the health-check message.
KEY_HELP = {
    "groq": "https://console.groq.com/keys",
    "nvidia": "https://build.nvidia.com (free credits)",
    "cerebras": "https://cloud.cerebras.ai",
    "openrouter": "https://openrouter.ai/keys",
    "mistral": "https://console.mistral.ai/api-keys",
    "gemini": "https://aistudio.google.com/apikey",
    "together": "https://api.together.xyz/settings/api-keys",
    "openai": "https://platform.openai.com/api-keys",
}


def _check_ollama(base_url: str) -> str | None:
    """Return install guidance if a local Ollama isn't reachable, else None."""
    root = base_url.rsplit("/v1", 1)[0]
    try:
        urllib.request.urlopen(root + "/api/tags", timeout=2)
        return None
    except Exception:  # noqa: BLE001 — any failure means "not ready"
        return (
            f"Ollama isn't reachable at {root}. It's free — install from "
            "https://ollama.com, then run `ollama serve` and `ollama pull llama3.1`."
        )


class ThinkRouter:
    """Split a content stream into visible text vs. ``<think>...</think>`` reasoning.

    Streams each part to the right callback while accumulating the visible text
    (with the think sections removed) for the message history.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, cb: StreamCallbacks):
        self.cb = cb
        self.mode = "text"
        self.pending = ""
        self._visible: list[str] = []

    def feed(self, delta: str) -> None:
        self.pending += delta
        self._drain(final=False)

    def close(self) -> None:
        self._drain(final=True)

    def visible_text(self) -> str:
        return "".join(self._visible)

    def _emit(self, s: str) -> None:
        if not s:
            return
        if self.mode == "text":
            self._visible.append(s)
            self.cb.text(s)
        else:
            self.cb.thinking(s)

    def _drain(self, final: bool) -> None:
        while True:
            marker = self.CLOSE if self.mode == "thinking" else self.OPEN
            idx = self.pending.find(marker)
            if idx != -1:
                self._emit(self.pending[:idx])
                self.pending = self.pending[idx + len(marker):]
                self.mode = "thinking" if marker == self.OPEN else "text"
                continue
            if final:
                self._emit(self.pending)
                self.pending = ""
                return
            hold = _partial_marker_len(self.pending, (self.OPEN, self.CLOSE))
            cut = len(self.pending) - hold
            self._emit(self.pending[:cut])
            self.pending = self.pending[cut:]
            return


def _partial_marker_len(pending: str, markers: tuple[str, ...]) -> int:
    """Longest suffix of `pending` that is a proper prefix of any marker."""
    best = 0
    for m in markers:
        limit = min(len(pending), len(m) - 1)
        for k in range(limit, 0, -1):
            if pending[-k:] == m[:k]:
                best = max(best, k)
                break
    return best


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


class OpenAIBackend(Backend):
    def __init__(self, config: Config, client: Any = None):
        self.config = config
        self.model = config.model
        self.label = config.label
        self._system = ""
        self._oa_tools: list[dict] = []
        self.messages: list[dict[str, Any]] = []
        self._client = client
        if client is None:
            self._client = self._build_client()

    def _build_client(self) -> Any:
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ProviderError(
                "The 'openai' package is required for this provider. Install it with:\n"
                "  pip install openai"
            ) from e
        # Keyless providers (Ollama) still need a non-empty api_key placeholder.
        api_key = self.config.api_key or "not-needed"
        return openai.OpenAI(base_url=self.config.base_url, api_key=api_key)

    # -- Backend interface --------------------------------------------------

    def configure(self, system: str, tools: list[dict]) -> None:
        self._system = system
        self._oa_tools = _to_openai_tools(tools)

    def reset(self) -> None:
        self.messages = []

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[ToolResultMsg]) -> None:
        for r in results:
            self.messages.append(
                {"role": "tool", "tool_call_id": r.id, "content": r.content}
            )

    def health_check(self) -> str | None:
        if self.config.provider == "ollama":
            return _check_ollama(self.config.base_url or "")
        if self.config.needs_key() and not self.config.api_key:
            where = KEY_HELP.get(self.config.provider, "the provider's dashboard")
            return (
                f"No API key for {self.label}. Set {self.config.api_key_env} "
                f"(free key: {where})."
            )
        return None

    def run(self, callbacks: StreamCallbacks) -> AssistantTurn:
        try:
            return self._run_stream(callbacks)
        except (ProviderAuthError, ProviderError):
            raise
        except Exception:  # noqa: BLE001 — streaming may be unsupported; try once plain
            return self._run_nonstream(callbacks)

    # -- request assembly ---------------------------------------------------

    def _kwargs(self, *, stream: bool) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": self._system}] + self.messages,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        if self._oa_tools:
            kw["tools"] = self._oa_tools
            kw["tool_choice"] = "auto"
        return kw

    def _create(self, *, stream: bool) -> Any:
        try:
            return self._client.chat.completions.create(**self._kwargs(stream=stream))
        except Exception as e:  # noqa: BLE001
            raise self._wrap(e)

    def _wrap(self, e: Exception) -> ProviderError:
        name = type(e).__name__
        msg = str(e)
        if "Authentication" in name or "PermissionDenied" in name or "401" in msg:
            where = KEY_HELP.get(self.config.provider, "the provider's dashboard")
            return ProviderAuthError(
                f"{self.label} rejected the credentials. Check {self.config.api_key_env} "
                f"(free key: {where})."
            )
        if "Connection" in name:
            hint = (
                " Is `ollama serve` running?"
                if self.config.provider == "ollama"
                else ""
            )
            return ProviderError(f"Could not reach {self.label} at {self.config.base_url}.{hint}")
        if "NotFound" in name or "404" in msg:
            return ProviderError(f"Model '{self.model}' not found on {self.label}. Check --model.")
        return ProviderError(f"{self.label} request failed: {msg}")

    # -- streaming ----------------------------------------------------------

    def _run_stream(self, cb: StreamCallbacks) -> AssistantTurn:
        stream = self._create(stream=True)
        router = ThinkRouter(cb)
        think_parts: list[str] = []
        slots: dict[int, dict[str, Any]] = {}
        finish: str | None = None

        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                router.feed(content)
            rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if rc:
                think_parts.append(rc)
                cb.thinking(rc)
            for tc in getattr(delta, "tool_calls", None) or []:
                self._accumulate_tool_call(tc, slots, cb)

        router.close()
        return self._finish(router.visible_text(), "".join(think_parts), slots, finish)

    def _accumulate_tool_call(self, tc: Any, slots: dict[int, dict], cb: StreamCallbacks) -> None:
        idx = getattr(tc, "index", 0) or 0
        slot = slots.setdefault(idx, {"id": None, "name": None, "args": ""})
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["name"] = fn.name
                cb.tool_start(fn.name)
            if getattr(fn, "arguments", None):
                slot["args"] += fn.arguments

    # -- non-streaming fallback --------------------------------------------

    def _run_nonstream(self, cb: StreamCallbacks) -> AssistantTurn:
        resp = self._create(stream=False)
        message = resp.choices[0].message
        finish = resp.choices[0].finish_reason
        text = getattr(message, "content", None) or ""
        if text:
            router = ThinkRouter(cb)
            router.feed(text)
            router.close()
            text = router.visible_text()
        thinking = getattr(message, "reasoning_content", None) or ""
        if thinking:
            cb.thinking(thinking)
        slots: dict[int, dict[str, Any]] = {}
        for i, tc in enumerate(getattr(message, "tool_calls", None) or []):
            fn = tc.function
            slots[i] = {"id": getattr(tc, "id", None), "name": fn.name, "args": fn.arguments or ""}
            cb.tool_start(fn.name)
        return self._finish(text, thinking, slots, finish)

    # -- shared finalization ------------------------------------------------

    def _finish(
        self, text: str, thinking: str, slots: dict[int, dict], finish: str | None
    ) -> AssistantTurn:
        tool_calls: list[ToolCall] = []
        raw_calls: list[dict] = []
        for idx in sorted(slots):
            slot = slots[idx]
            cid = slot["id"] or f"call_{idx}"
            name = slot["name"] or "unknown"
            args_str = slot["args"] or "{}"
            try:
                args = json.loads(args_str) if args_str.strip() else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=cid, name=name, args=args))
            raw_calls.append(
                {"id": cid, "type": "function", "function": {"name": name, "arguments": args_str}}
            )

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if raw_calls:
            assistant_msg["tool_calls"] = raw_calls
        self.messages.append(assistant_msg)

        if tool_calls:
            stop = "tool_use"
        elif finish == "length":
            stop = "max_tokens"
        else:
            stop = "end_turn"
        return AssistantTurn(text=text, thinking=thinking, tool_calls=tool_calls, stop_reason=stop)
