"""Configuration and per-model capability resolution.

The default model is ``claude-opus-4-8``. Set ``--model claude-fable-5`` (or the
``JARVIS_MODEL`` env var) to run on Anthropic's most capable model; JARVIS adapts
the request shape automatically (see :func:`model_caps`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-4-8"
FALLBACK_MODEL = "claude-opus-4-8"  # refusal fallback target for Fable/Mythos

# Effort levels understood by the API, ordered.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ModelCaps:
    """What a given model supports, so we build a valid request every time."""

    supports_thinking: bool
    supports_effort: bool
    supports_web_tools: bool
    is_fable_family: bool  # Fable 5 / Mythos 5 — always-on thinking + refusal path
    max_effort: str  # highest effort level this model accepts


def model_caps(model: str) -> ModelCaps:
    """Resolve capabilities from a model id without a network round-trip.

    Kept deliberately conservative: when in doubt we disable a feature rather
    than send a request the API will 400.
    """
    m = model.lower()
    is_fable = m.startswith("claude-fable-") or m.startswith("claude-mythos-")
    is_opus_46plus = any(
        m.startswith(p)
        for p in ("claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8")
    )
    is_opus_45 = m.startswith("claude-opus-4-5")
    is_sonnet_5 = m.startswith("claude-sonnet-5")
    is_sonnet_46 = m.startswith("claude-sonnet-4-6")
    is_haiku = "haiku" in m

    supports_thinking = is_fable or is_opus_46plus or is_opus_45 or is_sonnet_5 or is_sonnet_46
    supports_effort = supports_thinking  # same generation gate in practice
    supports_web = not is_haiku  # server web tools available on current mid/large models

    # `xhigh` exists on Fable/Opus 4.7+/Sonnet 5; `max` on 4.6+/Sonnet families.
    if is_fable or m.startswith("claude-opus-4-7") or m.startswith("claude-opus-4-8") or is_sonnet_5:
        max_effort = "max"
    elif is_opus_46plus or is_sonnet_46 or is_opus_45:
        max_effort = "max"
    else:
        max_effort = "high"

    return ModelCaps(
        supports_thinking=supports_thinking,
        supports_effort=supports_effort,
        supports_web_tools=supports_web,
        is_fable_family=is_fable,
        max_effort=max_effort,
    )


def clamp_effort(effort: str, caps: ModelCaps) -> str:
    """Clamp a requested effort to what the model accepts."""
    if effort not in EFFORT_LEVELS:
        effort = "high"
    if EFFORT_LEVELS.index(effort) > EFFORT_LEVELS.index(caps.max_effort):
        return caps.max_effort
    return effort


@dataclass
class Config:
    """Everything the agent needs to know before it starts."""

    model: str = DEFAULT_MODEL
    effort: str = "high"
    max_tokens: int = 32000
    workspace: Path = field(default_factory=Path.cwd)
    state_dir: Path = field(default_factory=lambda: Path.cwd() / ".jarvis")
    autonomous: bool = False
    enable_web: bool = True
    enable_subagents: bool = True
    max_iterations: int = 50
    subagent_effort: str = "medium"
    subagent_max_depth: int = 2
    verbose: bool = False

    def __post_init__(self) -> None:
        # Environment overrides (flags take precedence — the CLI applies those after).
        self.model = os.environ.get("JARVIS_MODEL", self.model)
        self.effort = os.environ.get("JARVIS_EFFORT", self.effort)
        if "JARVIS_WORKSPACE" in os.environ:
            self.workspace = Path(os.environ["JARVIS_WORKSPACE"])
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.state_dir = Path(self.state_dir).expanduser().resolve()
        self.caps = model_caps(self.model)
        self.effort = clamp_effort(self.effort, self.caps)

    @property
    def memory_path(self) -> Path:
        return self.state_dir / "memory.db"

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def has_api_credentials(self) -> bool:
        """True if an API key is present in the environment.

        Absence does NOT mean there are no credentials — an ``ant auth login``
        profile also works. We only use this to print a friendlier hint.
        """
        return bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
