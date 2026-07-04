"""Configuration and provider resolution.

JARVIS is provider-agnostic. It runs on any OpenAI-compatible endpoint — Groq,
NVIDIA NIM, Cerebras, OpenRouter, Mistral, or a local Ollama — as well as the
Anthropic API. The default path is **free**: it auto-detects whichever free
provider key you have in your environment, or falls back to a local Ollama.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-4-8"          # only used on the anthropic backend
FALLBACK_MODEL = "claude-opus-4-8"         # refusal fallback target for Fable/Mythos
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


# -- provider presets -------------------------------------------------------


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    kind: str                 # "openai" | "anthropic"
    base_url: str | None
    env_key: str | None       # env var holding the API key (None = keyless)
    default_model: str
    label: str
    free: bool


PROVIDERS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        "groq", "openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
        "llama-3.3-70b-versatile", "Groq (free tier)", True,
    ),
    "nvidia": ProviderPreset(
        "nvidia", "openai", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
        "meta/llama-3.3-70b-instruct", "NVIDIA NIM (free)", True,
    ),
    "cerebras": ProviderPreset(
        "cerebras", "openai", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
        "llama-3.3-70b", "Cerebras (free tier)", True,
    ),
    "openrouter": ProviderPreset(
        "openrouter", "openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free", "OpenRouter (free models)", True,
    ),
    "mistral": ProviderPreset(
        "mistral", "openai", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
        "mistral-large-latest", "Mistral (free tier)", True,
    ),
    "gemini": ProviderPreset(
        "gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY", "gemini-2.0-flash", "Google Gemini (free tier)", True,
    ),
    "together": ProviderPreset(
        "together", "openai", "https://api.together.xyz/v1", "TOGETHER_API_KEY",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo", "Together AI", False,
    ),
    "ollama": ProviderPreset(
        "ollama", "openai", "http://localhost:11434/v1", None,
        "llama3.1", "Ollama (local, free)", True,
    ),
    "openai": ProviderPreset(
        "openai", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY",
        "gpt-4o-mini", "OpenAI", False,
    ),
    "anthropic": ProviderPreset(
        "anthropic", "anthropic", None, "ANTHROPIC_API_KEY",
        DEFAULT_MODEL, "Anthropic (Claude)", False,
    ),
}

# Auto-detection only ever selects a FREE provider, so a stray paid key
# (OPENAI_API_KEY, etc.) can never silently route you to a billable service.
DETECT_ORDER = ["groq", "cerebras", "nvidia", "gemini", "openrouter", "mistral"]


def detect_provider() -> str:
    """Pick a free provider from the environment: first free key wins, else local Ollama."""
    env = os.environ.get("JARVIS_PROVIDER")
    if env:
        return env
    for name in DETECT_ORDER:
        preset = PROVIDERS[name]
        if preset.env_key and os.environ.get(preset.env_key):
            return name
    return "ollama"  # keyless local default — always free


# -- Anthropic-only capability resolution -----------------------------------


@dataclass(frozen=True)
class ModelCaps:
    supports_thinking: bool
    supports_effort: bool
    supports_web_tools: bool
    is_fable_family: bool
    max_effort: str


def model_caps(model: str) -> ModelCaps:
    """Anthropic model capabilities (only consulted on the anthropic backend)."""
    m = model.lower()
    is_fable = m.startswith("claude-fable-") or m.startswith("claude-mythos-")
    is_opus_46plus = any(
        m.startswith(p) for p in ("claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8")
    )
    is_opus_45 = m.startswith("claude-opus-4-5")
    is_sonnet_5 = m.startswith("claude-sonnet-5")
    is_sonnet_46 = m.startswith("claude-sonnet-4-6")
    is_haiku = "haiku" in m
    supports = is_fable or is_opus_46plus or is_opus_45 or is_sonnet_5 or is_sonnet_46
    if is_fable or m.startswith("claude-opus-4-7") or m.startswith("claude-opus-4-8") or is_sonnet_5:
        max_effort = "max"
    elif is_opus_46plus or is_sonnet_46 or is_opus_45:
        max_effort = "max"
    else:
        max_effort = "high"
    return ModelCaps(
        supports_thinking=supports,
        supports_effort=supports,
        supports_web_tools=not is_haiku,
        is_fable_family=is_fable,
        max_effort=max_effort,
    )


def clamp_effort(effort: str, caps: ModelCaps) -> str:
    if effort not in EFFORT_LEVELS:
        effort = "high"
    if EFFORT_LEVELS.index(effort) > EFFORT_LEVELS.index(caps.max_effort):
        return caps.max_effort
    return effort


# -- config -----------------------------------------------------------------


@dataclass
class Config:
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    effort: str = "high"
    max_tokens: int = 4096
    workspace: Path = field(default_factory=Path.cwd)
    state_dir: Path | None = None
    autonomous: bool = False
    enable_web: bool = True
    enable_subagents: bool = True
    max_iterations: int = 50
    subagent_max_depth: int = 2
    allow_paid: bool = False  # free-only by default; paid backends need an explicit opt-in
    verbose: bool = False

    def __post_init__(self) -> None:
        if "JARVIS_WORKSPACE" in os.environ and self.workspace == Path.cwd():
            self.workspace = Path(os.environ["JARVIS_WORKSPACE"])
        self.workspace = Path(self.workspace).expanduser().resolve()
        if self.state_dir is None:
            self.state_dir = self.workspace / ".jarvis"
        self.state_dir = Path(self.state_dir).expanduser().resolve()

        # Resolve the provider and its preset.
        if self.provider is None:
            self.provider = detect_provider()
        self.preset = PROVIDERS.get(self.provider)
        if self.preset is None:
            # Unknown name → generic OpenAI-compatible endpoint (needs --base-url).
            self.preset = ProviderPreset(
                self.provider, "openai", self.base_url,
                self.api_key_env or "OPENAI_API_KEY", "custom", "Custom endpoint", False,
            )

        self.kind = self.preset.kind
        if self.model is None:
            self.model = os.environ.get("JARVIS_MODEL") or self.preset.default_model
        if self.base_url is None:
            self.base_url = os.environ.get("JARVIS_BASE_URL") or self.preset.base_url
        if self.api_key_env is None:
            self.api_key_env = self.preset.env_key
        self.api_key: str | None = (
            os.environ.get("JARVIS_API_KEY")
            or (os.environ.get(self.api_key_env) if self.api_key_env else None)
        )

        # Anthropic capabilities (unused by the openai backend).
        if self.kind == "anthropic":
            self.caps = model_caps(self.model)
            self.effort = clamp_effort(self.effort, self.caps)
        else:
            self.caps = ModelCaps(False, False, False, False, "high")

    @property
    def memory_path(self) -> Path:
        return self.state_dir / "memory.db"

    @property
    def label(self) -> str:
        return self.preset.label if self.preset else self.provider

    @property
    def is_free_provider(self) -> bool:
        return bool(self.preset and self.preset.free)

    def paid_and_not_allowed(self) -> bool:
        """True if this backend could cost money and the user hasn't opted in."""
        return not self.is_free_provider and not self.allow_paid

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def needs_key(self) -> bool:
        return bool(self.api_key_env) and self.kind != "anthropic"

    def has_credentials(self) -> bool:
        """Keyless providers (Ollama) always pass; others need a key present."""
        if self.provider == "ollama":
            return True
        if self.kind == "anthropic":
            return bool(self.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        return bool(self.api_key)
