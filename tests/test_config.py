from jarvis.config import PROVIDERS, Config, clamp_effort, detect_provider, model_caps


def test_opus_caps():
    caps = model_caps("claude-opus-4-8")
    assert caps.supports_thinking
    assert caps.supports_effort
    assert caps.supports_web_tools
    assert not caps.is_fable_family
    assert caps.max_effort == "max"


def test_fable_caps():
    caps = model_caps("claude-fable-5")
    assert caps.is_fable_family
    assert caps.supports_thinking
    assert caps.max_effort == "max"


def test_haiku_caps():
    caps = model_caps("claude-haiku-4-5")
    assert not caps.supports_thinking
    assert not caps.supports_effort


def test_clamp_effort():
    caps = model_caps("claude-opus-4-8")
    assert clamp_effort("max", caps) == "max"
    assert clamp_effort("nonsense", caps) == "high"
    haiku = model_caps("claude-haiku-4-5")
    assert clamp_effort("max", haiku) == "high"  # clamped to model max


def test_detect_provider_prefers_free_key(monkeypatch):
    for var in ("JARVIS_PROVIDER", "GROQ_API_KEY", "NVIDIA_API_KEY", "CEREBRAS_API_KEY",
                "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY",
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert detect_provider() == "ollama"  # no keys → local default
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert detect_provider() == "groq"


def test_config_resolves_provider_preset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    cfg = Config(provider="groq", workspace=".")
    assert cfg.kind == "openai"
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.model == "llama-3.3-70b-versatile"
    assert cfg.api_key == "gsk_test"
    assert cfg.has_credentials()


def test_config_ollama_is_keyless():
    cfg = Config(provider="ollama", workspace=".")
    assert cfg.provider == "ollama"
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.has_credentials()  # keyless
    assert not cfg.needs_key()


def test_config_custom_endpoint():
    cfg = Config(provider="mylocal", base_url="http://127.0.0.1:9999/v1", workspace=".")
    assert cfg.kind == "openai"
    assert cfg.base_url == "http://127.0.0.1:9999/v1"


def test_detection_ignores_paid_keys(monkeypatch):
    for var in ("JARVIS_PROVIDER", "GROQ_API_KEY", "NVIDIA_API_KEY", "CEREBRAS_API_KEY",
                "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # A paid key present but no free key → still falls back to free local Ollama.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-paid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-paid")
    assert detect_provider() == "ollama"


def test_gemini_is_a_free_provider():
    assert "gemini" in PROVIDERS
    assert PROVIDERS["gemini"].free is True


def test_free_only_gate():
    assert Config(provider="groq", workspace=".").is_free_provider is True
    assert Config(provider="ollama", workspace=".").is_free_provider is True
    # Paid backends are blocked unless explicitly allowed.
    assert Config(provider="anthropic", workspace=".").paid_and_not_allowed() is True
    assert Config(provider="openai", workspace=".").paid_and_not_allowed() is True
    assert Config(provider="anthropic", allow_paid=True, workspace=".").paid_and_not_allowed() is False
    assert Config(provider="groq", workspace=".").paid_and_not_allowed() is False
