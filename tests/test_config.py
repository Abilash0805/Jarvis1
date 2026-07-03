from jarvis.config import clamp_effort, model_caps


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
