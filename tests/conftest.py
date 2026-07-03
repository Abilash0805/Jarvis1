"""Shared test fixtures and fakes."""
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console as RichConsole

from jarvis.config import Config
from jarvis.memory import Memory
from jarvis.tools.base import ToolContext
from jarvis.ui import Console


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def config(workspace: Path) -> Config:
    return Config(
        model="claude-opus-4-8",
        workspace=workspace,
        state_dir=workspace / ".jarvis",
        autonomous=True,
    )


@pytest.fixture
def memory(config: Config) -> Memory:
    config.ensure_dirs()
    mem = Memory(config.memory_path)
    yield mem
    mem.close()


@pytest.fixture
def console() -> Console:
    c = Console(quiet=True, show_thinking=False)
    c.rich = RichConsole(file=io.StringIO(), force_terminal=False)
    return c


@pytest.fixture
def ctx(config: Config, memory: Memory, console: Console) -> ToolContext:
    return ToolContext(
        config=config,
        memory=memory,
        console=console,
        workspace=config.workspace,
        autonomous=True,
        confirm=lambda action: True,
        delegate=None,
        depth=0,
        state={},
    )


# -- fakes for the agent loop ----------------------------------------------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id, name, args):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def fake_message(content, stop_reason, stop_details=None):
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=stop_details)


class FakeLLM:
    """Returns scripted messages in order; records the kwargs of each call."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def complete(self, **kwargs):
        # Snapshot the messages list — the agent mutates it in place, and the
        # real SDK serializes it at call time, so tests must see a point-in-time copy.
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs.get("messages", []))
        self.calls.append(snapshot)
        return self.scripted.pop(0)
