"""Shared test fixtures and fakes."""
import io
from pathlib import Path

import pytest
from rich.console import Console as RichConsole

from jarvis.config import Config
from jarvis.memory import Memory
from jarvis.providers.base import AssistantTurn, Backend, ToolCall
from jarvis.tools.base import ToolContext
from jarvis.ui import Console


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def config(workspace: Path) -> Config:
    # Keyless local provider keeps the fixture deterministic and offline.
    return Config(
        provider="ollama",
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


# -- backend fakes for the agent loop --------------------------------------

def turn(text="", tool_calls=None, stop_reason=None, refusal_category=None) -> AssistantTurn:
    calls = tool_calls or []
    stop = stop_reason or ("tool_use" if calls else "end_turn")
    return AssistantTurn(
        text=text, tool_calls=calls, stop_reason=stop, refusal_category=refusal_category
    )


def tcall(call_id, name, args) -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)


class FakeBackend(Backend):
    """Returns scripted AssistantTurns; records the neutral message log."""

    label = "fake"
    model = "fake-model"

    def __init__(self, turns):
        self.turns = list(turns)
        self.log = []  # ("user"|"tool"|"assistant", payload)
        self.system = None
        self.tools = None
        self.calls = 0

    def configure(self, system, tools):
        self.system, self.tools = system, tools

    def reset(self):
        self.log = []

    def add_user_message(self, text):
        self.log.append(("user", text))

    def add_tool_results(self, results):
        self.log.append(("tool", results))

    def run(self, callbacks):
        self.calls += 1
        t = self.turns.pop(0)
        if t.text:
            callbacks.text(t.text)
        self.log.append(("assistant", t))
        return t
