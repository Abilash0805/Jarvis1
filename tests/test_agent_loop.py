"""Agent loop tests driven by a fake backend — no network required."""
from jarvis.agent import Agent

from .conftest import FakeBackend, tcall, turn


def make_agent(config, memory, console, turns, *, factory=None):
    backend = FakeBackend(turns)
    return Agent(config, backend, memory, console, backend_factory=factory), backend


def test_simple_end_turn(config, memory, console):
    agent, be = make_agent(config, memory, console, [turn("Hello, I am JARVIS.")])
    result = agent.run("say hello")
    assert "JARVIS" in result
    assert be.calls == 1


def test_tool_use_then_finish(config, memory, console):
    turns = [
        turn(tool_calls=[tcall("t1", "write_file", {"path": "out.txt", "content": "data"})]),
        turn("Wrote the file."),
    ]
    agent, be = make_agent(config, memory, console, turns)
    result = agent.run("create out.txt")
    assert result == "Wrote the file."
    assert (config.workspace / "out.txt").read_text() == "data"
    assert be.calls == 2
    # A tool-results message was fed back between the two runs.
    kinds = [k for k, _ in be.log]
    assert "tool" in kinds
    results = next(p for k, p in be.log if k == "tool")
    assert results[0].id == "t1" and results[0].is_error is False


def test_unknown_tool_reports_error(config, memory, console):
    turns = [turn(tool_calls=[tcall("t1", "does_not_exist", {})]), turn("Recovered.")]
    agent, be = make_agent(config, memory, console, turns)
    result = agent.run("do something")
    assert result == "Recovered."
    results = next(p for k, p in be.log if k == "tool")
    assert results[0].is_error is True
    assert "Unknown tool" in results[0].content


def test_refusal_stops(config, memory, console):
    agent, be = make_agent(config, memory, console, [turn(stop_reason="refusal")])
    result = agent.run("disallowed")
    assert "refused" in result.lower()
    assert be.calls == 1


def test_iteration_cap(config, memory, console):
    config.max_iterations = 3
    turns = [turn(tool_calls=[tcall(f"t{i}", "list_dir", {"path": "."})]) for i in range(5)]
    agent, be = make_agent(config, memory, console, turns)
    agent.run("loop")
    assert be.calls == 3


def test_pause_turn_resumes(config, memory, console):
    turns = [turn("searching...", stop_reason="pause_turn"), turn("Here are the results.")]
    agent, be = make_agent(config, memory, console, turns)
    result = agent.run("search")
    assert "results" in result
    assert be.calls == 2


def test_max_tokens_continues(config, memory, console):
    turns = [turn("partial", stop_reason="max_tokens"), turn("done")]
    agent, be = make_agent(config, memory, console, turns)
    result = agent.run("write something long")
    assert result == "done"
    assert be.calls == 2
    assert any(k == "user" and "continue" in str(p).lower() for k, p in be.log)


def test_delegation(config, memory, console):
    # Main agent delegates once, then finishes; sub-agent returns a report.
    sub_backends = [FakeBackend([turn("Sub-agent report: found 3 files.")])]
    factory = lambda: sub_backends.pop(0)  # noqa: E731
    main_turns = [
        turn(tool_calls=[tcall("d1", "delegate", {"objective": "count files"})]),
        turn("Done — the sub-agent found 3 files."),
    ]
    agent, be = make_agent(config, memory, console, main_turns, factory=factory)
    result = agent.run("count the files")
    assert "3 files" in result
    results = next(p for k, p in be.log if k == "tool")
    assert "Sub-agent report" in results[0].content
