"""Agent loop tests driven by a fake LLM — no network required."""
from jarvis.agent import Agent

from .conftest import FakeLLM, fake_message, text_block, tool_block


def make_agent(config, memory, console, scripted):
    llm = FakeLLM(scripted)
    return Agent(config, llm, memory, console), llm


def test_simple_end_turn(config, memory, console):
    agent, llm = make_agent(
        config, memory, console,
        [fake_message([text_block("Hello, I am JARVIS.")], "end_turn")],
    )
    result = agent.run("say hello")
    assert "JARVIS" in result
    assert len(llm.calls) == 1


def test_tool_use_then_finish(config, memory, console):
    scripted = [
        fake_message(
            [tool_block("t1", "write_file", {"path": "out.txt", "content": "data"})],
            "tool_use",
        ),
        fake_message([text_block("Wrote the file.")], "end_turn"),
    ]
    agent, llm = make_agent(config, memory, console, scripted)
    result = agent.run("create out.txt")
    assert "Wrote the file." in result
    assert (config.workspace / "out.txt").read_text() == "data"
    # The tool_result was fed back as a user message before the second call.
    assert len(llm.calls) == 2
    second_msgs = llm.calls[1]["messages"]
    assert second_msgs[-1]["role"] == "user"
    assert second_msgs[-1]["content"][0]["type"] == "tool_result"


def test_unknown_tool_reports_error(config, memory, console):
    scripted = [
        fake_message([tool_block("t1", "does_not_exist", {})], "tool_use"),
        fake_message([text_block("Recovered.")], "end_turn"),
    ]
    agent, llm = make_agent(config, memory, console, scripted)
    result = agent.run("do something")
    assert result == "Recovered."
    tool_result = llm.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "Unknown tool" in tool_result["content"]


def test_refusal_stops(config, memory, console):
    agent, llm = make_agent(
        config, memory, console,
        [fake_message([], "refusal", stop_details=None)],
    )
    result = agent.run("something disallowed")
    assert "refused" in result.lower()
    assert len(llm.calls) == 1


def test_iteration_cap(config, memory, console):
    config.max_iterations = 3
    # Always asks for a tool → never terminates on its own.
    scripted = [
        fake_message([tool_block(f"t{i}", "list_dir", {"path": "."})], "tool_use")
        for i in range(5)
    ]
    agent, llm = make_agent(config, memory, console, scripted)
    agent.run("loop forever")
    assert len(llm.calls) == 3  # capped


def test_pause_turn_resumes(config, memory, console):
    scripted = [
        fake_message([text_block("searching...")], "pause_turn"),
        fake_message([text_block("Here are the results.")], "end_turn"),
    ]
    agent, llm = make_agent(config, memory, console, scripted)
    result = agent.run("search the web")
    assert "results" in result
    assert len(llm.calls) == 2
