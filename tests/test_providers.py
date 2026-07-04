"""OpenAI-compatible backend + ThinkRouter, with a fake OpenAI client."""
from types import SimpleNamespace

from jarvis.config import Config
from jarvis.providers.base import StreamCallbacks
from jarvis.providers.openai_compat import OpenAIBackend, ThinkRouter, _to_openai_tools


# -- fake OpenAI streaming client -------------------------------------------

def chunk(content=None, tool_calls=None, finish=None, reasoning=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def tc_delta(index, id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


class _Completions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.chunks)


class FakeOpenAIClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_Completions(chunks))


def make_backend(chunks):
    cfg = Config(provider="groq", model="test-model", workspace=".")
    return OpenAIBackend(cfg, client=FakeOpenAIClient(chunks))


# -- tests ------------------------------------------------------------------

def test_text_only_stream():
    be = make_backend([chunk("Hi "), chunk("there", finish="stop")])
    be.configure("system", [])
    be.add_user_message("hello")
    captured = []
    turn = be.run(StreamCallbacks(on_text=captured.append))
    assert turn.text == "Hi there"
    assert turn.stop_reason == "end_turn"
    assert "".join(captured) == "Hi there"
    assert be.messages[-1] == {"role": "assistant", "content": "Hi there"}


def test_tool_call_stream_accumulates_arguments():
    chunks = [
        chunk(tool_calls=[tc_delta(0, id="call_1", name="web_search", arguments='{"query": ')]),
        chunk(tool_calls=[tc_delta(0, arguments='"cats"}')], finish="tool_calls"),
    ]
    be = make_backend(chunks)
    be.configure("system", [{"name": "web_search", "description": "d", "input_schema": {}}])
    be.add_user_message("find cats")
    turn = be.run(StreamCallbacks())
    assert turn.stop_reason == "tool_use"
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "web_search"
    assert call.args == {"query": "cats"}
    assert call.id == "call_1"
    # History assistant message carries the raw tool_calls.
    assert be.messages[-1]["tool_calls"][0]["function"]["name"] == "web_search"


def test_tool_results_appended_as_tool_role():
    from jarvis.providers.base import ToolResultMsg

    be = make_backend([chunk("ok", finish="stop")])
    be.configure("s", [])
    be.add_tool_results([ToolResultMsg(id="call_1", name="web_search", content="result", is_error=False)])
    assert be.messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "result"}


def test_reasoning_content_routed_to_thinking():
    be = make_backend([chunk(reasoning="let me think"), chunk("answer", finish="stop")])
    be.configure("s", [])
    thoughts = []
    turn = be.run(StreamCallbacks(on_thinking=thoughts.append))
    assert "".join(thoughts) == "let me think"
    assert turn.text == "answer"


def test_to_openai_tools_shape():
    tools = _to_openai_tools([{"name": "x", "description": "d", "input_schema": {"type": "object"}}])
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "x"
    assert tools[0]["function"]["parameters"] == {"type": "object"}


# -- ThinkRouter ------------------------------------------------------------

def run_router(deltas):
    text, think = [], []
    r = ThinkRouter(StreamCallbacks(on_text=text.append, on_thinking=think.append))
    for d in deltas:
        r.feed(d)
    r.close()
    return "".join(text), "".join(think), r.visible_text()


def test_router_plain_text():
    text, think, visible = run_router(["hello ", "world"])
    assert text == "hello world"
    assert think == ""
    assert visible == "hello world"


def test_router_think_block():
    text, think, visible = run_router(["<think>reasoning here</think>the answer"])
    assert think == "reasoning here"
    assert text == "the answer"
    assert visible == "the answer"


def test_router_tag_split_across_deltas():
    text, think, visible = run_router(["<th", "ink>hidden</th", "ink>shown"])
    assert think == "hidden"
    assert text == "shown"
    assert visible == "shown"
