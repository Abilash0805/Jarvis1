"""Server-side web tools.

Web search and web fetch run on Anthropic's infrastructure — they are declared,
not implemented here. The agent appends these declarations to the tool list when
web access is enabled and the model supports them.
"""
from __future__ import annotations


def web_tool_declarations() -> list[dict]:
    """Tool declarations for Claude's server-side web search + fetch.

    The ``_20260209`` variants include dynamic filtering; they run code under the
    hood, so we do NOT separately declare a code-execution tool alongside them.
    """
    return [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 8},
    ]
