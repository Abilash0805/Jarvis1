"""Keyless, client-side web tools.

These work with any backend and require no API key: web search via DuckDuckGo's
HTML endpoint, and page fetch with HTML stripped to readable text. Uses only the
standard library, so there's nothing extra to install.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from .base import Tool, ToolContext, ToolResult

USER_AGENT = "Mozilla/5.0 (compatible; JARVIS/0.2; +https://github.com/)"
DDG_HTML = "https://html.duckduckgo.com/html/"
MAX_FETCH_BYTES = 2_000_000
MAX_TEXT_CHARS = 12_000


def _http(url: str, *, data: bytes | None = None, timeout: int = 20) -> str:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(MAX_FETCH_BYTES)
    charset = "utf-8"
    ctype = ""
    try:
        ctype = resp.headers.get_content_charset() or ""
    except Exception:  # pragma: no cover
        ctype = ""
    if ctype:
        charset = ctype
    return raw.decode(charset, errors="replace")


# -- DuckDuckGo result parsing ----------------------------------------------

_RESULT_RX = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RX = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(?P<snip>.*?)</a>', re.DOTALL
)


def _strip_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _decode_ddg_href(href: str) -> str:
    """DuckDuckGo wraps targets in a redirect: /l/?uddg=<encoded>."""
    if "uddg=" in href:
        q = urllib.parse.urlparse(href).query
        params = urllib.parse.parse_qs(q)
        if "uddg" in params:
            return urllib.parse.unquote(params["uddg"][0])
    if href.startswith("//"):
        return "https:" + href
    return href


def parse_ddg_results(html_text: str, limit: int) -> list[dict[str, str]]:
    titles = list(_RESULT_RX.finditer(html_text))
    snippets = list(_SNIPPET_RX.finditer(html_text))
    out: list[dict[str, str]] = []
    for i, m in enumerate(titles[:limit]):
        snip = _strip_tags(snippets[i].group("snip")) if i < len(snippets) else ""
        out.append(
            {
                "title": _strip_tags(m.group("title")),
                "url": _decode_ddg_href(m.group("href")),
                "snippet": snip,
            }
        )
    return out


class WebSearch(Tool):
    name = "web_search"
    description = "Search the web (via DuckDuckGo, no API key). Returns titles, URLs, and snippets."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "Max results (default 6)"},
        },
        "required": ["query"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = args["query"]
        limit = int(args.get("limit", 6))
        try:
            body = _http(DDG_HTML, data=urllib.parse.urlencode({"q": query}).encode())
        except Exception as e:  # noqa: BLE001 — network is best-effort
            return ToolResult.error(f"Web search failed: {e}")
        results = parse_ddg_results(body, limit)
        if not results:
            return ToolResult.ok(f"No results for {query!r}.")
        lines = [
            f"{i + 1}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
            for i, r in enumerate(results)
        ]
        return ToolResult.ok("\n".join(lines))


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "head", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip == 0:
            text = data.strip()
            if text:
                self.parts.append(text)


def html_to_text(html_text: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html_text)
    except Exception:  # noqa: BLE001 — tolerate malformed markup
        pass
    text = "\n".join(parser.parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class WebFetch(Tool):
    name = "web_fetch"
    description = "Fetch a URL and return its readable text content (HTML stripped). No API key."
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = args["url"]
        if not re.match(r"^https?://", url):
            return ToolResult.error("URL must start with http:// or https://")
        try:
            body = _http(url)
        except Exception as e:  # noqa: BLE001
            return ToolResult.error(f"Fetch failed: {e}")
        text = html_to_text(body)
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n... (truncated)"
        return ToolResult.ok(text or "(no extractable text)")


WEB_TOOLS = [WebSearch(), WebFetch()]
