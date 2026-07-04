"""Keyless web-tool parsing (no network — fixtures only)."""
from jarvis.tools.web_client import (
    _decode_ddg_href,
    html_to_text,
    parse_ddg_results,
)

DDG_FIXTURE = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">First Result</a>
  <a class="result__snippet">A snippet about the first result.</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb">Second &amp; Result</a>
  <a class="result__snippet">Second <b>snippet</b>.</a>
</div>
"""


def test_parse_ddg_results():
    results = parse_ddg_results(DDG_FIXTURE, limit=5)
    assert len(results) == 2
    assert results[0]["title"] == "First Result"
    assert results[0]["url"] == "https://example.com/a"
    assert "first result" in results[0]["snippet"].lower()
    # HTML entities decoded, inner tags stripped.
    assert results[1]["title"] == "Second & Result"
    assert results[1]["snippet"] == "Second snippet."


def test_parse_ddg_limit():
    assert len(parse_ddg_results(DDG_FIXTURE, limit=1)) == 1


def test_decode_ddg_href():
    assert _decode_ddg_href("//duckduckgo.com/l/?uddg=https%3A%2F%2Ffoo.com%2Fx") == "https://foo.com/x"
    assert _decode_ddg_href("//example.com/direct") == "https://example.com/direct"
    assert _decode_ddg_href("https://plain.com") == "https://plain.com"


def test_html_to_text_strips_tags_and_scripts():
    html = """
    <html><head><style>.x{color:red}</style><title>T</title></head>
    <body><script>alert(1)</script><h1>Heading</h1><p>Some <b>bold</b> text.</p></body></html>
    """
    text = html_to_text(html)
    assert "Heading" in text
    assert "bold" in text
    assert "alert(1)" not in text  # script content skipped
    assert "color:red" not in text  # style content skipped
