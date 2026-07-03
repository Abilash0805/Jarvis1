import pytest

from jarvis.tools.base import PathError
from jarvis.tools.filesystem import EditFile, Glob, Grep, ListDir, ReadFile, WriteFile
from jarvis.tools.memory_tool import Recall, Remember
from jarvis.tools.tasks import UpdateTasks


def test_path_confinement(ctx):
    # A path escaping the workspace must be refused.
    with pytest.raises(PathError):
        ctx.resolve_path("../../etc/passwd")
    with pytest.raises(PathError):
        ctx.resolve_path("/etc/passwd")
    # A path inside the workspace resolves fine.
    p = ctx.resolve_path("sub/file.txt")
    assert str(p).startswith(str(ctx.workspace))


def test_write_read_roundtrip(ctx):
    w = WriteFile().run({"path": "notes/hello.txt", "content": "line1\nline2"}, ctx)
    assert not w.is_error
    r = ReadFile().run({"path": "notes/hello.txt"}, ctx)
    assert "line1" in r.content and "line2" in r.content


def test_read_missing_file(ctx):
    r = ReadFile().run({"path": "nope.txt"}, ctx)
    assert r.is_error


def test_edit_single_occurrence(ctx):
    WriteFile().run({"path": "a.txt", "content": "alpha beta gamma"}, ctx)
    e = EditFile().run({"path": "a.txt", "old_string": "beta", "new_string": "BETA"}, ctx)
    assert not e.is_error
    r = ReadFile().run({"path": "a.txt"}, ctx)
    assert "BETA" in r.content


def test_edit_ambiguous_without_replace_all(ctx):
    WriteFile().run({"path": "b.txt", "content": "x x x"}, ctx)
    e = EditFile().run({"path": "b.txt", "old_string": "x", "new_string": "y"}, ctx)
    assert e.is_error  # 3 occurrences, replace_all not set


def test_edit_replace_all(ctx):
    WriteFile().run({"path": "c.txt", "content": "x x x"}, ctx)
    e = EditFile().run(
        {"path": "c.txt", "old_string": "x", "new_string": "y", "replace_all": True}, ctx
    )
    assert not e.is_error


def test_list_dir(ctx):
    WriteFile().run({"path": "d/one.txt", "content": "1"}, ctx)
    WriteFile().run({"path": "d/two.txt", "content": "2"}, ctx)
    r = ListDir().run({"path": "d"}, ctx)
    assert "one.txt" in r.content and "two.txt" in r.content


def test_glob(ctx):
    WriteFile().run({"path": "src/a.py", "content": "pass"}, ctx)
    WriteFile().run({"path": "src/b.py", "content": "pass"}, ctx)
    WriteFile().run({"path": "src/c.md", "content": "x"}, ctx)
    r = Glob().run({"pattern": "src/*.py"}, ctx)
    assert "a.py" in r.content and "b.py" in r.content
    assert "c.md" not in r.content


def test_grep(ctx):
    WriteFile().run({"path": "code.py", "content": "def foo():\n    return 42\n"}, ctx)
    r = Grep().run({"pattern": r"def \w+", "glob": "*.py"}, ctx)
    assert "code.py" in r.content and "def foo" in r.content


def test_grep_invalid_regex(ctx):
    r = Grep().run({"pattern": "[unclosed"}, ctx)
    assert r.is_error


def test_memory_tools(ctx):
    Remember().run({"content": "prefers tabs over spaces", "kind": "preference"}, ctx)
    r = Recall().run({"query": "tabs"}, ctx)
    assert "tabs" in r.content


def test_update_tasks(ctx):
    r = UpdateTasks().run(
        {"tasks": [{"title": "plan", "status": "done"}, {"title": "build", "status": "in_progress"}]},
        ctx,
    )
    assert not r.is_error
    assert ctx.state["tasks"][0]["title"] == "plan"
    assert ctx.state["tasks"][1]["status"] == "in_progress"
