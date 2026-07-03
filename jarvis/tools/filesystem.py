"""Filesystem tools — all confined to the workspace root.

read_file / write_file / edit_file / list_dir / glob / grep. Path confinement is
enforced by :meth:`ToolContext.resolve_path`.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from .base import PathError, Tool, ToolContext, ToolResult

MAX_READ_BYTES = 400_000
MAX_MATCHES = 200
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".jarvis", "dist", "build"}


def _rel(ctx: ToolContext, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ctx.workspace.resolve()))
    except ValueError:
        return str(p)


class ReadFile(Tool):
    name = "read_file"
    description = "Read a text file from the workspace. Returns numbered lines. Use offset/limit for large files."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root"},
            "offset": {"type": "integer", "description": "1-indexed start line (optional)"},
            "limit": {"type": "integer", "description": "Max lines to read (optional)"},
        },
        "required": ["path"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve_path(args["path"])
        except PathError as e:
            return ToolResult.error(str(e))
        if not path.exists():
            return ToolResult.error(f"File not found: {args['path']}")
        if path.is_dir():
            return ToolResult.error(f"{args['path']} is a directory; use list_dir")
        try:
            data = path.read_bytes()
        except OSError as e:
            return ToolResult.error(f"Cannot read {args['path']}: {e}")
        if len(data) > MAX_READ_BYTES:
            data = data[:MAX_READ_BYTES]
            truncated = True
        else:
            truncated = False
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        offset = max(1, int(args.get("offset", 1)))
        limit = int(args.get("limit", len(lines)))
        chunk = lines[offset - 1 : offset - 1 + limit]
        numbered = "\n".join(f"{offset + i:>6}\t{ln}" for i, ln in enumerate(chunk))
        note = "\n... (truncated at 400KB)" if truncated else ""
        if not numbered:
            return ToolResult.ok(f"(empty or no lines in range for {args['path']})")
        return ToolResult.ok(numbered + note)


class WriteFile(Tool):
    name = "write_file"
    description = "Create or overwrite a file in the workspace with the given content."
    mutating = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve_path(args["path"])
        except PathError as e:
            return ToolResult.error(str(e))
        existed = path.exists()
        gate = self._gate(ctx, f"{'overwrite' if existed else 'create'} file {_rel(ctx, path)}")
        if gate:
            return gate
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Cannot write {args['path']}: {e}")
        n = len(args["content"].splitlines())
        verb = "Overwrote" if existed else "Created"
        return ToolResult.ok(f"{verb} {_rel(ctx, path)} ({n} lines).")


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must appear exactly once "
        "unless replace_all is true. Use this for surgical edits."
    )
    mutating = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve_path(args["path"])
        except PathError as e:
            return ToolResult.error(str(e))
        if not path.exists():
            return ToolResult.error(f"File not found: {args['path']}")
        gate = self._gate(ctx, f"edit file {_rel(ctx, path)}")
        if gate:
            return gate
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Cannot read {args['path']}: {e}")
        old, new = args["old_string"], args["new_string"]
        count = text.count(old)
        if count == 0:
            return ToolResult.error("old_string not found in file; no change made.")
        if count > 1 and not args.get("replace_all"):
            return ToolResult.error(
                f"old_string appears {count} times; pass replace_all=true or add more context."
            )
        updated = text.replace(old, new)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Cannot write {args['path']}: {e}")
        return ToolResult.ok(f"Edited {_rel(ctx, path)} ({count} replacement(s)).")


class ListDir(Tool):
    name = "list_dir"
    description = "List the entries of a directory in the workspace."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path (default '.')"}},
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve_path(args.get("path", "."))
        except PathError as e:
            return ToolResult.error(str(e))
        if not path.exists():
            return ToolResult.error(f"Not found: {args.get('path', '.')}")
        if not path.is_dir():
            return ToolResult.error(f"{args.get('path', '.')} is not a directory")
        entries = []
        for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            marker = "/" if entry.is_dir() else ""
            entries.append(f"{entry.name}{marker}")
        listing = "\n".join(entries) if entries else "(empty)"
        return ToolResult.ok(f"{_rel(ctx, path)}/\n{listing}")


class Glob(Tool):
    name = "glob"
    description = "Find files by glob pattern (e.g. '**/*.py'). Returns matching paths."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Base dir (default workspace root)"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            base = ctx.resolve_path(args.get("path", "."))
        except PathError as e:
            return ToolResult.error(str(e))
        matches = []
        for p in sorted(base.glob(args["pattern"])):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            matches.append(_rel(ctx, p))
            if len(matches) >= MAX_MATCHES:
                break
        if not matches:
            return ToolResult.ok(f"No files matching {args['pattern']!r}.")
        return ToolResult.ok("\n".join(matches))


class Grep(Tool):
    name = "grep"
    description = "Search file contents by regex across the workspace. Returns matching lines with file:line."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression"},
            "path": {"type": "string", "description": "Base dir (default workspace root)"},
            "glob": {"type": "string", "description": "Only search files matching this glob (e.g. '*.py')"},
            "ignore_case": {"type": "boolean"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            base = ctx.resolve_path(args.get("path", "."))
        except PathError as e:
            return ToolResult.error(str(e))
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        try:
            rx = re.compile(args["pattern"], flags)
        except re.error as e:
            return ToolResult.error(f"Invalid regex: {e}")
        file_glob = args.get("glob")
        out: list[str] = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if file_glob and not fnmatch.fnmatch(fname, file_glob):
                    continue
                fpath = Path(root) / fname
                try:
                    with fpath.open("r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if rx.search(line):
                                out.append(f"{_rel(ctx, fpath)}:{lineno}: {line.rstrip()[:300]}")
                                if len(out) >= MAX_MATCHES:
                                    out.append("... (truncated)")
                                    return ToolResult.ok("\n".join(out))
                except OSError:
                    continue
        if not out:
            return ToolResult.ok(f"No matches for {args['pattern']!r}.")
        return ToolResult.ok("\n".join(out))


FILESYSTEM_TOOLS = [ReadFile(), WriteFile(), EditFile(), ListDir(), Glob(), Grep()]
