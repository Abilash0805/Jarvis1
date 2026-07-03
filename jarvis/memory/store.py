"""Persistent memory: SQLite + FTS5 full-text search.

Memory survives across sessions. Each record has a kind (``fact`` / ``lesson`` /
``preference`` / ``note``), free text, optional tags, and timestamps. Recall is
full-text ranked when FTS5 is available, with a graceful LIKE fallback otherwise.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

VALID_KINDS = ("fact", "lesson", "preference", "note")


@dataclass
class MemoryRecord:
    id: int
    kind: str
    content: str
    tags: str
    created_at: float
    updated_at: float

    def render(self) -> str:
        tag = f" [{self.tags}]" if self.tags else ""
        return f"#{self.id} ({self.kind}){tag}: {self.content}"


def _fts_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


class Memory:
    """A durable key-facts / lessons store backed by SQLite."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.fts = _fts_available(self.conn)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                kind       TEXT NOT NULL DEFAULT 'note',
                content    TEXT NOT NULL,
                tags       TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        if self.fts:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, tags, content='memories', content_rowid='id')
                """
            )
            # Keep the FTS index in sync with the base table via triggers.
            cur.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                  INSERT INTO memories_fts(rowid, content, tags)
                  VALUES (new.id, new.content, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                  INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                  VALUES ('delete', old.id, old.content, old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                  INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                  VALUES ('delete', old.id, old.content, old.tags);
                  INSERT INTO memories_fts(rowid, content, tags)
                  VALUES (new.id, new.content, new.tags);
                END;
                """
            )
        self.conn.commit()

    # -- writes -------------------------------------------------------------

    def add(self, content: str, *, kind: str = "note", tags: str = "") -> int:
        content = (content or "").strip()
        if not content:
            raise ValueError("cannot store empty memory content")
        if kind not in VALID_KINDS:
            kind = "note"
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO memories (kind, content, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (kind, content, tags.strip(), now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def forget(self, memory_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # -- reads --------------------------------------------------------------

    def _row(self, r: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=r["id"],
            kind=r["kind"],
            content=r["content"],
            tags=r["tags"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    def get(self, memory_id: int) -> MemoryRecord | None:
        r = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row(r) if r else None

    def list(self, *, kind: str | None = None, limit: int = 100) -> list[MemoryRecord]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE kind = ? ORDER BY updated_at DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def search(self, query: str, *, limit: int = 10) -> list[MemoryRecord]:
        query = (query or "").strip()
        if not query:
            return []
        if self.fts:
            try:
                rows = self.conn.execute(
                    """
                    SELECT m.* FROM memories_fts f
                    JOIN memories m ON m.id = f.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (_fts_query(query), limit),
                ).fetchall()
                if rows:
                    return [self._row(r) for r in rows]
            except sqlite3.OperationalError:
                pass  # malformed FTS expression — fall through to LIKE
        like = f"%{query}%"
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE content LIKE ? OR tags LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (like, like, limit),
        ).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def digest(self, *, limit: int = 12) -> str:
        """A compact snapshot for injection into the system prompt."""
        recs = self.list(limit=limit)
        if not recs:
            return ""
        lines = "\n".join(f"- {r.render()}" for r in recs)
        return f"\n\n# Memory (most recent {len(recs)} of {self.count()})\n{lines}"

    def close(self) -> None:
        self.conn.close()


def _fts_query(raw: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Quote each token to neutralize FTS operator characters, then OR them so a
    multi-word query still recalls partial matches.
    """
    tokens = [t for t in raw.replace('"', " ").split() if t]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)
