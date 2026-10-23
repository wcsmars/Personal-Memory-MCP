"""Append-only bitemporal assertions and leakage-safe hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embedding import EMBEDDING_ID, cosine_distance, embed, pack, tokens, unpack


def instant(value: str | None = None) -> str:
    """Canonical UTC ISO timestamp. Date-only input means UTC midnight."""
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif len(value) == 10:
        parsed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Timestamps must include a timezone; date-only means UTC midnight")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS revisions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    known_at TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0,1)),
    source_json TEXT NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE(namespace, memory_id, known_at),
    CHECK(valid_to IS NULL OR valid_to > valid_from)
);
CREATE INDEX IF NOT EXISTS ix_visibility ON revisions(namespace, known_at, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS ix_identity ON revisions(namespace, memory_id, known_at);
CREATE TRIGGER IF NOT EXISTS immutable_update BEFORE UPDATE ON revisions
BEGIN SELECT RAISE(ABORT, 'revisions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_delete BEFORE DELETE ON revisions
BEGIN SELECT RAISE(ABORT, 'revisions are append-only'); END;
"""


class MemoryStore:
    def __init__(self, path: str | Path, *, vector_backend: str = "auto"):
        if vector_backend not in {"auto", "python", "sqlite-vec"}:
            raise ValueError("vector_backend must be auto, python, or sqlite-vec")
        path = str(path)
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            path = str(Path(path).expanduser())
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('schema_version', '1')")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('embedding_id', ?)", (EMBEDDING_ID,))
        metadata = dict(self.connection.execute("SELECT key,value FROM metadata").fetchall())
        if metadata != {"schema_version": "1", "embedding_id": EMBEDDING_ID}:
            self.close()
            raise ValueError("Unsupported database schema or embedding version; migrate explicitly")
        self.connection.commit()
        self.vector_backend = "python"
        self.vector_note = "Explicit Python cosine baseline" if vector_backend == "python" else "sqlite-vec unavailable"
        if vector_backend != "python":
            try:
                import sqlite_vec
                self.connection.enable_load_extension(True)
                try:
                    sqlite_vec.load(self.connection)
                finally:
                    self.connection.enable_load_extension(False)
                self.vector_backend = "sqlite-vec"
                version = self.connection.execute("SELECT vec_version()").fetchone()[0]
                self.vector_note = version
            except (ImportError, AttributeError, sqlite3.Error) as exc:
                self.vector_note = f"Python cosine fallback: {type(exc).__name__}: {exc}"
                if vector_backend == "sqlite-vec":
                    self.close()
                    raise RuntimeError(self.vector_note) from exc

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.connection.close()

    @staticmethod
    def _scope(namespace: str) -> str:
        if not isinstance(namespace, str) or not namespace.strip() or len(namespace) > 200:
            raise ValueError("namespace must be a nonempty string of at most 200 characters")
        return namespace

    @staticmethod
    def _public(row: sqlite3.Row | dict) -> dict[str, Any]:
        result = dict(row)
        result.pop("embedding", None)
        result.pop("rn", None)
        result["source"] = json.loads(result.pop("source_json"))
        result["deleted"] = bool(result["deleted"])
        return result

    def remember(self, *, namespace: str, memory_id: str, content: str, title: str = "",
                 known_at: str | None = None, valid_from: str | None = None,
                 valid_to: str | None = None, source: dict | None = None,
                 deleted: bool = False) -> dict:
        self._scope(namespace)
        if not isinstance(memory_id, str) or not memory_id.strip() or len(memory_id) > 500:
            raise ValueError("memory_id must be nonempty and at most 500 characters")
        if not isinstance(content, str) or len(content.encode()) > 2_000_000:
            raise ValueError("content must be UTF-8 text under 2 MB")
        if not deleted and not content.strip():
            raise ValueError("Non-deleted memories require content")
        known = instant(known_at)
        start = instant(valid_from) if valid_from else known
        end = instant(valid_to) if valid_to else None
        if end is not None and end <= start:
            raise ValueError("valid_to must be later than valid_from")
        if not isinstance(source or {}, dict):
            raise ValueError("source must be a JSON object")
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        source_json = json.dumps(source or {}, sort_keys=True, ensure_ascii=False, allow_nan=False)
        values = (memory_id, namespace, title, content, content_hash, known, start, end,
                  int(deleted), source_json, pack(embed(title + " " + content)))
        existing = self.connection.execute(
            "SELECT * FROM revisions WHERE namespace=? AND memory_id=? AND known_at=?",
            (namespace, memory_id, known),
        ).fetchone()
        if existing:
            names = ["memory_id", "namespace", "title", "content", "content_hash", "known_at",
                     "valid_from", "valid_to", "deleted", "source_json", "embedding"]
            if tuple(existing[key] for key in names) != values:
                raise ValueError("A different revision already has this identity and known_at")
            return self._public(existing)
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO revisions(memory_id,namespace,title,content,content_hash,known_at,"
                "valid_from,valid_to,deleted,source_json,embedding) VALUES(?,?,?,?,?,?,?,?,?,?,?)", values)
        return self._public(self.connection.execute("SELECT * FROM revisions WHERE id=?", (cursor.lastrowid,)).fetchone())

    def visible(self, *, namespace: str, as_of: str | None = None,
                valid_at: str | None = None, include_deleted: bool = False) -> list[dict]:
        return [self._public(row) for row in self._visible_rows(
            namespace=namespace, as_of=as_of, valid_at=valid_at, include_deleted=include_deleted)]

    def _visible_rows(self, *, namespace: str, as_of: str | None = None,
                      valid_at: str | None = None, include_deleted: bool = False) -> list[sqlite3.Row]:
        cutoff = instant(as_of)
        when = instant(valid_at) if valid_at else cutoff
        rows = self.connection.execute("""
            WITH eligible AS (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY namespace,memory_id ORDER BY known_at DESC,id DESC
              ) rn FROM revisions
              WHERE namespace=? AND known_at<=? AND valid_from<=?
                AND (valid_to IS NULL OR ?<valid_to)
            ) SELECT * FROM eligible WHERE rn=1 AND (? OR deleted=0) ORDER BY memory_id
        """, (self._scope(namespace), cutoff, when, when, int(include_deleted))).fetchall()
        return rows

    def forget(self, *, namespace: str, memory_id: str, known_at: str | None = None,
               valid_from: str | None = None, valid_to: str | None = None) -> dict:
        known = instant(known_at)
        prior = self.connection.execute(
            "SELECT * FROM revisions WHERE namespace=? AND memory_id=? AND known_at<=? ORDER BY known_at DESC LIMIT 1",
            (self._scope(namespace), memory_id, known)).fetchone()
        if prior is None:
            raise KeyError("No memory with that identity was known by the deletion time")
        return self.remember(namespace=namespace, memory_id=memory_id, content="", title=prior["title"],
                             known_at=known, valid_from=valid_from or known, valid_to=valid_to,
                             deleted=True, source={"operation": "forget", "previous_revision": prior["id"]})

    def history(self, *, namespace: str, memory_id: str, as_of: str | None = None) -> list[dict]:
        """Audit history known by cutoff (defaults now); includes tombstones."""
        return [self._public(row) for row in self.connection.execute(
            "SELECT * FROM revisions WHERE namespace=? AND memory_id=? AND known_at<=? ORDER BY known_at,id",
            (self._scope(namespace), memory_id, instant(as_of))).fetchall()]

    def search(self, query: str, *, namespace: str, as_of: str | None = None,
               valid_at: str | None = None, limit: int = 8, candidate_limit: int = 100) -> dict:
        if not isinstance(query, str) or len(query) > 10_000:
            raise ValueError("query must be a string of at most 10,000 characters")
        if not 1 <= limit <= 100 or not limit <= candidate_limit <= 1000:
            raise ValueError("Require 1 <= limit <= 100 and limit <= candidate_limit <= 1000")
        cutoff = instant(as_of)
        when = instant(valid_at) if valid_at else cutoff
        rows = self._visible_rows(namespace=namespace, as_of=cutoff, valid_at=when)
        response = {"namespace": namespace, "as_of": cutoff, "valid_at": when,
                    "eligible_count": len(rows), "vector_backend": self.vector_backend,
                    "embedding_id": EMBEDDING_ID, "results": []}
        terms = list(dict.fromkeys(tokens(query)))[:100]
        if not rows or not terms:
            return response
        # Scope *before* computing corpus-dependent BM25, not only before LIMIT.
        self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.visible_fts USING fts5(title,content)")
        self.connection.execute("DELETE FROM visible_fts")
        self.connection.executemany("INSERT INTO visible_fts(rowid,title,content) VALUES(?,?,?)",
                                    [(row["id"], row["title"], row["content"]) for row in rows])
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        lexical = self.connection.execute(
            "SELECT rowid,bm25(visible_fts,2.0,1.0) score FROM visible_fts WHERE visible_fts MATCH ? "
            "ORDER BY score,rowid LIMIT ?", (match, candidate_limit)).fetchall()
        query_vector = embed(query)
        # An eligible-only regular table gives exact cosine over only visible rows.
        self.connection.execute("CREATE TEMP TABLE IF NOT EXISTS visible_vectors(id INTEGER PRIMARY KEY, embedding BLOB)")
        self.connection.execute("DELETE FROM visible_vectors")
        self.connection.executemany("INSERT INTO visible_vectors VALUES(?,?)", [(r["id"], r["embedding"]) for r in rows])
        if self.vector_backend == "sqlite-vec":
            vector_rows = self.connection.execute(
                "SELECT id,vec_distance_cosine(embedding,?) distance FROM visible_vectors "
                "WHERE vec_distance_cosine(embedding,?)<0.999999 ORDER BY distance,id LIMIT ?",
                (pack(query_vector), pack(query_vector), candidate_limit)).fetchall()
            vectors = [(r["id"], r["distance"]) for r in vector_rows]
        else:
            vectors = sorted(((r["id"], cosine_distance(query_vector, unpack(r["embedding"]))) for r in rows),
                             key=lambda pair: (pair[1], pair[0]))
            vectors = [(i, d) for i, d in vectors if d < 0.999999][:candidate_limit]
        scores: dict[int, dict] = {}
        for method, rankings in [("fts5", [(r["rowid"], r["score"]) for r in lexical]), ("vector", vectors)]:
            for rank, (row_id, raw_score) in enumerate(rankings, 1):
                detail = scores.setdefault(row_id, {"score": 0.0, "retrieval": {}})
                detail["score"] += 1.0 / (60 + rank)
                detail["retrieval"][method] = {"rank": rank, "raw_score": raw_score}
        by_id = {row["id"]: row for row in rows}
        ordered = sorted(scores, key=lambda i: (-scores[i]["score"], by_id[i]["memory_id"]))[:limit]
        response["results"] = [{**self._public(by_id[i]), **scores[i]} for i in ordered]
        self.connection.commit()
        return response

    def stats(self) -> dict:
        return {"schema_version": 1, "revisions": self.connection.execute("SELECT count(*) FROM revisions").fetchone()[0],
                "vector_backend": self.vector_backend, "vector_note": self.vector_note,
                "embedding_id": EMBEDDING_ID,
                "namespaces": [row[0] for row in self.connection.execute("SELECT DISTINCT namespace FROM revisions ORDER BY namespace")]}

    def export(self, path: str | Path, *, namespace: str, as_of: str | None = None,
               valid_at: str | None = None) -> dict:
        """Export visible snapshot as structured JSON, preserving exact provenance."""
        cutoff = instant(as_of)
        when = instant(valid_at) if valid_at else cutoff
        payload = {"schema_version": 1, "namespace": namespace, "as_of": cutoff, "valid_at": when,
                   "memories": self.visible(namespace=namespace, as_of=cutoff, valid_at=when)}
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Explicitly avoid clobbering the database or an existing user artifact.
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        return {"path": str(destination.resolve()), "memories": len(payload["memories"])}
