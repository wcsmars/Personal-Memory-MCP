"""Append-only bitemporal assertions and leakage-safe hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embedding import EMBEDDING_ID, cosine_distance, embed, pack, unpack

MAX_CONTENT_BYTES = 2_000_000
MAX_TITLE_CHARS = 1_000
MAX_ID_CHARS = 500
# Cosine distances closer than this are ties (broken by memory_id). sqlite-vec computes in
# float32 and Python in float64; equal scores can differ by ~1e-7 between them, so exact
# comparison or a fixed rounding grid would let the two backends order ties differently.
TIE_TOLERANCE = 1e-6
TIME_HELP = "Use YYYY-MM-DD (UTC midnight) or an ISO 8601 timestamp with a timezone, such as 2025-01-01T09:30:00Z"
_DATE_ONLY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_HAS_TIME = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]")
_EXCESS_FRACTION = re.compile(r"[.,][0-9]{7,}")
_FTS_TERM = re.compile(r"[^\W_]+", re.UNICODE)


def _competition_ranks(pairs: list[tuple[int, float]], name, tolerance: float = 0.0) -> list[tuple[int, float, int]]:
    """Rank ascending scores; tied rows share the best rank ("1224") and are listed by memory_id.

    Values within ``tolerance`` of the previous value are ties, so float32 and float64
    computations of the same score group identically.
    """
    keyed, group, previous = [], 0, None
    for row_id, score in sorted(pairs, key=lambda pair: pair[1]):
        if previous is not None and score - previous > tolerance:
            group += 1
        previous = score
        keyed.append((group, name(row_id), row_id, score))
    keyed.sort()
    ranked, first_position = [], {}
    for position, (group, _, row_id, score) in enumerate(keyed, 1):
        ranked.append((row_id, score, first_position.setdefault(group, position)))
    return ranked


def _connect_read_only(uri: str) -> sqlite3.Connection:
    connection = sqlite3.connect(uri + "?mode=ro", uri=True)
    try:
        connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return connection
    except sqlite3.OperationalError as exc:
        connection.close()
        if "readonly" not in str(exc):
            raise
    # A WAL database in a read-only folder cannot create its -shm file. Nothing can write to
    # it there either, so it is safe to open as immutable (no locking, no WAL recovery).
    return sqlite3.connect(uri + "?mode=ro&immutable=1", uri=True)


class NotFound(KeyError):
    """No memory with the requested identity exists at the requested time."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "Memory not found"


def instant(value: str | None = None) -> str:
    """Canonical UTC ISO timestamp. Date-only input means UTC midnight."""
    if value is None:
        parsed = datetime.now(timezone.utc)
    else:
        if not isinstance(value, str):
            raise ValueError(f"Times must be strings. {TIME_HELP}")
        date_only = bool(_DATE_ONLY.fullmatch(value))
        if not date_only and not _HAS_TIME.search(value):
            raise ValueError(f"Invalid time {value!r}. {TIME_HELP}")
        if not date_only and _EXCESS_FRACTION.search(value):
            raise ValueError(f"Times support at most microsecond precision: {value!r}")
        try:
            parsed = datetime.fromisoformat(value if date_only else value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid time {value!r}. {TIME_HELP}") from exc
        if date_only:
            parsed = parsed.replace(tzinfo=timezone.utc)
        elif parsed.tzinfo is None:
            raise ValueError(f"Timestamps must include a timezone: {value!r}. {TIME_HELP}")
    try:
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    except OverflowError as exc:
        raise ValueError(f"Time is outside the supported range: {value!r}") from exc


def fts_text(text: str) -> str:
    """Compatibility-normalize text for the FTS5 unicode61 tokenizer (ligatures, NFD/NFC forms)."""
    return unicodedata.normalize("NFKC", text)


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
    """SQLite store. ``read_only=True`` opens an existing database without writing to it."""

    def __init__(self, path: str | Path, *, vector_backend: str = "auto", read_only: bool = False):
        if vector_backend not in {"auto", "python", "sqlite-vec"}:
            raise ValueError("vector_backend must be auto, python, or sqlite-vec")
        path = str(path)
        self.read_only = read_only
        if read_only:
            file = Path(path).expanduser()
            if path == ":memory:" or not file.is_file():
                raise FileNotFoundError(f"Memory database not found: {path}")
            self.connection = _connect_read_only(file.resolve().as_uri())
        else:
            if path != ":memory:":
                Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
                path = str(Path(path).expanduser())
            self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=5000")
        try:
            if not read_only:
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.executescript(SCHEMA)
                self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('schema_version', '1')")
                self.connection.execute("INSERT OR IGNORE INTO metadata VALUES('embedding_id', ?)", (EMBEDDING_ID,))
                self.connection.commit()
            metadata = dict(self.connection.execute("SELECT key,value FROM metadata").fetchall())
        except sqlite3.DatabaseError as exc:
            self.close()
            raise ValueError(f"Not a usable memory database: {exc}") from exc
        if metadata != {"schema_version": "1", "embedding_id": EMBEDDING_ID}:
            self.close()
            raise ValueError("Unsupported database schema or embedding version; migrate explicitly")
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
                self.vector_note = self.connection.execute("SELECT vec_version()").fetchone()[0]
            except (ImportError, AttributeError, sqlite3.Error) as exc:
                if vector_backend == "sqlite-vec":
                    self.close()
                    raise RuntimeError(f"sqlite-vec requested but unavailable: {type(exc).__name__}: {exc}") from exc
                self.vector_note = f"Python cosine fallback: {type(exc).__name__}: {exc}"

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
        if not isinstance(memory_id, str) or not memory_id.strip() or len(memory_id) > MAX_ID_CHARS:
            raise ValueError(f"memory_id must be nonempty and at most {MAX_ID_CHARS} characters")
        if not isinstance(content, str) or len(content.encode()) > MAX_CONTENT_BYTES:
            raise ValueError("content must be UTF-8 text under 2 MB")
        if not isinstance(title, str) or len(title) > MAX_TITLE_CHARS:
            raise ValueError(f"title must be text of at most {MAX_TITLE_CHARS} characters")
        if not deleted and not content.strip():
            raise ValueError("Non-deleted memories require content")
        if source is not None and not isinstance(source, dict):
            raise ValueError("source must be a JSON object")
        known = instant(known_at)
        start = instant(valid_from) if valid_from else known
        end = instant(valid_to) if valid_to else None
        if end is not None and end <= start:
            raise ValueError("valid_to must be later than valid_from")
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

    def _visible_rows(self, *, namespace: str, as_of: str | None = None, valid_at: str | None = None,
                      include_deleted: bool = False, memory_id: str | None = None) -> list[sqlite3.Row]:
        cutoff = instant(as_of)
        when = instant(valid_at) if valid_at else cutoff
        return self.connection.execute("""
            WITH eligible AS (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY namespace,memory_id ORDER BY known_at DESC,id DESC
              ) rn FROM revisions
              WHERE namespace=? AND known_at<=? AND valid_from<=?
                AND (valid_to IS NULL OR ?<valid_to) AND (? IS NULL OR memory_id=?)
            ) SELECT * FROM eligible WHERE rn=1 AND (? OR deleted=0) ORDER BY memory_id
        """, (self._scope(namespace), cutoff, when, when, memory_id, memory_id, int(include_deleted))).fetchall()

    def forget(self, *, namespace: str, memory_id: str, known_at: str | None = None,
               valid_from: str | None = None, valid_to: str | None = None) -> dict:
        """Append a tombstone hiding ``memory_id`` over [valid_from, valid_to).

        Repeating an identical forget is a no-op. The tombstone records the revision it hides.
        """
        self._scope(namespace)
        known = instant(known_at)
        start = instant(valid_from) if valid_from else known
        end = instant(valid_to) if valid_to else None
        existing = self.connection.execute(
            "SELECT * FROM revisions WHERE namespace=? AND memory_id=? AND known_at=?",
            (namespace, memory_id, known)).fetchone()
        if existing and existing["deleted"] and existing["valid_from"] == start and existing["valid_to"] == end:
            return self._public(existing)
        current = self._visible_rows(namespace=namespace, as_of=known, valid_at=start,
                                     include_deleted=True, memory_id=memory_id)
        if current and current[0]["deleted"]:
            raise ValueError("That memory is already forgotten at valid_from")
        if current:
            prior = current[0]
        else:
            # Not valid at valid_from (for example a scheduled assertion): use the latest live revision.
            prior = self.connection.execute(
                "SELECT * FROM revisions WHERE namespace=? AND memory_id=? AND known_at<=? AND deleted=0 "
                "ORDER BY known_at DESC,id DESC LIMIT 1", (namespace, memory_id, known)).fetchone()
        if prior is None:
            raise NotFound("No memory with that identity was known by the deletion time")
        return self.remember(namespace=namespace, memory_id=memory_id, content="", title=prior["title"],
                             known_at=known, valid_from=start, valid_to=end,
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
        if type(limit) is not int or type(candidate_limit) is not int:
            raise ValueError("limit and candidate_limit must be integers")
        if not 1 <= limit <= 100 or not limit <= candidate_limit <= 1000:
            raise ValueError("Require 1 <= limit <= 100 and limit <= candidate_limit <= 1000")
        cutoff = instant(as_of)
        when = instant(valid_at) if valid_at else cutoff
        rows = self._visible_rows(namespace=namespace, as_of=cutoff, valid_at=when)
        response = {"namespace": namespace, "as_of": cutoff, "valid_at": when,
                    "eligible_count": len(rows), "vector_backend": self.vector_backend,
                    "embedding_id": EMBEDDING_ID, "results": []}
        normalized = fts_text(query)
        # FTS5's unicode61 tokenizer folds case itself; casefolding here would break words such as "Straße".
        terms = list(dict.fromkeys(_FTS_TERM.findall(normalized)))[:100]
        if not rows or not terms:
            return response
        by_id = {row["id"]: row for row in rows}
        name = lambda row_id: by_id[row_id]["memory_id"]
        # Scope *before* computing corpus-dependent BM25, not only before LIMIT.
        self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.visible_fts USING fts5(title,content)")
        self.connection.execute("DELETE FROM temp.visible_fts")
        self.connection.executemany("INSERT INTO temp.visible_fts(rowid,title,content) VALUES(?,?,?)",
                                    [(row["id"], fts_text(row["title"]), fts_text(row["content"])) for row in rows])
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        lexical = _competition_ranks([(r["rowid"], r["score"]) for r in self.connection.execute(
            "SELECT rowid,bm25(visible_fts,2.0,1.0) score FROM temp.visible_fts WHERE visible_fts MATCH ?",
            (match,))], name)[:candidate_limit]
        # Both backends score the same float32 query vector over only the visible rows.
        query_blob = pack(embed(query))
        if self.vector_backend == "sqlite-vec":
            self.connection.execute("CREATE TEMP TABLE IF NOT EXISTS visible_vectors(id INTEGER PRIMARY KEY, embedding BLOB)")
            self.connection.execute("DELETE FROM temp.visible_vectors")
            self.connection.executemany("INSERT INTO temp.visible_vectors VALUES(?,?)", [(r["id"], r["embedding"]) for r in rows])
            distances = [(r["id"], r["distance"]) for r in self.connection.execute(
                "SELECT id,vec_distance_cosine(embedding,?) distance FROM temp.visible_vectors", (query_blob,))]
        else:
            query_vector = list(unpack(query_blob))
            distances = [(r["id"], cosine_distance(query_vector, unpack(r["embedding"]))) for r in rows]
        vectors = _competition_ranks([(i, d) for i, d in distances if d < 1.0 - TIE_TOLERANCE],
                                     name, TIE_TOLERANCE)[:candidate_limit]
        scores: dict[int, dict] = {}
        for method, rankings in [("fts5", lexical), ("vector", vectors)]:
            for row_id, raw_score, rank in rankings:
                detail = scores.setdefault(row_id, {"score": 0.0, "retrieval": {}})
                detail["score"] += 1.0 / (60 + rank)
                detail["retrieval"][method] = {"rank": rank, "raw_score": raw_score}
        ordered = sorted(scores, key=lambda i: (-scores[i]["score"], by_id[i]["memory_id"]))[:limit]
        response["results"] = [{**self._public(by_id[i]), **scores[i]} for i in ordered]
        self.connection.commit()  # end the temp-table transaction so no read snapshot stays open
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
