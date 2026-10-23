"""Explicit, bounded Markdown ingestion. File timestamps are never knowledge dates."""

import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path

from .store import MemoryStore, instant

MAX_BYTES = 2_000_000
CHUNK_CHARS = 1800


def chunks(text: str) -> list[str]:
    result: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > CHUNK_CHARS:
            result.append(current)
            current = ""
        while len(paragraph) > CHUNK_CHARS:
            if current:
                result.append(current)
                current = ""
            result.append(paragraph[:CHUNK_CHARS])
            paragraph = paragraph[CHUNK_CHARS:]
        if paragraph:
            current += ("\n\n" if current else "") + paragraph
    if current:
        result.append(current)
    return result


def import_markdown(store: MemoryStore, root: str | Path, *, namespace: str,
                    known_at: str | None = None, valid_from: str | None = None,
                    collection: str = "notes") -> dict:
    """Upsert content by collection/relative-path/chunk index; shrink tombstones old chunks.

    Missing files are deliberately not deleted. Deletion requires an explicit forget.
    Each import is atomic; a malformed file cannot leave half a snapshot committed.
    """
    store._scope(namespace)
    if not collection.strip():
        raise ValueError("collection must be nonempty")
    original = Path(root).expanduser()
    if original.is_symlink():
        raise ValueError("Markdown root must not be a symbolic link")
    base = original.resolve(strict=True)
    if not base.is_dir():
        raise ValueError("Markdown root must be a directory")
    known = instant(known_at)
    valid = instant(valid_from) if valid_from else known
    # Latest revision per identity known by this import's knowledge time.
    latest: dict[str, sqlite3.Row] = {}
    for row in store.connection.execute(
            "SELECT * FROM revisions WHERE namespace=? AND known_at<=? ORDER BY known_at DESC,id DESC",
            (namespace, known)):
        latest.setdefault(row["memory_id"], row)
    planned: list[dict] = []
    files = 0
    skipped = 0
    for directory, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if name.startswith(".") or path.suffix.lower() != ".md" or path.is_symlink() or not path.is_file():
                continue
            # Path resolution is checked, and O_NOFOLLOW guards the final open.
            path.resolve(strict=True).relative_to(base)
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(fd, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError(f"Markdown must be a regular file: {path.relative_to(base)}")
                data = handle.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES or b"\x00" in data:
                raise ValueError(f"Invalid or oversized Markdown: {path.relative_to(base)}")
            text = data.decode("utf-8")
            relative = path.relative_to(base).as_posix()
            files += 1
            title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), path.stem)
            parts = chunks(text)
            for index, content in enumerate(parts):
                memory_id = "md:" + hashlib.sha256(f"{collection}\0{relative}\0{index}".encode()).hexdigest()[:32]
                source = {"kind": "markdown", "collection": collection, "path": relative, "chunk": index}
                row = latest.get(memory_id)
                if (row and not row["deleted"] and row["content"] == content and row["title"] == title
                        and row["valid_to"] is None and (valid_from is None or row["valid_from"] == valid)):
                    skipped += 1
                    continue
                planned.append(dict(namespace=namespace, memory_id=memory_id, content=content, title=title,
                                    known_at=known, valid_from=valid, source=source))
            for memory_id, row in latest.items():
                source = json.loads(row["source_json"])
                if (not row["deleted"] and source.get("kind") == "markdown" and source.get("collection") == collection
                        and source.get("path") == relative and source.get("chunk", -1) >= len(parts)):
                    planned.append(dict(namespace=namespace, memory_id=memory_id, content="", title=row["title"],
                                        known_at=known, valid_from=valid, source=source, deleted=True))
    # Validate in an isolated store before one atomic destination transaction.
    validated = []
    with MemoryStore(":memory:", vector_backend="python") as staging:
        for item in planned:
            result = staging.remember(**item)
            validated.append(staging.connection.execute("SELECT * FROM revisions WHERE id=?", (result["id"],)).fetchone())
    fields = ["memory_id", "namespace", "title", "content", "content_hash", "known_at", "valid_from", "valid_to", "deleted", "source_json", "embedding"]
    statement = "INSERT INTO revisions(" + ",".join(fields) + ") VALUES(" + ",".join("?" * len(fields)) + ")"
    try:
        with store.connection:
            for row in validated:
                store.connection.execute(statement, tuple(row[key] for key in fields))
    except sqlite3.IntegrityError as exc:
        raise ValueError("A changed chunk already has a revision at this known_at; "
                         "use a later --known-at (nothing was imported)") from exc
    return {"files": files, "written_revisions": len(validated), "unchanged_chunks": skipped,
            "known_at": known, "valid_from": valid, "collection": collection}
