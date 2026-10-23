"""Independent temporal/import regression tests from a second implementation review."""

import sqlite3
import os
import stat

import pytest

from personal_memory.markdown import CHUNK_CHARS, import_markdown
from personal_memory.store import MemoryStore


@pytest.fixture
def store():
    with MemoryStore(":memory:", vector_backend="python") as result:
        yield result


def add(store, key, content, *, namespace="review", known="2025-01-01", **kwargs):
    return store.remember(namespace=namespace, memory_id=key, content=content,
                          known_at=known, **kwargs)


def test_historical_ranking_unaffected_by_future_or_other_namespace(store):
    add(store, "alpha", "SQLite memory storage and provenance")
    add(store, "beta", "SQLite research research research history")
    query = dict(namespace="review", as_of="2025-02-01", limit=2)
    before = store.search("SQLite memory research", **query)
    for index in range(40):
        add(store, f"future-{index}", "SQLite memory research", known="2025-05-01")
        add(store, f"other-{index}", "SQLite memory research", namespace="private-other")
    after = store.search("SQLite memory research", **query)
    assert after == before


def test_tombstone_does_not_erase_past_or_leave_stale_search_context(store):
    row = add(store, "role", "quantitative trading role")
    assert store.search("trading", namespace="review", as_of="2025-02-01")["results"]
    store.forget(namespace="review", memory_id="role", known_at="2025-03-01")
    assert not store.search("trading", namespace="review", as_of="2025-04-01")["results"]
    before = store.search("trading", namespace="review", as_of="2025-02-01")["results"]
    assert [r["id"] for r in before] == [row["id"]]
    assert len(store.history(namespace="review", memory_id="role", as_of="2025-02-01")) == 1


def test_scheduled_assertion_preserves_earlier_valid_time(store):
    add(store, "role", "research role")
    add(store, "role", "trading role", known="2025-02-01", valid_from="2025-06-01")
    assert store.visible(namespace="review", as_of="2025-03-01")[0]["content"] == "research role"
    assert store.visible(namespace="review", as_of="2025-03-01", valid_at="2025-07-01")[0]["content"] == "trading role"
    assert store.visible(namespace="review", as_of="2025-01-15", valid_at="2025-07-01")[0]["content"] == "research role"


def test_import_shrink_reimport_and_history(store, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("a" * (CHUNK_CHARS + 10), encoding="utf-8")
    first = import_markdown(store, tmp_path, namespace="review", known_at="2025-01-01")
    assert first["written_revisions"] == 2
    original = store.visible(namespace="review", as_of="2025-01-10")
    path.write_text("Short new note", encoding="utf-8")
    second = import_markdown(store, tmp_path, namespace="review", known_at="2025-02-01")
    assert second["written_revisions"] == 2
    assert len(store.visible(namespace="review", as_of="2025-03-01")) == 1
    assert store.visible(namespace="review", as_of="2025-01-10") == original
    count = store.stats()["revisions"]
    repeat = import_markdown(store, tmp_path, namespace="review", known_at="2025-02-01")
    assert repeat["written_revisions"] == 0
    assert store.stats()["revisions"] == count


def test_import_validity_change_is_not_deduplicated(store, tmp_path):
    (tmp_path / "note.md").write_text("A fictional project uses SQLite.", encoding="utf-8")
    import_markdown(store, tmp_path, namespace="review", known_at="2025-03-01", valid_from="2025-03-01")
    result = import_markdown(store, tmp_path, namespace="review", known_at="2025-04-01", valid_from="2025-02-01")
    assert result["written_revisions"] == 1
    assert store.visible(namespace="review", as_of="2025-05-01", valid_at="2025-02-15")
    assert not store.visible(namespace="review", as_of="2025-03-15", valid_at="2025-02-15")


def test_import_parse_error_is_atomic(store, tmp_path):
    (tmp_path / "a.md").write_text("First valid note", encoding="utf-8")
    (tmp_path / "z.md").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="not valid UTF-8: z.md"):
        import_markdown(store, tmp_path, namespace="review", known_at="2025-01-01")
    assert store.stats()["revisions"] == 0


def test_import_insert_conflict_rolls_back_earlier_rows(store, tmp_path):
    (tmp_path / "z.md").write_text("Old text", encoding="utf-8")
    import_markdown(store, tmp_path, namespace="review", known_at="2025-01-01")
    (tmp_path / "a.md").write_text("New file sorted first", encoding="utf-8")
    (tmp_path / "z.md").write_text("Conflicting text at same timestamp", encoding="utf-8")
    with pytest.raises((sqlite3.IntegrityError, ValueError)):
        import_markdown(store, tmp_path, namespace="review", known_at="2025-01-01")
    assert store.stats()["revisions"] == 1
    assert store.visible(namespace="review", as_of="2025-02-01")[0]["content"] == "Old text"


def test_import_skips_symlink_files_and_directories(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("Not authorized to import this", encoding="utf-8")
    root = tmp_path / "inside"
    root.mkdir()
    (root / "allowed.md").write_text("Allowed text", encoding="utf-8")
    (root / "linked.md").symlink_to(outside / "secret.md")
    (root / "linked-directory").symlink_to(outside, target_is_directory=True)
    result = import_markdown(store, root, namespace="review", known_at="2025-01-01")
    assert result["files"] == 1
    assert [r["content"] for r in store.visible(namespace="review", as_of="2025-02-01")] == ["Allowed text"]


def test_import_namespace_and_collection_are_distinct_identities(store, tmp_path):
    (tmp_path / "same.md").write_text("same source text", encoding="utf-8")
    for namespace, collection in [("review", "one"), ("review", "two"), ("another", "one")]:
        import_markdown(store, tmp_path, namespace=namespace, collection=collection, known_at="2025-01-01")
    assert len(store.visible(namespace="review", as_of="2025-02-01")) == 2
    assert len(store.visible(namespace="another", as_of="2025-02-01")) == 1


def test_equal_instants_normalized_before_identity_conflict(store):
    first = add(store, "time", "timestamp normalization", known="2025-01-01T08:00:00+08:00")
    second = add(store, "time", "timestamp normalization", known="2025-01-01T00:00:00Z")
    assert first == second
    with pytest.raises(ValueError, match="different revision"):
        add(store, "time", "conflicting content", known="2025-01-01")


def test_cli_supports_documented_search_limit(tmp_path, capsys):
    from personal_memory.cli import main

    database = str(tmp_path / "memory.db")
    assert main(["--db", database, "--vector-backend", "python", "search", "research",
                 "--namespace", "review"]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["--db", database, "init"]) == 0
    code = main(["--db", database, "--vector-backend", "python",
                 "search", "research", "--namespace", "review", "--limit", "100"])
    assert code == 0, capsys.readouterr().err


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO test requires POSIX")
def test_import_never_blocks_on_special_file(store, tmp_path, monkeypatch):
    (tmp_path / "allowed.md").write_text("Allowed text", encoding="utf-8")
    pipe = tmp_path / "pipe.md"
    os.mkfifo(pipe)
    original_open = os.open

    def guarded_open(path, flags, *args, **kwargs):
        if stat.S_ISFIFO(os.stat(path).st_mode) and not flags & os.O_NONBLOCK:
            pytest.fail("Importer would block opening a FIFO Markdown file")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    try:
        result = import_markdown(store, tmp_path, namespace="review", known_at="2025-01-01")
    except ValueError:
        return  # Explicit rejection is also a safe contract.
    assert result["files"] == 1
