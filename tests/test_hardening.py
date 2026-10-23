"""Regression tests for issues found in an independent review of the store and importer."""

import os
import random
import stat

import pytest

from personal_memory.markdown import CHUNK_CHARS, TITLE_CHARS, chunks, heading, import_markdown
from personal_memory.store import MAX_TITLE_CHARS, MemoryStore, NotFound, instant


@pytest.fixture
def store():
    with MemoryStore(":memory:", vector_backend="python") as value:
        yield value


def add(store, key, content, known="2025-01-01", **kwargs):
    return store.remember(namespace="n", memory_id=key, content=content, known_at=known, **kwargs)


# Time parsing --------------------------------------------------------------

@pytest.mark.parametrize("value", ["0001-01-01T00:00:00+05:00", "9999-12-31T23:00:00-05:00"])
def test_out_of_range_times_are_value_errors(value):
    with pytest.raises(ValueError, match="outside the supported range"):
        instant(value)


@pytest.mark.parametrize("value", ["20250101", "2025-01-01Z", "2025-W01-1", "", "2025-1-1", "not-a-date"])
def test_malformed_times_get_a_helpful_message(value):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        instant(value)


def test_sub_microsecond_input_is_rejected_not_truncated():
    with pytest.raises(ValueError, match="microsecond"):
        instant("2025-01-01T10:00:00.1234567Z")
    assert instant("2025-01-01T10:00:00.12345Z") == "2025-01-01T10:00:00.123450Z"


# Validation ----------------------------------------------------------------

def test_title_length_source_type_and_integer_limits_are_checked(store):
    with pytest.raises(ValueError, match="title"):
        add(store, "a", "text", title="t" * (MAX_TITLE_CHARS + 1))
    with pytest.raises(ValueError, match="source"):
        add(store, "a", "text", source=[])
    add(store, "a", "text")
    for limit in (2.0, True, "3"):
        with pytest.raises(ValueError, match="integers"):
            store.search("text", namespace="n", limit=limit)


# Forget ----------------------------------------------------------------------

def test_forget_is_idempotent_and_records_the_hidden_revision(store):
    current = add(store, "r", "open-ended value", valid_from="2023-01-01")
    add(store, "r", "bounded correction", known="2025-02-01", valid_from="2024-01-01", valid_to="2024-07-01")
    tombstone = store.forget(namespace="n", memory_id="r", known_at="2025-03-01")
    assert tombstone["source"]["previous_revision"] == current["id"]
    assert store.forget(namespace="n", memory_id="r", known_at="2025-03-01") == tombstone
    with pytest.raises(ValueError, match="already forgotten"):
        store.forget(namespace="n", memory_id="r", known_at="2025-04-01")
    # The bounded correction still answers questions about its own period.
    assert store.visible(namespace="n", as_of="2025-04-01", valid_at="2024-03-01")[0]["content"] == "bounded correction"


def test_missing_memory_error_has_a_clean_message(store):
    with pytest.raises(NotFound) as caught:
        store.forget(namespace="n", memory_id="missing")
    assert str(caught.value) == "No memory with that identity was known by the deletion time"
    assert isinstance(caught.value, KeyError)


# Retrieval -------------------------------------------------------------------

@pytest.mark.parametrize("document,query", [
    ("Die Straße ist lang", "Straße"),
    ("ﬁnance office", "finance"),
    ("finance office", "ﬁnance"),
    ("naïve plan", "naïve"),
    ("İstanbul trip", "İstanbul"),
])
def test_keyword_search_handles_unicode_forms(store, document, query):
    add(store, "doc", document)
    add(store, "other", "unrelated words entirely")
    results = store.search(query, namespace="n", as_of="2025-02-01")["results"]
    assert any(r["memory_id"] == "doc" and "fts5" in r["retrieval"] for r in results)


def test_ties_are_broken_by_memory_id_not_insertion_order(store):
    add(store, "z-memory", "identical text")
    add(store, "a-memory", "identical text")
    results = store.search("identical text", namespace="n", as_of="2025-02-01")["results"]
    assert [r["memory_id"] for r in results] == ["a-memory", "z-memory"]
    # Tied rows share a component rank, so neither ID nor insertion order changes the score.
    assert results[0]["score"] == results[1]["score"]
    assert results[0]["retrieval"] == results[1]["retrieval"]


def test_sqlite_vec_and_python_backends_rank_identically(tmp_path):
    pytest.importorskip("sqlite_vec")
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(30)] + "equity bond trading research".split()
    path = tmp_path / "parity.db"
    with MemoryStore(path, vector_backend="python") as store:
        for i in range(150):
            store.remember(namespace="n", memory_id=f"m{i:03d}", known_at="2025-01-01",
                           content=" ".join(rng.choice(vocab[:rng.randrange(4, len(vocab))])
                                            for _ in range(rng.randrange(1, 25))))
    queries = [" ".join(rng.sample(vocab, rng.randrange(1, 4))) for _ in range(60)]
    with MemoryStore(path, vector_backend="python") as py, MemoryStore(path, vector_backend="sqlite-vec") as native:
        assert native.vector_backend == "sqlite-vec"
        for query in queries:
            a = py.search(query, namespace="n", as_of="2025-02-01", limit=30)["results"]
            b = native.search(query, namespace="n", as_of="2025-02-01", limit=30)["results"]
            assert [(r["memory_id"], r["retrieval"].get("vector", {}).get("rank")) for r in a] == \
                   [(r["memory_id"], r["retrieval"].get("vector", {}).get("rank")) for r in b], query


# Read-only access --------------------------------------------------------

def test_read_only_store_reads_a_write_protected_file(tmp_path):
    path = tmp_path / "memory.db"
    with MemoryStore(path, vector_backend="python") as store:
        store.remember(namespace="n", memory_id="a", content="equity research", known_at="2025-01-01")
    os.chmod(path, stat.S_IRUSR)
    try:
        with MemoryStore(path, vector_backend="python", read_only=True) as store:
            assert store.search("equity", namespace="n")["results"]
            assert store.search("research", namespace="n")["results"]
            with pytest.raises(Exception, match="readonly"):
                store.remember(namespace="n", memory_id="b", content="x")
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def test_read_only_store_does_not_create_a_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        MemoryStore(tmp_path / "typo.db", read_only=True)
    assert not (tmp_path / "typo.db").exists()


# Markdown import -----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("```bash\n# install deps\n```\n\n# Real Title", "Real Title"),
    ("#todo buy milk\n\n# Groceries", "Groceries"),
    ("# #1 priority", "#1 priority"),
    ("#\n\nbody", "stem"),
    ("   # Indented heading ##", "Indented heading"),
    ("~~~\n# not this\n~~~\n## This one", "This one"),
    ("no heading at all", "stem"),
])
def test_heading_extraction(text, expected):
    assert heading(text, "stem") == expected


def test_chunks_cut_long_paragraphs_at_whitespace_and_keep_all_text():
    rng = random.Random(3)
    for _ in range(500):
        text = "".join(rng.choice(["word ", "longerword ", "\n", "\n\n", "\r\n", "\t", "x" * rng.randrange(1, 80)])
                       for _ in range(rng.randrange(0, 1500)))
        parts = chunks(text)
        assert all(0 < len(part) <= CHUNK_CHARS and part.strip() for part in parts)
        assert "".join("".join(parts).split()) == "".join(text.split())
    words = chunks("word " * 1000)
    assert all(part.endswith("word") for part in words)


def test_import_handles_bom_crlf_and_caps_titles(store, tmp_path):
    (tmp_path / "bom.md").write_bytes("﻿# Bom Title\r\n\r\nFirst para\r\nsame para".encode())
    (tmp_path / "long.md").write_text("# " + "word " * 5000 + "\n\nbody", encoding="utf-8")
    import_markdown(store, tmp_path, namespace="n", known_at="2025-01-01")
    rows = {row["source"]["path"]: row for row in store.visible(namespace="n", as_of="2025-02-01")}
    assert rows["bom.md"]["title"] == "Bom Title"
    assert rows["bom.md"]["content"] == "# Bom Title\n\nFirst para\nsame para"
    assert all(len(row["title"]) <= TITLE_CHARS for row in rows.values())


def test_import_names_a_file_that_is_not_utf8(store, tmp_path):
    (tmp_path / "bad.md").write_bytes(b"ok \xff")
    with pytest.raises(ValueError, match="not valid UTF-8: bad.md"):
        import_markdown(store, tmp_path, namespace="n", known_at="2025-01-01")
