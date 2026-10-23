import json
import sqlite3

import pytest

from personal_memory.store import MemoryStore, instant


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory.db", vector_backend="python") as value:
        yield value


def put(store, key="role", content="equity research", known="2025-01-01", **kwargs):
    return store.remember(namespace="personal", memory_id=key, content=content, known_at=known, **kwargs)


def test_future_revision_and_namespace_never_enter_context(store):
    put(store)
    put(store, content="quantitative trading", known="2025-03-01")
    store.remember(namespace="secret", memory_id="role", content="secret trading role", known_at="2024-01-01")
    past = store.search("trading equity research", namespace="personal", as_of="2025-02-01")
    assert past["eligible_count"] == 1
    assert [x["content"] for x in past["results"]] == ["equity research"]
    assert store.visible(namespace="personal", as_of="2024-12-31") == []
    assert store.visible(namespace="personal", as_of="2025-04-01")[0]["content"] == "quantitative trading"


def test_independent_valid_and_known_times_and_half_open_interval(store):
    put(store, content="initial estimate", valid_from="2024-01-01")
    put(store, content="corrected estimate", known="2025-03-01", valid_from="2024-01-01", valid_to="2024-07-01")
    assert store.visible(namespace="personal", as_of="2025-02-01", valid_at="2024-04-01")[0]["content"] == "initial estimate"
    assert store.visible(namespace="personal", as_of="2025-04-01", valid_at="2024-04-01")[0]["content"] == "corrected estimate"
    assert store.visible(namespace="personal", as_of="2025-04-01", valid_at="2024-07-01")[0]["content"] == "initial estimate"


def test_deletion_has_historical_visibility_and_cannot_leak_future(store):
    put(store)
    store.forget(namespace="personal", memory_id="role", known_at="2025-03-01")
    assert store.search("equity", namespace="personal", as_of="2025-02-01")["results"]
    assert store.search("equity", namespace="personal", as_of="2025-04-01")["results"] == []
    assert store.visible(namespace="personal", as_of="2025-04-01", valid_at="2025-02-01")[0]["content"] == "equity research"
    assert len(store.history(namespace="personal", memory_id="role", as_of="2025-02-01")) == 1
    assert len(store.history(namespace="personal", memory_id="role", as_of="2025-04-01")) == 2


def test_historical_ranks_and_scores_invariant_to_future_corpus(store):
    put(store, key="a", content="research markets")
    put(store, key="b", content="research research financial markets")
    before = store.search("research markets", namespace="personal", as_of="2025-02-01")
    for i in range(25):
        put(store, key=f"future-{i}", content="research markets " * (i + 1), known="2025-03-01")
    after = store.search("research markets", namespace="personal", as_of="2025-02-01")
    assert before == after


def test_revisions_idempotent_and_immutable(store):
    original = put(store)
    assert put(store)["id"] == original["id"]
    with pytest.raises(ValueError, match="different revision"):
        put(store, content="another claim")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connection.execute("UPDATE revisions SET content='changed'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.connection.execute("DELETE FROM revisions")


def test_export_is_point_in_time_and_does_not_overwrite(store, tmp_path):
    put(store)
    put(store, content="future", known="2025-05-01")
    destination = tmp_path / "export.json"
    result = store.export(destination, namespace="personal", as_of="2025-02-01")
    assert result["memories"] == 1
    assert json.loads(destination.read_text())["memories"][0]["content"] == "equity research"
    with pytest.raises(FileExistsError):
        store.export(destination, namespace="personal")


def test_queries_are_literal_and_invalid_timestamps_rejected(store):
    put(store)
    assert store.search('" OR equity* - research', namespace="personal", as_of="2025-02-01")["results"]
    assert store.search("---", namespace="personal", as_of="2025-02-01")["results"] == []
    with pytest.raises(ValueError, match="timezone"):
        instant("2025-01-01T10:00:00")
    assert instant("2025-01-01T08:00:00+08:00") == instant("2025-01-01")
    with pytest.raises(ValueError):
        put(store, valid_from="2025-01-02", valid_to="2025-01-01")
    with pytest.raises(ValueError):
        store.search("equity", namespace="personal", limit=0)
    assert store.search("equity", namespace="personal", limit=100)


def test_namespace_match_is_exact(store):
    put(store)
    assert store.search("equity", namespace="person%", as_of="2025-02-01")["results"] == []
    assert store.search("equity", namespace="personal' OR 1=1--", as_of="2025-02-01")["results"] == []


def test_sqlite_vec_matches_python_ranking(tmp_path):
    pytest.importorskip("sqlite_vec")
    path = tmp_path / "vectors.db"
    with MemoryStore(path, vector_backend="sqlite-vec") as native:
        put(native, key="a", content="equity research")
        put(native, key="b", content="research company analysis")
        put(native, key="future", content="equity research", known="2025-03-01")
        result = native.search("equity research", namespace="personal", as_of="2025-02-01")
    with MemoryStore(path, vector_backend="python") as python:
        baseline = python.search("equity research", namespace="personal", as_of="2025-02-01")
    assert result["vector_backend"] == "sqlite-vec"
    assert [x["memory_id"] for x in result["results"]] == [x["memory_id"] for x in baseline["results"]]
    for a, b in zip(result["results"], baseline["results"]):
        assert a["retrieval"]["vector"]["raw_score"] == pytest.approx(b["retrieval"]["vector"]["raw_score"], abs=1e-6)
