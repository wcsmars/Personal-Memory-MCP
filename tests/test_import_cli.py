import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from personal_memory.markdown import import_markdown
from personal_memory.store import MemoryStore


def test_nested_import_idempotence_and_changed_content_history(tmp_path):
    root = tmp_path / "notes"
    (root / "career").mkdir(parents=True)
    note = root / "career" / "role.md"
    note.write_text("# Career\n\nEquity research")
    with MemoryStore(tmp_path / "store.db", vector_backend="python") as store:
        first = import_markdown(store, root, namespace="me", known_at="2025-01-01")
        assert first["written_revisions"] == 1
        repeated = import_markdown(store, root, namespace="me", known_at="2025-02-01")
        assert repeated["written_revisions"] == 0
        note.write_text("# Career\n\nQuantitative trading")
        import_markdown(store, root, namespace="me", known_at="2025-03-01")
        assert store.visible(namespace="me", as_of="2025-02-15")[0]["content"].endswith("Equity research")
        assert store.visible(namespace="me", as_of="2025-04-01")[0]["content"].endswith("Quantitative trading")
        assert store.visible(namespace="me", as_of="2025-04-01")[0]["source"]["path"] == "career/role.md"


def test_import_does_not_follow_symlinks_and_uses_import_time(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    outside = tmp_path / "secret.md"
    outside.write_text("private secret")
    (root / "link.md").symlink_to(outside)
    (root / "legit.md").write_text("legitimate note")
    os.utime(root / "legit.md", (0, 0))
    with MemoryStore(":memory:", vector_backend="python") as store:
        result = import_markdown(store, root, namespace="me")
        assert result["files"] == 1
        assert result["known_at"] > "2020-01-01"
        assert store.search("secret", namespace="me")["results"] == []


def test_malformed_import_is_atomic(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("should not survive a failed import")
    (root / "b.md").write_bytes(b"\x00")
    with MemoryStore(":memory:", vector_backend="python") as store:
        with pytest.raises(ValueError):
            import_markdown(store, root, namespace="me")
        assert store.stats()["revisions"] == 0


def test_cli_demo_and_export(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    prefix = [sys.executable, "-m", "personal_memory", "--db", str(tmp_path / "demo.db"), "--vector-backend", "python"]
    result = subprocess.run(prefix + ["demo"], capture_output=True, text=True, env=env, check=True)
    payload = json.loads(result.stdout)
    assert payload["synthetic"]
    assert "equity" in payload["before"]["results"][0]["content"]
    assert "trading" in payload["after"]["results"][0]["content"]
    export_path = tmp_path / "snapshot.json"
    subprocess.run(prefix + ["export", str(export_path), "--namespace", "demo", "--as-of", "2025-02-01"], env=env, capture_output=True, check=True)
    assert json.loads(export_path.read_text())["memories"][0]["content"].endswith("equity research roles.")


def test_reimport_with_reused_known_at_fails_atomically(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("first version", encoding="utf-8")
    (notes / "b.md").write_text("untouched", encoding="utf-8")
    with MemoryStore(tmp_path / "m.db", vector_backend="python") as store:
        import_markdown(store, notes, namespace="t", known_at="2025-01-01")
        (notes / "a.md").write_text("second version", encoding="utf-8")
        (notes / "c.md").write_text("new file", encoding="utf-8")
        with pytest.raises(ValueError, match="later --known-at"):
            import_markdown(store, notes, namespace="t", known_at="2025-01-01")
        assert store.stats()["revisions"] == 2
        later = import_markdown(store, notes, namespace="t", known_at="2025-01-02")
        assert later["written_revisions"] == 2 and later["unchanged_chunks"] == 1


def test_demo_output_matches_saved_example(tmp_path, capsys):
    from personal_memory.cli import main

    assert main(["--db", str(tmp_path / "demo.db"), "--vector-backend", "python", "demo"]) == 0
    saved = json.loads((Path(__file__).resolve().parents[1] / "examples" / "demo-output.json").read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == saved
