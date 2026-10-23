"""Command-line access with JSON outputs and explicit storage paths."""

import argparse
import json
import sqlite3
import sys

from .markdown import import_markdown
from .store import MemoryStore


def demo(store: MemoryStore) -> dict:
    store.remember(namespace="demo", memory_id="preferred-role", title="Career preference",
                   content="Fictional Alex prefers equity research roles.", known_at="2025-01-10",
                   source={"kind": "synthetic", "scenario": "initial preference"})
    store.remember(namespace="demo", memory_id="preferred-role", title="Career preference",
                   content="Fictional Alex now prefers quantitative trading roles.", known_at="2025-03-10",
                   source={"kind": "synthetic", "scenario": "changed preference"})
    return {"synthetic": True, "before": store.search("career role preference", namespace="demo", as_of="2025-02-01"),
            "after": store.search("career role preference", namespace="demo", as_of="2025-04-01"),
            "stats": store.stats()}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Local point-in-time personal memory (UTC dates/times)")
    result.add_argument("--db", default="local/memory.db", help="SQLite path; default local/memory.db")
    result.add_argument("--vector-backend", choices=["auto", "python", "sqlite-vec"], default="auto")
    subs = result.add_subparsers(dest="command", required=True)
    subs.add_parser("init", help="Create database and show capabilities")
    subs.add_parser("stats", help="Show storage and vector backend")
    subs.add_parser("demo", help="Insert/retrieve fictional dated memories in the demo namespace")
    imported = subs.add_parser("import", help="Import bounded chunks from a local Markdown directory")
    imported.add_argument("directory")
    imported.add_argument("--namespace", required=True)
    imported.add_argument("--collection", default="notes")
    imported.add_argument("--known-at", help="Defaults to import time, never mtime")
    imported.add_argument("--valid-from")
    remembered = subs.add_parser("remember", help="Append one assertion or revision")
    remembered.add_argument("memory_id")
    remembered.add_argument("content")
    remembered.add_argument("--namespace", required=True)
    remembered.add_argument("--title", default="")
    remembered.add_argument("--known-at")
    remembered.add_argument("--valid-from")
    remembered.add_argument("--valid-to")
    search = subs.add_parser("search")
    search.add_argument("query")
    search.add_argument("--namespace", required=True)
    search.add_argument("--as-of")
    search.add_argument("--valid-at")
    search.add_argument("--limit", type=int, default=8)
    history = subs.add_parser("history")
    history.add_argument("memory_id")
    history.add_argument("--namespace", required=True)
    history.add_argument("--as-of")
    forgotten = subs.add_parser("forget", help="Append a tombstone; retained history is not erased")
    forgotten.add_argument("memory_id")
    forgotten.add_argument("--namespace", required=True)
    forgotten.add_argument("--known-at")
    forgotten.add_argument("--valid-from")
    forgotten.add_argument("--valid-to")
    exported = subs.add_parser("export", help="Export visible JSON snapshot to a NEW file")
    exported.add_argument("path")
    exported.add_argument("--namespace", required=True)
    exported.add_argument("--as-of")
    exported.add_argument("--valid-at")
    return result


def main(argv: list[str] | None = None) -> int:
    args = vars(parser().parse_args(argv))
    db = args.pop("db")
    backend = args.pop("vector_backend")
    command = args.pop("command")
    try:
        with MemoryStore(db, vector_backend=backend) as store:
            if command in {"init", "stats"}:
                output = store.stats()
            elif command == "demo":
                output = demo(store)
            elif command == "import":
                output = import_markdown(store, root=args.pop("directory"), **args)
            else:
                output = getattr(store, command)(**args)
        print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
