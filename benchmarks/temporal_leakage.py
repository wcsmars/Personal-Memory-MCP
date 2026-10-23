"""Measure how two common retrieval shortcuts leak future information.

A synthetic, seeded corpus of dated memory revisions is queried at historical
cutoffs three ways:

1. latest-only: keep only each memory's newest revision and search it.
   Any returned revision known after the cutoff is a direct future leak.
2. global index + post-filter: one FTS5 index over every revision; MATCH,
   take the top candidates, then drop rows not visible at the cutoff.
   BM25 statistics include future rows, and the candidate cut happens before
   the date filter, so rankings can differ from what was knowable.
3. scoped (this project): resolve visibility first, then rank only those rows.

Ground truth for (2) is BM25 over exactly the rows visible at the cutoff.
No network or model calls; the corpus is generated, not real data.

    PYTHONPATH=src python benchmarks/temporal_leakage.py
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from personal_memory.embedding import tokens
from personal_memory.store import MemoryStore, instant

TOPICS = {
    "markets": "equity bond yield spread volatility momentum value liquidity earnings dividend",
    "study": "statistics probability regression calculus exam practice feedback lecture notes",
    "career": "research trading analyst interview resume portfolio internship desk offer",
    "health": "running sleep diet gym stretching recovery hydration routine walking",
}
NAMESPACE = "bench"


def corpus(seed: int, identities: int) -> list[dict]:
    rng = random.Random(seed)
    start = date(2024, 1, 1)
    rows = []
    for index in range(identities):
        topic = rng.choice(sorted(TOPICS))
        words = TOPICS[topic].split()
        day = rng.randrange(0, 365)
        for revision in range(rng.choice([1, 1, 2, 3])):
            content = " ".join(rng.choice(words) for _ in range(rng.randrange(4, 12)))
            rows.append({"memory_id": f"{topic}-{index}", "title": topic, "content": content,
                         "known_at": (start + timedelta(days=day)).isoformat()})
            day += rng.randrange(20, 200)
    return rows


def fts_ranking(connection: sqlite3.Connection, table: str, query: str, limit: int) -> list[int]:
    match = " OR ".join('"' + term + '"' for term in dict.fromkeys(tokens(query)))
    return [row[0] for row in connection.execute(
        f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table},2.0,1.0),rowid LIMIT ?",
        (match, limit))]


def run(seed: int = 7, identities: int = 400, queries: int = 300, top_k: int = 5,
        candidates: int = 20) -> dict:
    rng = random.Random(seed + 1)
    rows = corpus(seed, identities)
    with MemoryStore(":memory:", vector_backend="python") as store:
        for row in rows:
            store.remember(namespace=NAMESPACE, **row)
        db = store.connection
        # Baseline index over every revision, built once like a conventional search index.
        db.execute("CREATE VIRTUAL TABLE temp.global_fts USING fts5(title,content)")
        db.execute("INSERT INTO global_fts(rowid,title,content) SELECT id,title,content FROM revisions")
        db.execute("CREATE VIRTUAL TABLE temp.truth_fts USING fts5(title,content)")
        newest = {}
        for row in db.execute("SELECT id,memory_id,known_at FROM revisions ORDER BY known_at,id"):
            newest[row["memory_id"]] = row
        known_at = {row["id"]: row["known_at"] for row in db.execute("SELECT id,known_at FROM revisions")}
        counts = {"latest_only_future_leak": 0, "post_filter_rank_mismatch": 0,
                  "post_filter_missing_results": 0, "scoped_rank_mismatch": 0, "queries_with_results": 0}
        for _ in range(queries):
            words = TOPICS[rng.choice(sorted(TOPICS))].split()
            query = " ".join(rng.sample(words, 2))
            cutoff = (date(2024, 3, 1) + timedelta(days=rng.randrange(0, 540))).isoformat()
            visible = store._visible_rows(namespace=NAMESPACE, as_of=cutoff)
            visible_ids = {row["id"] for row in visible}
            db.execute("DELETE FROM truth_fts")
            db.executemany("INSERT INTO truth_fts(rowid,title,content) VALUES(?,?,?)",
                           [(r["id"], r["title"], r["content"]) for r in visible])
            truth = fts_ranking(db, "truth_fts", query, top_k)
            if not truth:
                continue
            counts["queries_with_results"] += 1
            # 1. Latest-only store: newest revision per identity, no time filter.
            latest_ids = {row["id"] for row in newest.values()}
            ranked = [i for i in fts_ranking(db, "global_fts", query, 10_000) if i in latest_ids][:top_k]
            if any(known_at[i] > instant(cutoff) for i in ranked):
                counts["latest_only_future_leak"] += 1
            # 2. Global index, candidate cut, then visibility filter.
            filtered = [i for i in fts_ranking(db, "global_fts", query, candidates) if i in visible_ids][:top_k]
            if filtered != truth:
                counts["post_filter_rank_mismatch"] += 1
            if len(filtered) < len(truth):
                counts["post_filter_missing_results"] += 1
            # 3. This store's lexical component.
            result = store.search(query, namespace=NAMESPACE, as_of=cutoff, limit=100, candidate_limit=1000)
            lexical = sorted((r for r in result["results"] if "fts5" in r["retrieval"]),
                             key=lambda r: r["retrieval"]["fts5"]["rank"])
            if [r["id"] for r in lexical][:top_k] != truth:
                counts["scoped_rank_mismatch"] += 1
    total = counts["queries_with_results"]
    return {"synthetic": True, "seed": seed, "revisions": len(rows), "identities": identities,
            "queries": total, "top_k": top_k, "candidate_limit": candidates,
            "rates": {key: round(value / total, 4) for key, value in counts.items() if key != "queries_with_results"},
            "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = run(seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
