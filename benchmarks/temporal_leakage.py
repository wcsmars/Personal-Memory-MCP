"""Measure how common retrieval shortcuts leak information from outside a historical cutoff.

A seeded synthetic corpus of dated memory revisions (with later revisions, bounded
corrections of past periods, deletions and a second namespace) is queried at random
historical cutoffs. Ground truth is computed independently of the store:

* visibility: a brute-force Python implementation of the documented rule (per memory, the
  newest revision known by ``as_of`` whose validity interval contains ``valid_at``,
  unless that revision is a tombstone);
* ranking: BM25 from a fresh FTS5 index containing exactly those visible rows, ties
  broken by memory id.

Compared approaches:

1. latest-only: a store that overwrites, so only each memory's final revision is searchable.
2. global index + post-filter: one index over every revision; take the top ``c`` matches,
   then drop rows that are not visible. Reported for several candidate cut-offs ``c``.
3. known-by-cutoff index + post-filter: rows known after the cutoff are excluded before
   indexing, but superseded and deleted revisions are not. This separates the effect of
   future rows from the effect of stale ones.
4. this store: ``MemoryStore.search``, checked against the ground truth.

No network or model calls; the corpus is generated, not real data.

    PYTHONPATH=src python3.11 benchmarks/temporal_leakage.py
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from personal_memory.store import MemoryStore, instant

TOPICS = {
    "markets": "equity bond yield spread volatility momentum value liquidity earnings dividend",
    "study": "statistics probability regression calculus exam practice feedback lecture notes",
    "career": "research trading analyst interview resume portfolio internship desk offer",
    "health": "running sleep diet gym stretching recovery hydration routine walking",
}
NAMESPACE, OTHER = "bench", "other"
CUTOFFS = (20, 50, 100, None)
START = date(2024, 1, 1)


def day(offset: int) -> str:
    return (START + timedelta(days=offset)).isoformat()


def text(rng: random.Random, topic: str) -> str:
    words = TOPICS[topic].split()
    return " ".join(rng.choice(words) for _ in range(rng.randrange(4, 12)))


def build(store: MemoryStore, rng: random.Random, identities: int) -> None:
    for namespace, count in ((NAMESPACE, identities), (OTHER, identities // 4)):
        for index in range(count):
            topic = rng.choice(sorted(TOPICS))
            memory_id, known = f"{topic}-{index}", rng.randrange(0, 365)
            for _ in range(rng.choice([1, 1, 2, 3])):
                store.remember(namespace=namespace, memory_id=memory_id, title=topic,
                               content=text(rng, topic), known_at=day(known))
                known += rng.randrange(20, 160)
            if rng.random() < 0.15:  # a later correction that applies only to an earlier period
                start = rng.randrange(0, known)
                store.remember(namespace=namespace, memory_id=memory_id, title=topic, content=text(rng, topic),
                               known_at=day(known + 1), valid_from=day(start),
                               valid_to=day(start + rng.randrange(10, 90)))
            if rng.random() < 0.10:
                store.forget(namespace=namespace, memory_id=memory_id, known_at=day(known + 30))


def visible_truth(rows: list[dict], as_of: str, valid_at: str) -> dict[str, dict]:
    """Brute-force visibility, written from the documented rule rather than the store's SQL."""
    best: dict[str, dict] = {}
    for row in rows:
        if (row["namespace"] == NAMESPACE and row["known_at"] <= as_of and row["valid_from"] <= valid_at
                and (row["valid_to"] is None or valid_at < row["valid_to"])):
            current = best.get(row["memory_id"])
            if current is None or (row["known_at"], row["id"]) > (current["known_at"], current["id"]):
                best[row["memory_id"]] = row
    return {key: row for key, row in best.items() if not row["deleted"]}


def index(db: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    db.execute(f"DROP TABLE IF EXISTS temp.{table}")
    db.execute(f"CREATE VIRTUAL TABLE temp.{table} USING fts5(title,content)")
    db.executemany(f"INSERT INTO temp.{table}(rowid,title,content) VALUES(?,?,?)",
                   [(r["id"], r["title"], r["content"]) for r in rows])


def search(db: sqlite3.Connection, table: str, names: dict[int, str], query: str) -> list[tuple[float, str, int]]:
    """(score, memory_id, row id) for every match in ``table``, best first.

    Scores within a relative 1e-9 of each other are ties ordered by memory id, so last-bit
    differences between platform maths libraries cannot change the comparison.
    """
    match = " OR ".join(f'"{term}"' for term in dict.fromkeys(query.split()))
    scored = sorted((score, names[rowid], rowid) for rowid, score in db.execute(
        f"SELECT rowid,bm25({table},2.0,1.0) FROM temp.{table} WHERE {table} MATCH ?", (match,)))
    keyed, group, previous = [], 0, None
    for score, name, rowid in scored:
        if previous is not None and score - previous > 1e-9 * max(abs(score), abs(previous)):
            group += 1
        previous = score
        keyed.append((group, name, rowid, score))
    return [(score, name, rowid) for _, name, rowid, score in sorted(keyed)]


def bm25(db: sqlite3.Connection, rows: list[dict], query: str) -> list[tuple[float, str, int]]:
    """Ranking from a fresh index containing exactly ``rows``."""
    index(db, "fresh_fts", rows)
    return search(db, "fresh_fts", {r["id"]: r["memory_id"] for r in rows}, query)


def compare(found: list[str], truth: list[str], counts: dict, key: str) -> None:
    counts[key]["different_results"] += set(found) != set(truth)
    counts[key]["different_order"] += found != truth
    counts[key]["lost_results"] += len(found) < len(truth)


def run(seed: int = 7, identities: int = 400, queries: int = 300, top_k: int = 5) -> dict:
    rng = random.Random(seed)
    with MemoryStore(":memory:", vector_backend="python") as store:
        build(store, rng, identities)
        db = store.connection
        rows = [dict(r) for r in db.execute("SELECT id,memory_id,namespace,title,content,known_at,valid_from,"
                                            "valid_to,deleted FROM revisions ORDER BY id")]
        final = {}
        for row in rows:
            if row["namespace"] == NAMESPACE:
                final[row["memory_id"]] = row  # ids ascend with insertion, so the last one wins
        latest_only = [r for r in final.values() if not r["deleted"]]
        keys = ["latest_only"] + [f"global_top{c or 'all'}" for c in CUTOFFS] + ["known_by_cutoff", "this_store"]
        counts = {k: {"different_results": 0, "different_order": 0, "lost_results": 0, "future_rows_returned": 0}
                  for k in keys}
        known_at = {r["id"]: r["known_at"] for r in rows}
        names = {r["id"]: r["memory_id"] for r in rows}
        index(db, "global_fts", rows)
        index(db, "latest_fts", latest_only)
        evaluated = past_period = 0
        for _ in range(queries):
            words = TOPICS[rng.choice(sorted(TOPICS))].split()
            query = " ".join(rng.sample(words, 2))
            cutoff = day(rng.randrange(60, 700))
            valid_at = cutoff if rng.random() < 0.8 else day(rng.randrange(0, 300))
            as_of, when = instant(cutoff), instant(valid_at)
            if when > as_of:
                when = as_of
            visible = visible_truth(rows, as_of, when)
            truth_ranked = bm25(db, list(visible.values()), query)
            truth = [name for _, name, _ in truth_ranked[:top_k]]
            if not truth:
                continue
            evaluated += 1
            past_period += when < as_of
            visible_ids = {r["id"] for r in visible.values()}

            ranked = [(name, rowid) for _, name, rowid in search(db, "latest_fts", names, query)][:top_k]
            compare([n for n, _ in ranked], truth, counts, "latest_only")
            counts["latest_only"]["future_rows_returned"] += any(known_at[i] > as_of for _, i in ranked)

            everything = search(db, "global_fts", names, query)
            for cut in CUTOFFS:
                kept = [name for _, name, rowid in everything[:cut] if rowid in visible_ids][:top_k]
                compare(kept, truth, counts, f"global_top{cut or 'all'}")

            known = [r for r in rows if r["namespace"] == NAMESPACE and r["known_at"] <= as_of]
            kept = [name for _, name, rowid in bm25(db, known, query) if rowid in visible_ids][:top_k]
            compare(kept, truth, counts, "known_by_cutoff")

            result = store.search(query, namespace=NAMESPACE, as_of=as_of, valid_at=when, limit=100,
                                  candidate_limit=1000)
            lexical = sorted((r for r in result["results"] if "fts5" in r["retrieval"]),
                             key=lambda r: (r["retrieval"]["fts5"]["rank"], r["memory_id"]))
            compare([r["memory_id"] for r in lexical][:top_k], truth, counts, "this_store")
            truth_scores = {name: score for score, name, _ in truth_ranked}
            wrong = (result["eligible_count"] != len(visible)
                     or any(r["id"] not in visible_ids for r in result["results"])
                     or any(abs(r["retrieval"]["fts5"]["raw_score"] - truth_scores[r["memory_id"]])
                            > 1e-9 * abs(truth_scores[r["memory_id"]])
                            for r in lexical))
            counts["this_store"]["different_results"] += wrong
    rates = {k: {m: round(v / evaluated, 4) for m, v in c.items()} for k, c in counts.items()}
    return {"synthetic": True, "seed": seed, "revisions": len(rows), "identities": identities,
            "queries": evaluated, "past_period_queries": past_period, "top_k": top_k,
            "corpus_span": [min(r["known_at"] for r in rows)[:10], max(r["known_at"] for r in rows)[:10]],
            "rates": rates, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = run(seed=args.seed)
    text_report = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text_report + "\n", encoding="utf-8")
    print(text_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
