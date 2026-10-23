# Storage and query contract

See also the [future-leakage benchmark](../benchmarks/temporal_leakage.py), which compares this design against latest-only and post-filtered retrieval.

The core is Python standard library code. `store.py` owns SQLite storage and retrieval; `markdown.py` handles explicit imports; `cli.py` is the JSON CLI. `server.py` launches the built-in `stdio.py` adapter by default, or its optional official SDK v2 adapter with `--sdk`. The SDK is imported only for the latter. sqlite-vec is discovered at store initialization.

## Persistent schema

`metadata` stores schema and embedding versions. `revisions` contains:

```sql
id INTEGER PRIMARY KEY AUTOINCREMENT,
memory_id TEXT NOT NULL,
namespace TEXT NOT NULL,
title TEXT NOT NULL,
content TEXT NOT NULL,
content_hash TEXT NOT NULL,
known_at TEXT NOT NULL,
valid_from TEXT NOT NULL,
valid_to TEXT,
deleted INTEGER NOT NULL,
source_json TEXT NOT NULL,
embedding BLOB NOT NULL,
UNIQUE(namespace, memory_id, known_at)
```

All dates are normalized to UTC with microsecond precision before insertion, so lexical comparisons have timestamp order. A check enforces `valid_to > valid_from`. Indexes cover scope/time and identity/time. Update/delete triggers reject mutation of the assertion table. Administrative access to the SQLite file can bypass triggers, so this is an application invariant, not tamper-proof storage.

An identical write to the same namespace, identity, and knowledge timestamp is idempotent. A different assertion at the same timestamp is rejected. Later knowledge appends a new row, including for deletes. SQLite transactions make each write atomic; an import validates all files before committing its revisions in one transaction. WAL and a busy timeout support ordinary local readers/writers; there is no multi-process distributed coordination.

## Visibility SQL

```sql
WITH eligible AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY namespace, memory_id
    ORDER BY known_at DESC, id DESC
  ) AS rn
  FROM revisions
  WHERE namespace = :namespace
    AND known_at <= :as_of
    AND valid_from <= :valid_at
    AND (valid_to IS NULL OR :valid_at < valid_to)
)
SELECT * FROM eligible
WHERE rn = 1 AND deleted = 0;
```

This is deliberately an interval-overlay representation. A later assertion only supersedes the portion of valid time it covers. A bounded overlay ending July 1 does not erase an earlier open-ended assertion after July 1. Explicit tombstones are used to remove unwanted intervals. A new preference normally uses an open-ended interval starting when it changed. A correction about a historical period can start before `known_at`, but it is inaccessible at an earlier knowledge cutoff.

`as_of` controls known information; `valid_at` controls the period being discussed. `valid_at` defaults to `as_of`. History also respects the knowledge cutoff, but includes obsolete versions and tombstones; it must not be treated as the ordinary retrieval result.

## Why temporary retrieval indexes

An FTS5 MATCH query against a global corpus followed by a date filter can still leak future corpus statistics into BM25. The implementation constructs `temp.visible_fts` from the already resolved visible rows. Namespace and date exclusions therefore precede both document selection and ranking statistics.

Vector embeddings are similarly copied to `temp.visible_vectors`. Native mode runs `vec_distance_cosine` only on that table; fallback mode computes the same cosine in Python. Zero/nonpositive similarities are excluded. Top candidates from each method are combined using `sum(1 / (60 + rank))`; ties are ordered by memory ID. Results contain component ranks, raw scores, source references, timestamps, and a content hash.

There is no LLM in this retrieval loop. A consuming assistant can still produce unsupported claims. Memory correctness alone cannot prove the model did not use future knowledge from elsewhere.

## Operational boundaries

- Each MCP process is pinned to one configured namespace; tool callers cannot supply another one.
- The server offers no file-import tool and no arbitrary SQL tool.
- MCP write tools are only registered when explicitly enabled at launch.
- Knowledge timestamps on MCP writes come from the server clock. Trusted historical ingestion through the CLI can provide dates, which the software cannot authenticate.
- Markdown traversal excludes hidden entries and symlinks, verifies resolved paths remain under the import root, and uses `O_NOFOLLOW` for the final file open when supported. It is not a sandbox against a hostile local process concurrently swapping parent directories.
- Plaintext SQLite is not encrypted. Protect it using operating-system permissions and disk encryption as appropriate.
- Tombstones are logical deletions. Secure erasure/export encryption are outside this implementation.
- The dependency-free adapter negotiates MCP protocol `2025-11-25`. It implements initialization, initialized notifications, ping, tools/list, and tools/call; it does not implement HTTP, resources, prompts, tasks, or concurrent cancellation. Input is newline-delimited UTF-8 JSON, limited to 4 MiB per message. Only protocol messages go to stdout. Malformed JSON, invalid calls, and tool execution failures produce the corresponding protocol or tool error response.
- The optional SDK adapter targets MCP SDK major version 2 (tested with 2.3.0); the dependency is pinned below 3. It reuses the built-in adapter's tool descriptions and read-only hints and requests structured output explicitly, because SDK v2 does not infer it from a bare `dict` return type.
