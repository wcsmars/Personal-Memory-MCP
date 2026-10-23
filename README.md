# Personal Memory MCP

[![tests](https://github.com/wcsmars/Personal-Memory-MCP/actions/workflows/tests.yml/badge.svg)](https://github.com/wcsmars/Personal-Memory-MCP/actions/workflows/tests.yml)

A local memory service for assistants that answers a question most memory stores cannot: **what did the system know at a given time?** Every fact is an immutable, dated assertion. Retrieval at a historical cutoff returns only what was known then, ranked as if later documents did not exist.

The point-in-time discipline is the same one a backtest needs. A store that only keeps the latest version, or that searches a global index and filters dates afterwards, leaks future information into past answers. This project measures that leak and removes it.

- SQLite storage with append-only revisions, two time axes, tombstones and source provenance
- Hybrid retrieval: SQLite FTS5 (BM25) plus exact cosine scoring, combined with reciprocal rank fusion
- Optional [sqlite-vec](https://alexgarcia.xyz/sqlite-vec/) native vector scoring, with an identical pure-Python fallback
- A dependency-free MCP stdio server, plus an adapter for the official MCP Python SDK v2
- Bounded, atomic Markdown import; JSON CLI; no model or network calls; fictional sample data only

## Results: future leakage in common shortcuts

[`benchmarks/temporal_leakage.py`](benchmarks/temporal_leakage.py) builds a seeded synthetic corpus (400 memories, 679 dated revisions across 2024–2025) and runs 300 queries at random historical cutoffs. Ground truth is BM25 computed over exactly the rows visible at each cutoff.

| Retrieval approach | Queries affected | What goes wrong |
| --- | ---: | --- |
| Latest version only, no time filter | **70.7%** | A top-5 result was not yet known at the cutoff |
| Global FTS index, top-20 candidates, then date filter | **37.3%** | Top-5 differs from what was knowable (corpus statistics and the candidate cut include future rows) |
| Same, counting only queries that lost valid results | 20.0% | Future rows filled the candidate list, so knowable rows never surfaced |
| This store: resolve visibility, then index and rank | **0.0%** | Matches ground truth on every query |

```bash
PYTHONPATH=src python3 benchmarks/temporal_leakage.py
```

Output is saved in [`benchmarks/results/temporal_leakage.json`](benchmarks/results/temporal_leakage.json). The corpus is synthetic and small; the rates show the mechanism, not a production estimate. A test reruns a smaller version and checks it is deterministic.

## Quick start

Python 3.11 or newer. The core has no third-party dependencies.

```bash
./run-demo.sh
```

The demo stores two versions of one fictional preference. The same memory ID returns "equity research" as of February 2025 and "quantitative trading" as of April 2025. The script uses `.venv/bin/python` when present, otherwise `python3.11` (override with `MEMORY_PYTHON`). Without the script:

```bash
PYTHONPATH=src python3.11 -m personal_memory --db local/demo.db --vector-backend python demo
```

Installed, with the native vector backend and the official SDK:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[all]'
.venv/bin/memory --db local/demo.db demo
.venv/bin/python -m pytest -q
```

## Data model: two times, immutable assertions

| Field | Meaning |
| --- | --- |
| `known_at` | When this assertion became available to the memory system |
| `valid_from`, `valid_to` | Half-open interval during which the assertion applies |
| `memory_id`, `namespace` | Identity of one fact and its retrieval scope |
| `source`, `content_hash` | Source reference and SHA-256 of the stored content |

A query takes a knowledge cutoff `as_of` and, optionally, a `valid_at` time (default: the cutoff). It selects assertions known by the cutoff whose validity interval contains `valid_at`; for each identity the newest of those wins.

```sql
WITH eligible AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY namespace, memory_id ORDER BY known_at DESC, id DESC) AS rn
  FROM revisions
  WHERE namespace = :namespace AND known_at <= :as_of
    AND valid_from <= :valid_at AND (valid_to IS NULL OR :valid_at < valid_to)
)
SELECT * FROM eligible WHERE rn = 1 AND deleted = 0;
```

This is an **interval overlay**: a bounded correction replaces earlier values only inside its interval, and an earlier open-ended assertion shows through outside it. A permanent change uses an open-ended assertion; removing a claim for a period uses an explicit tombstone. It is not a full bitemporal database with automatic interval splitting.

Triggers reject `UPDATE` and `DELETE` on the revisions table. `forget` appends a tombstone, so a query before the deletion still sees the old fact. **Forget is not secure erasure**; there is no purge operation.

All times are normalized to UTC with microsecond precision. Date-only input means 00:00:00 UTC; timestamps must carry a timezone.

## Retrieval pipeline

1. Apply namespace, `known_at` and valid-time filters and resolve versions and tombstones (the SQL above).
2. Copy only those rows into a temporary FTS5 table and rank with BM25. Future or out-of-scope documents cannot change corpus statistics.
3. Score the same rows by cosine similarity, using sqlite-vec `vec_distance_cosine` when installed, otherwise Python.
4. Fuse the two rankings with `sum(1 / (60 + rank))`, break ties by memory ID, and return each result with component ranks, raw scores, provenance and content hash.

The built-in embedding is a deterministic, signed 256-dimensional hash of words and adjacent word pairs. It is a reproducible **lexical baseline**, not a semantic embedding model, and works offline. The database records the embedding version and refuses to open with a different one. Vector search is an exact scan over visible rows, not an approximate index, which suits personal-scale collections rather than millions of rows.

## Architecture

```
src/personal_memory/
  store.py      SQLite schema, append-only writes, visibility query, hybrid search, export
  embedding.py  Deterministic hashed word/bigram vectors and cosine distance
  markdown.py   Bounded, atomic Markdown import with stable chunk IDs
  cli.py        JSON command-line interface
  stdio.py      Dependency-free MCP 2025-11-25 stdio server (tools subset)
  server.py     Launcher; --sdk selects the official MCP SDK v2 adapter
benchmarks/     Future-leakage benchmark and saved results
tests/          Temporal, ranking, import, CLI, stdio protocol and SDK round-trip tests
```

More detail on the schema, invariants and operating limits: [docs/architecture.md](docs/architecture.md).

## CLI

```bash
memory() { PYTHONPATH=src python3.11 -m personal_memory --db local/memory.db "$@"; }
memory import examples/notes --namespace demo
memory search "statistics practice" --namespace demo
memory remember preferred-role "I prefer equity research." --namespace personal --title "Career preference"
memory history preferred-role --namespace personal
memory export local/snapshot.json --namespace personal
```

Add `--as-of 2025-02-01` to `search`, `history` or `export` for a historical cutoff, and `--valid-at` to ask about an earlier period using what was known at the cutoff.

**Markdown import.** Recurses through visible `.md` files, skips symlinks, rejects invalid UTF-8 and NUL bytes, and caps files at 2 MB. Text is split into chunks of at most 1,800 characters with IDs derived from collection, relative path and chunk number. Re-importing unchanged chunks is a no-op, changes append revisions, and a shrunk document tombstones its surplus chunks. Missing or renamed files are not deleted automatically. Each import is validated in full before one transaction commits it.

`known_at` defaults to the actual import time; file modification dates are never treated as evidence of when something was known. `--known-at` on the CLI is a trusted assertion for historical datasets that the store cannot verify. MCP writes always use the server clock.

## MCP server

```bash
PYTHONPATH=src MEMORY_DB="$PWD/local/memory.db" MEMORY_NAMESPACE=personal \
  python3.11 -m personal_memory.server          # add --sdk for the official SDK adapter
```

[examples/mcp-config.json](examples/mcp-config.json) is a client configuration template; replace the placeholder paths. Each server process is pinned to one namespace.

| Tool | Behavior |
| --- | --- |
| `memory_search` | Search visible memories, optionally at historical dates |
| `memory_history` | Audit revisions known by a cutoff, including tombstones |
| `memory_status` | Namespace revision count, vector backend, write availability |
| `memory_remember` | Append an assertion (only with `MEMORY_ENABLE_WRITES=true`) |
| `memory_forget` | Append a current-time tombstone (only with writes enabled) |

The built-in server negotiates protocol `2025-11-25` and implements initialization, ping, `tools/list` and `tools/call` over newline-delimited JSON, with a 4 MiB message limit, strict argument validation and duplicate-key rejection. It does not implement HTTP transport, resources, prompts, tasks or cancellation. Both adapters share one set of tool descriptions and read-only hints and return structured content. There is no import tool and no SQL tool, and returned memory text is labelled as untrusted data. A local stdio server is not a hosted HTTPS connector; whether a given assistant client can use it depends on that client.

## Testing

```bash
.venv/bin/python -m pytest -q
```

43 tests cover temporal revisions, retroactive corrections, validity boundaries, deletion history, namespace isolation, BM25 and rank invariance when future documents are added, import bounds, idempotence and atomic failure, CLI round trips, export, real subprocess runs of the stdio server (lifecycle, malformed input, size limits, read-only default), the official SDK client against the SDK adapter, the sqlite-vec backend, and the benchmark. With the extras installed all 43 pass; without them the two optional-integration tests are skipped. GitHub Actions runs the full suite on Python 3.11 and 3.12. See [VALIDATION.md](VALIDATION.md).

## Limits

No model-based extraction, entity resolution or merging of contradictory notes; text is stored exactly as supplied. No encryption (protect the SQLite file with OS permissions and disk encryption), no secure erasure, no hosted service, and no multi-process coordination beyond SQLite WAL. Memory correctness alone cannot stop a consuming model from using knowledge it acquired elsewhere.

## References

- [MCP 2025-11-25 lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle) and [tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- [Official MCP Python SDK](https://py.sdk.modelcontextprotocol.io/run/)
- [sqlite-vec](https://alexgarcia.xyz/sqlite-vec/python.html)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)

## License

MIT. See [LICENSE](LICENSE).
