# Personal Memory MCP

[![tests](https://github.com/wcsmars/Personal-Memory-MCP/actions/workflows/tests.yml/badge.svg)](https://github.com/wcsmars/Personal-Memory-MCP/actions/workflows/tests.yml)

A local memory service for assistants that answers a question most memory stores cannot: **what did the system know at a given time?** Every fact is an immutable, dated assertion. Retrieval at a historical cutoff returns only what was known then, ranked as if no other version of the corpus existed.

The point-in-time discipline is the same one a backtest needs. A store that only keeps the latest version, or that searches one global index and filters dates afterwards, lets information from outside the cutoff into past answers. This project measures that leak and removes it.

- SQLite storage with append-only revisions, two time axes, tombstones and source provenance
- Hybrid retrieval: SQLite FTS5 (BM25) plus exact cosine scoring, combined with reciprocal rank fusion
- Optional [sqlite-vec](https://alexgarcia.xyz/sqlite-vec/) native vector scoring; the pure-Python fallback produces the same rankings
- A dependency-free MCP stdio server, plus an adapter for the official MCP Python SDK v2
- Bounded, atomic Markdown import; JSON CLI; no model or network calls; fictional sample data only

## Results: leakage in common shortcuts

[`benchmarks/temporal_leakage.py`](benchmarks/temporal_leakage.py) builds a seeded synthetic corpus: 400 memories plus a second namespace, 982 revisions dated 2024-01-01 to 2025-11-03, including later revisions, bounded corrections of past periods and deletions. It runs 299 queries at random historical cutoffs (42 of them ask about an earlier period than the cutoff).

Ground truth is computed independently of the store: a brute-force Python implementation of the visibility rule below, then BM25 from a fresh FTS5 index over exactly the visible rows. A query counts as affected when its top 5 contains different memories from the ground truth.

| Retrieval approach | Top 5 differs | Detail |
| --- | ---: | --- |
| Latest version only (an overwriting store) | **95.7%** | 59.9% of queries return a revision not yet known at the cutoff |
| One global index, top 20 matches, then filter | **39.5%** | 30.1% lose results entirely: invisible rows fill the candidate list |
| Same, top 50 / top 100 matches | 20.1% / 16.1% | 7.4% / 3.0% lose results |
| Same, every match kept, then filter | 13.0% | No truncation: BM25 statistics still count invisible rows |
| Index of rows known by the cutoff, then filter | 10.0% | No future rows, but superseded and deleted versions still distort BM25 |
| **This store**: resolve visibility, then index and rank | **0.0%** | Same memories, same order and identical BM25 scores on every query |

Future rows are not the only problem. Even with no future rows indexed, superseded and deleted versions shift corpus statistics enough to change one query in ten.

```bash
PYTHONPATH=src python3.11 benchmarks/temporal_leakage.py
```

Full output, including order-only differences, is in [`benchmarks/results/temporal_leakage.json`](benchmarks/results/temporal_leakage.json). The corpus is synthetic and small, so the rates show the mechanism, not a production estimate. Tests rerun a smaller version, check it is deterministic, and check the saved output is current.

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

Triggers reject `UPDATE` and `DELETE` on the revisions table. `forget` appends a tombstone that records which revision it hides, so a query before the deletion still sees the old fact. Repeating an identical `forget` is a no-op, and forgetting a memory that is already hidden at that time is an error. **Forget is not secure erasure**; there is no purge operation.

All times are normalized to UTC with microsecond precision. Accepted forms are `YYYY-MM-DD` (meaning 00:00:00 UTC) and ISO 8601 timestamps with a timezone; anything else, including out-of-range or sub-microsecond values, is rejected with a message rather than silently adjusted.

## Retrieval pipeline

1. Apply namespace, `known_at` and valid-time filters and resolve versions and tombstones (the SQL above).
2. Copy only those rows into a temporary FTS5 table and rank with BM25. Rows outside the cutoff cannot change corpus statistics. Indexed text and queries are Unicode-normalized (NFKC), and query words are passed to FTS5 without case folding, so forms such as "Straße", "ﬁnance" or decomposed accents match their stored text.
3. Score the same rows by cosine similarity, using sqlite-vec `vec_distance_cosine` when installed, otherwise Python.
4. Rank each component with tied scores sharing a rank, fuse with `sum(1 / (60 + rank))`, order by fused score then memory ID, and return each result with component ranks, raw scores, provenance and content hash.

sqlite-vec computes in 32-bit floats and Python in 64-bit, so equal cosine scores can differ by about 1e-7 between them. Distances within 1e-6 of each other are therefore treated as ties. With that rule, the two backends returned identical rankings on 10,000 randomized searches; a test checks this on every run.

The built-in embedding is a deterministic, signed 256-dimensional hash of words and adjacent word pairs. It is a reproducible **lexical baseline**, not a semantic embedding model, and works offline. The database records the embedding version and refuses to open with a different one. Vector search is an exact scan over visible rows, not an approximate index, which suits personal-scale collections rather than millions of rows.

## Architecture

```
src/personal_memory/
  store.py      SQLite schema, append-only writes, visibility query, hybrid search, export
  embedding.py  Deterministic hashed word/bigram vectors and cosine distance
  markdown.py   Bounded, atomic Markdown import with stable chunk IDs
  cli.py        JSON command-line interface
  stdio.py      Dependency-free MCP stdio server (tools subset) and shared server settings
  server.py     Launcher; --sdk selects the official MCP SDK v2 adapter
benchmarks/     Leakage benchmark and saved results
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

Add `--as-of 2025-02-01` to `search`, `history` or `export` for a historical cutoff, and `--valid-at` to ask about an earlier period using what was known at the cutoff. Read-only commands (`search`, `history`, `export`, `stats`) open the database read-only and report a missing file instead of creating one; `init`, `import` and `remember` create it. Errors are printed as JSON on stderr with exit code 2, and `export` never overwrites an existing file.

**Markdown import.** Recurses through visible `.md` files, skips symlinks, rejects NUL bytes and invalid UTF-8 (naming the file), and caps files at 2 MB. Line endings are normalized and a leading byte-order mark is dropped. Text is split at blank lines into chunks of at most 1,800 characters; a longer paragraph is cut at the last whitespace before the limit. Only whitespace between paragraphs is changed. The title is the first ATX heading (`#` to `######`) outside fenced code, or the file name, capped at 200 characters.

Chunk IDs derive from collection, relative path and chunk number. Re-importing unchanged chunks is a no-op, changes append revisions, and a shrunk document tombstones its surplus chunks. Missing or renamed files are not deleted automatically, and re-importing a file restores a chunk that was forgotten while the file still contains it. Each import is validated in full before one transaction commits it, and re-import time grows linearly with the number of files.

`known_at` defaults to the actual import time; file modification dates are never treated as evidence of when something was known. `--known-at` on the CLI is a trusted assertion for historical datasets that the store cannot verify. MCP writes always use the server clock.

## MCP server

```bash
PYTHONPATH=src MEMORY_DB="$PWD/local/memory.db" MEMORY_NAMESPACE=personal \
  python3.11 -m personal_memory.server          # add --sdk for the official SDK adapter
```

[examples/mcp-config.json](examples/mcp-config.json) is a client configuration template; replace the placeholder paths. Use an absolute `MEMORY_DB`: a relative one resolves against the client's working directory. Each server process is pinned to one namespace.

| Tool | Behavior |
| --- | --- |
| `memory_search` | Search visible memories, optionally at historical dates |
| `memory_history` | Audit revisions known by a cutoff, including tombstones |
| `memory_status` | Namespace revision count, vector backend, write availability |
| `memory_remember` | Append an assertion (only with `MEMORY_ENABLE_WRITES=true`) |
| `memory_forget` | Append a current-time tombstone (only with writes enabled) |

Without `MEMORY_ENABLE_WRITES=true` the server registers only the three read tools and opens the database read-only, so it works on a write-protected file and reports a mistyped path instead of creating an empty database. Invalid configuration, such as an unknown vector backend or `sqlite-vec` requested but not installed, stops the server at startup with a one-line message.

The built-in server negotiates protocol `2025-11-25` (and echoes `2025-06-18` to clients that request it). It implements initialization, ping, `tools/list` and `tools/call` over newline-delimited JSON. It has a 4 MiB message limit, rejects duplicate JSON keys, validates tool arguments strictly, and ignores unknown request fields and client responses. It does not implement HTTP transport, resources, prompts, tasks or cancellation.

Both adapters share tool descriptions, annotations (read-only, destructive and idempotent hints), argument bounds and the untrusted-data instructions. Both return structured content, and both report bad dates or missing memories as tool errors with the reason, so the model can correct itself. Two SDK behaviors differ: the SDK adapter ignores unknown arguments (the namespace and knowledge time stay pinned either way), and it reports an unknown tool name as a tool error rather than a protocol error. There is no import tool and no SQL tool. A local stdio server is not a hosted HTTPS connector; whether a given assistant client can use it depends on that client.

## Testing

```bash
.venv/bin/python -m pytest -q
```

86 tests cover:

- temporal behavior: revisions, retroactive corrections, validity boundaries, deletion history, namespace isolation, and BM25 and rank invariance when future documents are added;
- edge cases: time parsing, Unicode search and tie handling;
- backend parity between sqlite-vec and Python;
- Markdown import: bounds, headings, chunking, idempotence and atomic failure;
- the CLI and read-only access;
- the stdio server as a real subprocess: lifecycle, version negotiation, malformed input, size limits and read-only default;
- the official MCP client against both adapters;
- the benchmark.

With the extras installed all 86 pass; without them the 5 tests that need `mcp` or `sqlite-vec` are skipped. GitHub Actions runs the full suite on Python 3.11 and 3.12. See [VALIDATION.md](VALIDATION.md).

## Limits

- **No inference:** there is no model-based extraction, entity resolution or merging of contradictory notes.
- **Text is kept as written:** `remember` stores text exactly as supplied; imports change only whitespace between paragraphs and line endings.
- **Chinese and Japanese text:** keyword search uses the FTS5 `unicode61` tokenizer, which treats an unspaced run of CJK characters as one word, so a partial phrase matches only through the vector component, if at all.
- **Security:** there is no encryption (protect the SQLite file with OS permissions and disk encryption) and no secure erasure.
- **Deployment:** there is no hosted service and no multi-process coordination beyond SQLite WAL.
- **MCP writes and the clock:** MCP writes assume the server clock does not step backwards.
- **Outside knowledge:** memory correctness alone cannot stop a consuming model from using knowledge it acquired elsewhere.

## References

- [MCP 2025-11-25 lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle) and [tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- [Official MCP Python SDK](https://py.sdk.modelcontextprotocol.io/run/)
- [sqlite-vec](https://alexgarcia.xyz/sqlite-vec/python.html)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)

## License

MIT. See [LICENSE](LICENSE).
