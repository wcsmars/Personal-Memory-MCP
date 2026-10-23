# Local verification

Run on 2026-10-23 with Python 3.11.7, SQLite 3.41.2, `mcp` 2.3.0 and `sqlite-vec` 0.1.9 installed through `pip install -e '.[all]'`:

```text
python -m pytest -q
86 passed
```

Without the optional extras (standard library only), the same suite gives `81 passed, 5 skipped`. The skipped tests need the official MCP SDK (SDK adapter round trips and the official client against the built-in server) or `sqlite-vec` (native backend and backend parity).

What the passing tests cover:

- Point-in-time retrieval: revisions, retroactive corrections, validity boundaries, tombstones and history at a cutoff.
- Ranking invariance: BM25 scores and fused ranks at a past cutoff are unchanged after adding future documents or documents in another namespace.
- Ranking details: tied scores share a rank regardless of insertion order; sqlite-vec and Python produce identical rankings on a randomized corpus; Unicode forms (ß, ligatures, decomposed accents, dotted İ) match in keyword search.
- Input handling: malformed, out-of-range and sub-microsecond times are rejected with a message; title, source and limit types are checked; `forget` is idempotent, records the revision it hides and refuses an already-forgotten memory.
- Read-only access: a write-protected database can be searched; a missing one is reported, not created.
- Markdown import: nesting, idempotence, shrink tombstones, size and encoding limits (with the file named), symlink rejection, byte-order marks, CRLF, heading extraction, whitespace-only chunk boundaries, title caps, and atomic failure, including a re-import that reuses a knowledge time.
- CLI and export round trips; the demo output matches `examples/demo-output.json`.
- The dependency-free MCP stdio server run as a real subprocess: version negotiation, lifecycle ordering, ignored client responses and blank lines, malformed and oversized messages (the 4 MiB limit is exact), read-only default, optional writes, historical search, namespace isolation and tool annotations.
- The official MCP SDK v2 client against both adapters: identical tool sets, annotations and bounds, structured results, and readable error messages for bad dates, missing memories and out-of-range arguments. The SDK launcher reports configuration errors in one line.
- The leakage benchmark: deterministic, the saved output is current, and the store matches the independent ground truth on every query.

Additional checks run once for this release (not part of the suite):

- Backend parity on 10,000 randomized searches over 220 generated databases: no ranking differences.
- 3,000 random documents through the chunker: no chunk over the limit, and no text other than whitespace lost or duplicated.
- Re-importing 2,000 unchanged files: 0.11 s; time grows linearly.
- A fresh clone from GitHub, installed into a new virtual environment, passes the suite; the built wheel installs and both entry points start.

GitHub Actions installs all extras and runs the suite on Python 3.11 and 3.12 on every push.
