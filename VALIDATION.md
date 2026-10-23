# Local verification

Run on 2026-10-23 with Python 3.11.7, SQLite 3.41.2, `mcp` 2.3.0 and `sqlite-vec` 0.1.9 installed through `pip install -e '.[all]'`:

```text
python -m pytest -q
43 passed
```

Without the optional extras (standard library only), the same suite gives `41 passed, 2 skipped`. The two skipped tests are the official SDK subprocess round trip and the native sqlite-vec backend.

What the passing tests cover:

- Point-in-time retrieval: revisions, retroactive corrections, validity boundaries, tombstones and history at a cutoff.
- Ranking invariance: BM25 scores and fused ranks at a past cutoff are unchanged after adding future documents or documents in another namespace.
- Markdown import: nesting, idempotence, shrink tombstones, size and encoding limits, symlink rejection, and atomic failure, including a re-import that reuses a knowledge time.
- CLI and export round trips.
- The dependency-free MCP stdio server run as a real subprocess: protocol negotiation, lifecycle ordering, malformed and oversized messages, read-only default, optional writes, historical search and namespace isolation.
- The official MCP SDK v2 client talking to the SDK adapter over stdio, including structured tool results.
- The sqlite-vec `vec_distance_cosine` backend.
- The future-leakage benchmark: deterministic, with zero ranking mismatches for the scoped store.

`./run-demo.sh` and `benchmarks/temporal_leakage.py` both complete and match the outputs in `examples/demo-output.json` and `benchmarks/results/temporal_leakage.json`.

GitHub Actions installs all extras and runs the suite on Python 3.11 and 3.12 on every push.
