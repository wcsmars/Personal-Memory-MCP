"""The leakage benchmark is reproducible and the scoped store matches ground truth."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "temporal_leakage.py"


def load():
    spec = importlib.util.spec_from_file_location("temporal_leakage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_benchmark_shows_baseline_leaks_and_scoped_exactness():
    report = load().run(seed=3, identities=120, queries=60)
    assert report["queries"] > 0
    assert report["counts"]["scoped_rank_mismatch"] == 0
    assert report["counts"]["latest_only_future_leak"] > 0
    assert report["counts"]["post_filter_rank_mismatch"] > 0
    assert load().run(seed=3, identities=120, queries=60) == report
