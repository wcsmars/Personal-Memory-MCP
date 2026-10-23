"""The leakage benchmark is reproducible and the store matches its independent ground truth."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "temporal_leakage.py"


def load():
    spec = importlib.util.spec_from_file_location("temporal_leakage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_benchmark_shows_baseline_leaks_and_store_exactness():
    benchmark = load()
    report = benchmark.run(seed=3, identities=120, queries=60)
    assert report["queries"] > 40 and report["past_period_queries"] > 0
    assert set(report["counts"]["this_store"].values()) == {0}
    assert report["counts"]["latest_only"]["future_rows_returned"] > 0
    assert report["counts"]["global_top20"]["different_results"] > 0
    assert report["counts"]["global_top20"]["lost_results"] >= report["counts"]["global_topall"]["lost_results"] == 0
    assert benchmark.run(seed=3, identities=120, queries=60) == report


def test_saved_benchmark_output_is_current():
    import json

    saved = json.loads((SCRIPT.parent / "results" / "temporal_leakage.json").read_text(encoding="utf-8"))
    assert load().run(seed=saved["seed"]) == saved
