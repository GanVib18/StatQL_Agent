"""
Prompt stability test — runs golden questions against a live CachedAgent
and asserts structural completeness (not exact text).

Skipped automatically if GROQ_API_KEY is not set (safe for unit-test-only CI runs).
Set env var RUN_STABILITY=1 to force execution.
"""
import os
import json
import pytest
import pandas as pd

GOLDEN_PATH = "golden_outputs.json"
SKIP = not (os.getenv("GROQ_API_KEY") and os.getenv("RUN_STABILITY"))


@pytest.fixture(scope="module")
def cached_agent():
    if SKIP:
        pytest.skip("GROQ_API_KEY or RUN_STABILITY not set")
    import config
    from data.data_layer import DataLayer
    from agent.agent import AnalyticsAgent
    from cache.semantic_cache import SemanticCache, CachedAgent

    df    = pd.read_csv("data/online_retail.csv", encoding="latin-1")
    dl    = DataLayer(df, db_path=":memory:")
    agent = AnalyticsAgent(dl, groq_api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL)
    cache = SemanticCache(threshold=0.99, cache_path="/tmp/test_cache_meta.json",
                          index_path="/tmp/test_cache_index.faiss")
    return CachedAgent(agent, cache)


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def test_golden_outputs(cached_agent, golden):
    failures = []
    for spec in golden:
        q      = spec["question"]
        result = cached_agent.ask(q)

        for field in spec.get("required_fields", []):
            if not result.get(field):
                failures.append(f"[{q!r}] missing top-level field: {field!r}")

        stats = result.get("stats") or {}
        for field in spec.get("required_stats_fields", []):
            if field not in stats:
                failures.append(f"[{q!r}] stats missing field: {field!r}")

        if spec.get("answer_must_not_be_empty") and not (result.get("answer") or "").strip():
            failures.append(f"[{q!r}] answer is empty")

    if failures:
        pytest.fail("Prompt stability failures:\n" + "\n".join(f"  • {f}" for f in failures))
