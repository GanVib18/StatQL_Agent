"""
main.py — entrypoint for the Analytical AI Agent.

Usage:
    python main.py                    # starts API on port 8000
    python main.py --benchmark        # runs benchmark questions and prints analytics
"""
import argparse
import pandas as pd
import uvicorn

import config
from data.data_layer import DataLayer
from agent.agent import AnalyticsAgent
from cache.semantic_cache import SemanticCache, CachedAgent
from api.app import build_app


BENCHMARK_QUESTIONS = [
    "Which country generates the most total revenue?",
    "What is the monthly revenue trend over time?",
    "Compare revenue between the United Kingdom and Germany.",
    "What are the top 5 best-selling products by quantity?",
    "Which country generates the highest revenue?",      # near-duplicate → cache hit
    "What country has the highest sales revenue?",       # near-duplicate → cache hit
    "Show me revenue by country ranked highest to lowest",
    "Is UK revenue higher than Germany?",                # near-duplicate → cache hit
    "How does Germany compare to the UK in revenue?",
    "What products sold the most units?",                # near-duplicate → cache hit
    "What is the average order value per customer?",
    "Which month had the highest revenue?",
    "How many unique customers made purchases?",
    "What percentage of revenue comes from the UK?",
]


def build_cached_agent() -> CachedAgent:
    df    = pd.read_csv("data/online_retail.csv", encoding="latin-1")
    dl    = DataLayer(df, db_path=config.DB_PATH)
    agent = AnalyticsAgent(dl, groq_api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL)
    cache = SemanticCache(threshold=config.CACHE_THRESH,
                          cache_path=config.CACHE_META,
                          index_path=config.CACHE_INDEX)
    return CachedAgent(agent, cache)


def run_benchmark(ca: CachedAgent):
    import json
    for q in BENCHMARK_QUESTIONS:
        r   = ca.ask(q)
        tag = "✓ HIT" if r.get("cache_hit") else "✗ MISS"
        sim = f"  sim={r['similarity']:.3f}" if r.get("cache_hit") else ""
        print(f"\n[{tag}]{sim}  {r['latency_s']:.2f}s")
        print(f"  Q:      {q}")
        print(f"  SQL:    {(r.get('sql') or '')[:120]} …")
        print(f"  Stats:  {(r.get('stats') or {}).get('interpretation', '—')}")
        print(f"  Answer: {(r.get('answer') or '')[:250]}")
    print("\n── Cache analytics ──")
    print(json.dumps(ca.cache.get_analytics(), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark questions")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ca = build_cached_agent()

    if args.benchmark:
        run_benchmark(ca)
    else:
        app = build_app(ca)
        print(f"✓ API → http://localhost:{args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
