# StatQL: A Statistically Validated Data Analytics Agent with Semantic Caching

StatQL is an open-source, production-ready data analytics agent that bridges the gap between natural language and trustworthy enterprise insights. While standard LLM API wrappers blindly execute generated SQL and pass raw numbers back to the user, StatQL introduces a dedicated Statistical Validation Layer that automatically intercepts query results to run t-tests, fit linear regressions, and calculate bootstrap confidence intervals. By grounding the LLM's natural language responses in mathematical rigor and pairing it with a local, high-performance semantic caching architecture (DuckDB + FAISS), StatQL eliminates hallucinated metrics, quantifies uncertainty, and slashes API latency—transforming a brittle text-to-SQL script into a reliable, analytical decision-making engine.

## Architecture
<img width="8192" height="3845" alt="Customer Segmentation-2026-05-24-215854" src="https://github.com/user-attachments/assets/3abf4d4a-ec2f-474d-bfdd-f468e3169a58" />

```
User question
      │
      ▼
SemanticCache ──hit──▶ cached answer
      │ miss
      ▼
AnalyticsAgent (LangGraph)
  parse_question
      │
  generate_sql  ◀── retry (max 3×) ──┐
      │                               │
  execute_query ──── SQL error ───────┘
      │
  synthesize_answer
      │
  StatValidator → CI / p-value / bootstrap
      │
  hallucination_guard
      │
      ▼
    Answer + stats JSON
```

## What makes it different

**Statistical validation layer** — every answer includes either a 95% confidence interval (aggregates, bootstrap), a Welch's t-test p-value (comparisons), or a linear regression R² and slope (trends). The LLM is instructed to surface uncertainty explicitly. A hallucination guard flags numbers in the LLM response that can't be reconciled with raw query output.

**Semantic cache** — `all-MiniLM-L6-v2` embeddings + FAISS. Semantically similar questions (cosine ≥ 0.92) return cached answers without an LLM call.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/analytical-agent
cd analytical-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
echo "GROQ_API_KEY=your_key_here" > .env
echo "GROQ_MODEL=llama-3.3-70b-versatile" >> .env

# Place UCI Online Retail CSV at data/online_retail.csv

# 3. Start API
python main.py

# 4. Run benchmark
python main.py --benchmark
```

## API

```
POST /query          { "question": "Which country has the highest revenue?" }
GET  /cache/stats    hit rate, latency reduction, LLM calls saved
GET  /health
```

## Docker

```bash
docker build -t analytical-agent .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  analytical-agent
```

## Cache performance

Measured over 15 benchmark questions (7 unique + 8 semantic near-duplicates) across two runs. Dataset: UCI Online Retail (541,909 rows).

| Metric | Run 1 (cold) | Run 2 (warm) |
|---|---|---|
| Total queries | 15 | 15 |
| Cache hits | 7 | **14** |
| Cache misses | 8 | **1** |
| **Hit rate** | 46.7% | **93.3%** |
| Avg latency — cached | ~0.02 s | **0.059 s** |
| Avg latency — fresh | ~0.90 s | **1.353 s** |
| **Latency reduction** | ~97.7% | **95.6%** |
| LLM calls saved | 7 | **14** |

Similarity threshold: **0.92** cosine. Lowest hit similarity observed: **0.924**. The single Run 2 miss ("Show me revenue by country ranked highest to lowest") is a correct miss — it asks for all countries ranked, not the single top result.

> See [`results.md`](results.md) for full per-question breakdown and statistical output examples.

## Project structure

```
config.py          env vars
data/              DataLayer (DuckDB, Phase 2)
stats/             StatValidator (Phase 4)
agent/             AnalyticsAgent + LangGraph graph (Phase 3)
cache/             SemanticCache + CachedAgent (Phase 5)
api/               FastAPI app (Phase 6)
main.py            entrypoint
tests/             unit tests + prompt stability CI
golden_outputs.json  expected structural fields for CI diff
Dockerfile
.github/workflows/ci.yml
```
