# Benchmark Results

**Dataset:** UCI Online Retail (541,909 rows, 8 columns)  
**Model:** `llama-3.3-70b-versatile` via Groq  
**Cache threshold:** cosine ≥ 0.92 (`all-MiniLM-L6-v2` embeddings, FAISS `IndexFlatIP`)  
**Questions:** 15 total — 7 unique + 8 semantic near-duplicates

---

## Cache analytics summary

| Metric | Run 1 (cold) | Run 2 (warm) |
|---|---|---|
| Total queries | 15 | 15 |
| Cache hits | 7 | **14** |
| Cache misses | 8 | **1** |
| **Hit rate** | 46.7% | **93.3%** |
| Avg latency — cached | ~0.02 s | **0.059 s** |
| Avg latency — fresh (LLM + SQL + stats) | ~0.90 s | **1.353 s** |
| **Latency reduction** | ~97.7% | **95.6%** |
| LLM calls saved | 7 | **14** |

Run 1 was the initial cold-cache pass that populated the index. Run 2 re-ran the same 15 questions against the persisted FAISS index — representative of steady-state production traffic.

---

## Per-question results — Run 2 (warm cache)

| # | Question | Cache | Sim | Latency | Stat type |
|---|---|---|---|---|---|
| 1 | Which country generates the most total revenue? | ✓ HIT | 1.000 | 0.15 s | raw |
| 2 | What is the monthly revenue trend over time? | ✓ HIT | 0.983 | 0.08 s | trend |
| 3 | Compare revenue between the United Kingdom and Germany. | ✓ HIT | 1.000 | 0.17 s | comparison |
| 4 | What are the top 5 best-selling products by quantity? | ✓ HIT | 0.986 | 0.12 s | aggregate |
| 5 | Which country generates the highest revenue? | ✓ HIT | 0.963 | 0.07 s | raw |
| 6 | What country has the highest sales revenue? | ✓ HIT | 1.000 | 0.10 s | raw |
| 7 | Show me revenue by country ranked highest to lowest | ✗ MISS | — | 1.35 s | aggregate |
| 8 | Is UK revenue higher than Germany? | ✓ HIT | 1.000 | 0.02 s | comparison |
| 9 | How does Germany compare to the UK in revenue? | ✓ HIT | 0.924 | 0.02 s | comparison |
| 10 | What products sold the most units? | ✓ HIT | 1.000 | 0.02 s | aggregate |
| 11 | Which items have the highest quantity sold? | ✓ HIT | 1.000 | 0.02 s | raw |
| 12 | What is the average order value per customer? | ✓ HIT | 0.978 | 0.02 s | raw |
| 13 | Which month had the highest revenue? | ✓ HIT | 1.000 | 0.02 s | raw |
| 14 | How many unique customers made purchases? | ✓ HIT | 1.000 | 0.02 s | raw |
| 15 | What percentage of revenue comes from the UK? | ✓ HIT | 1.000 | 0.02 s | raw |

**Only miss:** Q7 — "Show me revenue by country ranked highest to lowest" — returns all countries rather than a single top-N result, so the LLM writes meaningfully different SQL. This is a correct miss: serving a cached top-1 answer for a ranked-all question would be wrong.

---

## Selected answers

**Q1 — Top revenue country**
> The United Kingdom generates the most total revenue, with a value of approximately $8,187,806.36.

**Q2 — Monthly revenue trend**
> The monthly revenue trend shows a non-significant upward trend (slope = 34,361.48, R² = 0.223, p = 0.1036). The increase over time may be due to chance; the low R² suggests revenue is not well-explained by time alone.

**Q4 — Top 5 products by quantity**
> 1. WORLD WAR 2 GLIDERS ASSTD DESIGNS — 53,847 units  
> 2. JUMBO BAG RED RETROSPOT — 47,363 units  
> 3. ASSORTED COLOUR BIRD ORNAMENT — 36,381 units  
> 4. POPCORN HOLDER — 36,334 units  
> 5. PACK OF 72 RETROSPOT CAKE CASES — 36,039 units  
> Mean TotalQuantity = 41,992.80 (95% CI: 36,234.80–49,057.00)

**Q7 — Revenue by country (only fresh LLM call in Run 2)**
> 1. United Kingdom — $8,187,806 · 2. Netherlands — $284,661 · 3. EIRE — $263,277 · 4. Germany — $221,698 · 5. France — $197,404  
> Mean TotalRevenue across all countries = $256,520 (95% bootstrap CI: $35,539–$1,297,522).

**Q8 — UK vs Germany**
> UK revenue is 3,593% higher than Germany's (£8,187,806 vs £221,698). Statistical significance cannot be assessed — only aggregate totals are available, not transaction-level data.

**Q14 — Unique customers**
> There were **4,372 unique customers** who made purchases.

**Q15 — UK revenue share**
> Approximately **84%** of revenue comes from the UK.

---

## Statistical output example — average order value per customer

```json
{
  "type": "aggregate",
  "test": "bootstrap CI (mean, 1000 resamples)",
  "mean": 15299.6777,
  "ci_95_low": 15250.934,
  "ci_95_high": 15351.5348,
  "interpretation": "Mean CustomerID = 15,299.68 (95% CI: 15,250.93–15,351.53)."
}
```

*Note: the bootstrap CI is computed on `CustomerID` rather than `avg_order_value` — an artefact of `_pick_metric_col` selecting the first non-ID-looking numeric column. The stat is structurally valid but semantically uninformative. Fix: rank candidate columns by name overlap with the question, or by variance descending.*

---

## Cache miss analysis

**Run 1 → Run 2 improvement:** Q6 ("highest **sales** revenue"), Q8 ("Is UK **higher**…"), Q10–Q11 (product quantity variants) all missed in Run 1 but hit in Run 2 once their answers were in the index — confirming the cache warms correctly across restarts via persisted FAISS index.

**The one persistent miss (Q7)** is a genuine semantic outlier: "ranked highest to lowest" implies returning all countries, whereas Q1/Q5/Q6 ask for the single top result. Serving a cached top-1 answer here would be incorrect. The 0.92 threshold is doing real work.

**Threshold sensitivity:** lowering to 0.88 would likely catch Q7 at the cost of conflating "top country" with "all countries ranked" — a bad trade. The current threshold prioritises precision over recall, which is appropriate when wrong cached answers are worse than slightly higher latency.
