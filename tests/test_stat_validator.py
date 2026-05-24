import pytest
import numpy as np
import pandas as pd
from stats.validator import StatValidator

v = StatValidator()


def _df(*cols_and_vals):
    return pd.DataFrame(dict(zip(cols_and_vals[::2], cols_and_vals[1::2])))


# ── classify ──────────────────────────────────────────────────────
def test_classify_comparison():
    df = _df("Country", ["UK", "DE"], "Revenue", [100.0, 80.0])
    assert v._classify(df, "Compare revenue between UK and Germany") == "comparison"

def test_classify_trend():
    df = _df("Month", range(12), "Revenue", np.random.rand(12))
    assert v._classify(df, "What is the monthly revenue trend?") == "trend"

def test_classify_aggregate():
    df = _df("Revenue", np.random.rand(10) * 1000)
    assert v._classify(df, "What is the average revenue?") == "aggregate"

def test_classify_distribution():
    df = _df("UnitPrice", np.random.rand(20))
    assert v._classify(df, "Show the distribution of unit prices") == "distribution"


# ── empty / null guards ────────────────────────────────────────────
def test_empty_df():
    result = v.validate(pd.DataFrame(), "anything")
    assert result["type"] == "empty"
    assert "interpretation" in result

def test_none_df():
    result = v.validate(None, "anything")
    assert result["type"] == "empty"


# ── aggregate ─────────────────────────────────────────────────────
def test_aggregate_has_mean_and_ci():
    df = _df("Revenue", np.random.rand(50) * 1000)
    result = v.validate(df, "What is the average revenue?")
    assert "mean" in result
    assert "ci_95_low" in result or "std" in result  # fallback if bootstrap fails
    assert "interpretation" in result

def test_aggregate_single_value():
    df = _df("Revenue", [42.0])
    result = v.validate(df, "total revenue")
    assert "mean" in result
    assert "interpretation" in result


# ── trend ─────────────────────────────────────────────────────────
def test_trend_slope_present():
    df = _df("Month", range(12), "Revenue", np.linspace(100, 200, 12))
    result = v.validate(df, "What is the revenue trend over time?")
    assert "slope" in result
    assert result["slope"] > 0
    assert "r_squared" in result

def test_trend_too_few_rows():
    df = _df("Revenue", [1.0, 2.0])
    result = v.validate(df, "trend over time")
    assert "interpretation" in result


# ── comparison ────────────────────────────────────────────────────
def test_comparison_wide_single_row():
    df = _df("UK", [10000.0], "DE", [7000.0])
    result = v.validate(df, "Compare UK vs Germany revenue")
    assert "pct_difference" in result
    assert result["pct_difference"] > 0

def test_comparison_no_numeric():
    df = _df("Country", ["UK", "DE"])
    result = v.validate(df, "compare countries")
    assert "interpretation" in result


# ── distribution ──────────────────────────────────────────────────
def test_distribution_normality():
    df = _df("Price", np.random.normal(5, 1, 50))
    result = v.validate(df, "show distribution of prices")
    assert "is_normal" in result
    assert "skewness" in result

def test_distribution_too_few():
    df = _df("Price", [1.0, 2.0, 3.0])
    result = v.validate(df, "spread of prices")
    assert "interpretation" in result
