import re
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Optional


class StatValidator:
    """Classify query results and attach statistical context before synthesis."""

    _ID_PATTERN = re.compile(r"\b(id|code|key|num|no|number|index)\b", re.IGNORECASE)

    def validate(self, df: Optional[pd.DataFrame], question: str) -> dict:
        if df is None or df.empty:
            return {"type": "empty", "interpretation": "No data returned."}
        kind = self._classify(df, question)
        ctx  = {"type": kind}
        ctx.update({
            "comparison":   self._comparison,
            "trend":        self._trend,
            "aggregate":    self._aggregate,
            "distribution": self._distribution,
            "raw":          self._raw,
        }.get(kind, self._raw)(df))
        return ctx

    def _pick(self, df: pd.DataFrame) -> Optional[str]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_id   = [c for c in num_cols if not self._ID_PATTERN.search(c)]
        return (non_id or num_cols or [None])[0]

    def _classify(self, df: pd.DataFrame, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["compare","vs","versus","higher","lower","differ","between","more than","less than"]):
            return "comparison"
        if any(k in q for k in ["trend","over time","monthly","daily","weekly","yearly","growth","change"]):
            return "trend"
        if any(k in q for k in ["spread","distribut","histogram","range","variance"]):
            return "distribution"
        if len(df.select_dtypes(include=[np.number]).columns) >= 1 and len(df) > 1:
            return "aggregate"
        return "raw"

    def _raw(self, df: pd.DataFrame) -> dict:
        col = self._pick(df)
        if not col:
            return {"interpretation": "No numeric values to summarise."}
        vals = df[col].dropna()
        if vals.empty:
            return {"interpretation": f"Column '{col}' has no non-null values."}
        v = float(vals.iloc[0]) if len(vals) == 1 else float(vals.mean())
        return {"n": len(vals), "mean": round(v, 4),
                "interpretation": f"{'Value' if len(vals)==1 else 'Mean'} of {col}: {v:,.2f} ({len(vals)} row(s))."}

    def _comparison(self, df: pd.DataFrame) -> dict:
        col = self._pick(df)
        if not col:
            return {"interpretation": "No numeric columns for comparison."}
        # Wide: single row, multiple numeric cols
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(df) == 1 and len(num_cols) >= 2:
            vals  = {c: float(df[c].iloc[0]) for c in num_cols[:4]}
            names = list(vals); v0, v1 = vals[names[0]], vals[names[1]]
            pct   = (v0 - v1) / (abs(v1) + 1e-9) * 100
            return {"test": "direct (single-row wide)", "values": vals,
                    "pct_difference": round(pct, 2),
                    "interpretation": f"{names[0]} ({v0:,.2f}) is {abs(pct):.1f}% {'higher' if pct>0 else 'lower'} than {names[1]} ({v1:,.2f}). No significance test on aggregated totals."}
        # Long: group by first categorical col
        groups = [g[col].dropna().values for _, g in df.groupby(df.columns[0])]
        if len(groups) < 2 or any(len(g) < 2 for g in groups[:2]):
            return {"note": "Need ≥2 values per group for t-test.",
                    "interpretation": "Aggregated totals only — statistical test not applicable."}
        _, p = sp_stats.ttest_ind(groups[0], groups[1], equal_var=False)
        ci   = sp_stats.t.interval(0.95, len(groups[0])-1, loc=np.mean(groups[0]), scale=sp_stats.sem(groups[0]))
        return {"test": "Welch's t-test", "p_value": round(float(p), 4),
                "significant": bool(p < 0.05), "ci_95": [round(float(ci[0]),2), round(float(ci[1]),2)],
                "interpretation": f"{'Significant' if p<0.05 else 'Not significant'} difference (p={p:.4f}). Group-1 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]."}

    def _trend(self, df: pd.DataFrame) -> dict:
        col = self._pick(df)
        if not col or len(df) < 3:
            return {"interpretation": "Need ≥3 rows for trend analysis."}
        y = df[col].dropna().values; x = np.arange(len(y), dtype=float)
        slope, _, r, p, _ = sp_stats.linregress(x, y)
        return {"test": "linear regression", "slope": round(float(slope), 4),
                "r_squared": round(float(r**2), 4), "p_value": round(float(p), 4),
                "significant": bool(p < 0.05),
                "interpretation": f"{'Significant' if p<0.05 else 'Non-significant'} {'upward' if slope>0 else 'downward'} trend (slope={slope:.4f}, R²={r**2:.3f}, p={p:.4f})."}

    def _aggregate(self, df: pd.DataFrame) -> dict:
        col = self._pick(df)
        if not col:
            return {"interpretation": "No numeric columns found."}
        vals = df[col].dropna().values
        if len(vals) < 2:
            v = float(vals[0]) if len(vals) else float("nan")
            return {"mean": round(v, 4), "interpretation": f"{col} = {v:,.2f} (single value)."}
        try:
            boot = sp_stats.bootstrap((vals,), np.mean, n_resamples=1000, confidence_level=0.95, random_state=42)
            lo, hi = boot.confidence_interval.low, boot.confidence_interval.high
            return {"test": "bootstrap CI (1000 resamples)", "mean": round(float(np.mean(vals)), 4),
                    "ci_95_low": round(float(lo), 4), "ci_95_high": round(float(hi), 4),
                    "interpretation": f"Mean {col} = {np.mean(vals):,.2f} (95% CI: {lo:,.2f}–{hi:,.2f})."}
        except Exception:
            mn, sd = float(np.mean(vals)), float(np.std(vals))
            return {"mean": round(mn, 4), "std": round(sd, 4),
                    "interpretation": f"Mean {col} = {mn:,.2f} (std={sd:,.2f})."}

    def _distribution(self, df: pd.DataFrame) -> dict:
        col = self._pick(df)
        if not col:
            return {"interpretation": "No numeric columns found."}
        vals = df[col].dropna().values
        if len(vals) < 8:
            return {"interpretation": "Need ≥8 values for normality test."}
        _, p = sp_stats.normaltest(vals)
        return {"test": "D'Agostino–Pearson", "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4), "skewness": round(float(sp_stats.skew(vals)), 4),
                "p_value": round(float(p), 4), "is_normal": bool(p > 0.05),
                "interpretation": f"{'Approximately normal' if p>0.05 else 'Not normal'} (p={p:.4f}), skew={sp_stats.skew(vals):.3f}."}
