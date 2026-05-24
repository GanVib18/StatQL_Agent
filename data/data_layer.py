import re
import numpy as np
import pandas as pd
import duckdb
from typing import Optional


class DataLayer:
    """Load a DataFrame into DuckDB; expose safe read-only query helpers."""

    _BLOCKED = re.compile(
        r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE|REPLACE)\b",
        re.IGNORECASE,
    )
    _ID_PATTERN = re.compile(r"\b(id|code|key|num|no|number|index)\b", re.IGNORECASE)

    def __init__(self, df: pd.DataFrame, db_path: str = "retail.duckdb"):
        self.conn = duckdb.connect(db_path)
        self._ingest(df)

    def _ingest(self, df: pd.DataFrame):
        df = df.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["Quantity"]    = pd.to_numeric(df["Quantity"],    errors="coerce")
        df["UnitPrice"]   = pd.to_numeric(df["UnitPrice"],   errors="coerce")
        df["Revenue"]     = df["Quantity"] * df["UnitPrice"]
        self.conn.execute("DROP TABLE IF EXISTS retail")
        self.conn.register("_src", df)
        self.conn.execute("CREATE TABLE retail AS SELECT * FROM _src")
        self.conn.unregister("_src")
        print(f"✓ DataLayer: {len(df):,} rows → DuckDB table `retail`")

    def execute_query(self, sql: str) -> pd.DataFrame:
        if self._BLOCKED.search(sql):
            raise ValueError(f"Destructive SQL blocked: {sql[:100]}")
        try:
            return self.conn.execute(sql).df()
        except duckdb.Error as e:
            raise RuntimeError(str(e)) from e

    def get_schema(self) -> str:
        schema = self.conn.execute("DESCRIBE retail").df()
        lines  = ["Table: retail", "Columns:"]
        for _, row in schema.iterrows():
            lines.append(f"  - {row['column_name']}  ({row['column_type']})")
        lines.append("\nExample rows (3):")
        lines.append(self.conn.execute("SELECT * FROM retail LIMIT 3").df().to_string(index=False))
        return "\n".join(lines)

    def sample_rows(self, n: int = 5) -> pd.DataFrame:
        return self.conn.execute(f"SELECT * FROM retail USING SAMPLE {n}").df()

    def pick_metric_col(self, df: pd.DataFrame) -> Optional[str]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_id   = [c for c in num_cols if not self._ID_PATTERN.search(c)]
        return (non_id or num_cols or [None])[0]
