import pytest
import pandas as pd
from data.data_layer import DataLayer


@pytest.fixture
def dl(tmp_path):
    df = pd.DataFrame({
        "InvoiceNo":   ["536365", "536366", "536367"],
        "StockCode":   ["85123A", "71053",  "84406B"],
        "Description": ["CREAM HANGING HEART", "WHITE METAL LANTERN", "CREAM CUPID"],
        "Quantity":    [6, 6, 8],
        "InvoiceDate": ["12/1/2010 8:26", "12/1/2010 8:26", "12/1/2010 8:34"],
        "UnitPrice":   [2.55, 3.39, 2.75],
        "CustomerID":  [17850, 17850, 13047],
        "Country":     ["United Kingdom", "United Kingdom", "United Kingdom"],
    })
    return DataLayer(df, db_path=str(tmp_path / "test.duckdb"))


def test_row_count(dl):
    result = dl.execute_query("SELECT COUNT(*) AS n FROM retail")
    assert result["n"].iloc[0] == 3


def test_revenue_column(dl):
    result = dl.execute_query("SELECT Revenue FROM retail LIMIT 1")
    assert abs(result["Revenue"].iloc[0] - 6 * 2.55) < 0.01


def test_schema_contains_columns(dl):
    schema = dl.get_schema()
    assert "Quantity" in schema
    assert "UnitPrice" in schema
    assert "Revenue" in schema


def test_sample_rows(dl):
    rows = dl.sample_rows(2)
    assert len(rows) <= 3  # dataset only has 3 rows


def test_empty_result(dl):
    result = dl.execute_query("SELECT * FROM retail WHERE Quantity > 9999")
    assert result.empty


def test_destructive_drop_blocked(dl):
    with pytest.raises(ValueError, match="Destructive SQL blocked"):
        dl.execute_query("DROP TABLE retail")


def test_destructive_delete_blocked(dl):
    with pytest.raises(ValueError, match="Destructive SQL blocked"):
        dl.execute_query("DELETE FROM retail WHERE Quantity < 0")


def test_destructive_insert_blocked(dl):
    with pytest.raises(ValueError, match="Destructive SQL blocked"):
        dl.execute_query("INSERT INTO retail VALUES (1,2,3,4,5,6,7,8,9)")


def test_invalid_sql_raises(dl):
    with pytest.raises(RuntimeError):
        dl.execute_query("SELECT * FROM nonexistent_table_xyz")
