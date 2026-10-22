"""
Holdings editor validation tests.

    python tests/test_portfolio_store.py
"""
import math
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from portfolio_store import add_holding, load_holdings, save_holdings, validate_holdings  # noqa: E402

NAN = float("nan")


def test_valid_rows_are_cleaned_and_saved() -> None:
    rows = [{"ticker": " aapl ", "quantity": 25, "avg_cost": 182.5, "purchase_date": date(2024, 3, 15)},
            {"ticker": "brk.b", "quantity": 2.5, "avg_cost": 0, "purchase_date": None},
            {"ticker": "", "quantity": NAN, "avg_cost": NAN, "purchase_date": None}]   # empty editor row
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "holdings.csv"
        assert save_holdings(rows, path) == []
        saved = load_holdings(path)
    assert [h["ticker"] for h in saved] == ["AAPL", "BRK.B"]
    assert saved[0]["purchase_date"] == "2024-03-15" and saved[1]["quantity"] == 2.5
    print("PASS: valid rows are normalised and saved; the editor's empty row is ignored")


def test_invalid_rows_are_rejected_and_nothing_is_written() -> None:
    bad = [{"ticker": "AAPL", "quantity": NAN, "avg_cost": NAN, "purchase_date": None},     # NaN from empty cells
           {"ticker": "B@D", "quantity": -2, "avg_cost": "x", "purchase_date": "2030-01-01"},
           {"ticker": "AAPL", "quantity": 1, "avg_cost": 1, "purchase_date": "15/03/2024"},
           {"ticker": "MSFT", "quantity": math.inf, "avg_cost": 1, "purchase_date": None}]
    _, errors = validate_holdings(bad)
    text = " | ".join(errors)
    for expected in ["Row 1: shares", "Row 1: average cost", "Row 2: 'B@D'", "Row 2: shares", "Row 2: average cost",
                     "Row 2: purchase date 2030-01-01 is in the future", "Row 3: AAPL appears twice",
                     "Row 3: purchase date must be YYYY-MM-DD", "Row 4: shares"]:
        assert expected in text, (expected, errors)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "holdings.csv"
        path.write_text("ticker,quantity,avg_cost,purchase_date\nAAPL,25,182.50,2024-03-15\n")
        before = path.read_text()
        assert save_holdings(bad, path)
        assert path.read_text() == before, "a rejected save must not touch the file"
    print("PASS: NaN/blank/negative/inf numbers, bad tickers, duplicates and bad or future dates are rejected; file untouched")


def test_add_holding_new_and_merge() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "holdings.csv"
        path.write_text("ticker,quantity,avg_cost,purchase_date\nAAPL,25,182.50,2024-03-15\n")
        assert add_holding({"ticker": "nvda", "quantity": 3, "avg_cost": 120, "purchase_date": ""}, path) == []
        assert add_holding({"ticker": "AAPL", "quantity": 5, "avg_cost": 300, "purchase_date": "2023-01-02"}, path) == []
        h = {x["ticker"]: x for x in load_holdings(path)}
        assert list(h) == ["AAPL", "NVDA"] and h["NVDA"]["quantity"] == 3
        assert h["AAPL"]["quantity"] == 30 and h["AAPL"]["avg_cost"] == round((25 * 182.5 + 5 * 300) / 30, 2)
        assert h["AAPL"]["purchase_date"] == "2023-01-02"                        # earlier lot's date kept
        before = path.read_text()
        assert add_holding({"ticker": "AAPL", "quantity": None, "avg_cost": 10, "purchase_date": ""}, path) == \
            ["shares must be a number greater than 0."]
        assert add_holding({"ticker": "", "quantity": None, "avg_cost": None, "purchase_date": ""}, path)
        assert path.read_text() == before
    print("PASS: add a new stock; buying more merges lots at the weighted average cost; bad input saves nothing")


if __name__ == "__main__":
    test_add_holding_new_and_merge()
    test_valid_rows_are_cleaned_and_saved()
    test_invalid_rows_are_rejected_and_nothing_is_written()
    print("\nAll portfolio store tests passed.")
