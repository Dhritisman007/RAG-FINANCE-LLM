"""
Portfolio insight tests (all computed by code). No network.

    python tests/test_insights.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import insights  # noqa: E402


def test_allocation_and_concentration() -> None:
    rows = [{"symbol": "AAPL", "currency": "USD", "value_inr": 600_000, "pnl_pct": 90.0},
            {"symbol": "TCS", "currency": "INR", "value_inr": 300_000, "pnl_pct": -10.0},
            {"symbol": "INFY", "currency": "INR", "value_inr": 100_000, "pnl_pct": 5.0},
            {"symbol": "GONE", "currency": "INR", "value_inr": None, "pnl_pct": None}]   # no price: ignored
    a = insights.allocation(rows)
    assert a["largest"] == ("AAPL", 60.0) and a["top3"] == 100.0
    assert round(a["india_pct"]) == 40 and round(a["us_pct"]) == 60
    assert a["best"] == ("AAPL", 90.0) and a["worst"] == ("TCS", -10.0)
    assert round(a["effective_holdings"], 2) == round(1 / (0.36 + 0.09 + 0.01), 2)
    assert insights.allocation([]) == {}
    print("PASS: weights, India/US split, concentration and best/worst performers")


def test_holding_periods() -> None:
    hs = [{"ticker": "TCS.NS", "purchase_date": "2026-03-01"}, {"ticker": "AAPL", "purchase_date": "2024-03-15"},
          {"ticker": "JPM", "purchase_date": "2020-01-02"}, {"ticker": "INFY.NS", "purchase_date": ""}]
    g = insights.holding_periods(hs, today=date(2026, 10, 23))
    assert g == {"under 1 year": ["TCS"], "1–3 years": ["AAPL"], "over 3 years": ["JPM"], "date not given": ["INFY"]}
    print("PASS: holdings grouped by how long ago they were bought")


def test_portfolio_history_and_performance() -> None:
    days = [f"2026-01-{d:02d}" for d in range(1, 31)]
    hist = {"TCS.NS": [(d, 100 + i) for i, d in enumerate(days)],
            "AAPL": [(d, 10.0) for d in days if d != "2026-01-15"]}    # a US holiday: that day is skipped
    hs = [{"ticker": "TCS.NS", "quantity": 2}, {"ticker": "AAPL", "quantity": 1}, {"ticker": "NOHIST", "quantity": 5}]
    values = insights.portfolio_history(hs, hist, {"TCS.NS": 1.0, "AAPL": 90.0, "NOHIST": 1.0})
    assert len(values) == 29 and "2026-01-15" not in dict(values)
    assert values[0] == ("2026-01-01", 2 * 100 + 1 * 10.0 * 90) and values[-1][1] == 2 * 129 + 900
    p = insights.performance(values)
    assert round(p["return_pct"], 2) == round((1158 / 1100 - 1) * 100, 2) and p["max_drawdown"] == 0.0
    assert p["volatility"] > 0 and insights.performance(values[:5]) == {}
    print("PASS: value of today's holdings over past common trading days; return, volatility, drawdown")


def test_headlines_are_descriptive() -> None:
    alloc = {"largest": ("AAPL", 46.0), "top3": 100.0, "india_pct": 0.0, "us_pct": 100.0}
    perf = {"return_pct": 18.2, "max_drawdown": -22.0}
    lines = insights.headlines(alloc, perf, {"Nifty 50": 9.1, "S&P 500": 21.0}, {"under 1 year": ["JPM"]})
    text = " ".join(t for _, t in lines)
    assert "+18.2%" in text and "ahead of Nifty 50" in text and "AAPL is 46%" in text
    assert "US stocks" in text and "-22% below its high" in text and "JPM" in text
    assert "should" not in text.lower() and "buy" not in text.lower().replace("bought", "")
    assert dict((t, k) for k, t in lines)[next(t for _, t in lines if "AAPL" in t)] == "warn"
    print("PASS: plain-English observations, flagged by tone, never advice")


if __name__ == "__main__":
    test_allocation_and_concentration()
    test_holding_periods()
    test_portfolio_history_and_performance()
    test_headlines_are_descriptive()
    print("\nAll insight tests passed.")
