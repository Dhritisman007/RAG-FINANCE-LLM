"""
Portfolio store — the user's own holdings, looked up as structured data.

Holdings are NOT embedded into Chroma: ticker / quantity / cost basis are exact
facts, so they're read directly from data/portfolio/holdings.csv and injected
into the prompt alongside retrieved filing chunks (see rag_loop.py).

PRIVACY: this data never leaves the machine. It is only ever read here and
sent to the local Ollama server — never to a cloud API. holdings.csv is
git-ignored so real positions are not committed.

Usage:
    python scripts/portfolio_store.py          # print current holdings
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
HOLDINGS_PATH = ROOT / "data" / "portfolio" / "holdings.csv"


def load_holdings(path: Path = HOLDINGS_PATH) -> list[dict]:
    """Read holdings.csv -> [{ticker, quantity, avg_cost, purchase_date}, ...]."""
    if not path.exists():
        return []
    holdings = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("ticker", "").strip():
                continue
            holdings.append({
                "ticker": row["ticker"].strip().upper(),
                "quantity": float(row["quantity"]),
                "avg_cost": float(row["avg_cost"]),
                "purchase_date": row.get("purchase_date", "").strip(),
            })
    return holdings


def get_holding(ticker: str, holdings: list[dict] | None = None) -> dict | None:
    """Look up one position by ticker (case-insensitive). None if not held."""
    if holdings is None:
        holdings = load_holdings()
    ticker = ticker.strip().upper()
    return next((h for h in holdings if h["ticker"] == ticker), None)


def find_held_tickers(query: str, holdings: list[dict] | None = None) -> list[str]:
    """
    Tickers from holdings that appear in the query as whole uppercase words
    (e.g. "my AAPL position"). Case-sensitive on purpose so short tickers
    don't match ordinary words.
    """
    if holdings is None:
        holdings = load_holdings()
    return [h["ticker"] for h in holdings
            if re.search(rf"(?<![A-Za-z]){re.escape(h['ticker'])}(?![A-Za-z])", query)]


def format_position(h: dict) -> str:
    qty = f"{h['quantity']:g}"
    cost_basis = h["quantity"] * h["avg_cost"]
    bought = f", purchased {h['purchase_date']}" if h["purchase_date"] else ""
    return (f"{h['ticker']}: {qty} shares @ avg cost ${h['avg_cost']:,.2f} "
            f"(total cost basis ${cost_basis:,.2f}){bought}")


if __name__ == "__main__":
    hs = load_holdings()
    if not hs:
        print(f"No holdings found at {HOLDINGS_PATH}")
    for h in hs:
        print(format_position(h))
