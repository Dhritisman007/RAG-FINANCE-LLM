"""
Day 14: Local price data for portfolio questions ("How is my portfolio doing?").

PRIVACY (deliberate design decision): prices are fetched for the WHOLE US stock
market in a single request — Nasdaq's public stock screener, ~7,000 stocks —
so the request contains no ticker and reveals nothing about your holdings. It
only happens when you ask (`--update`, or the UI's "Update prices" button).
Holdings never leave the machine; valuation is computed locally, by code — the
LLM never computes or restates position numbers.

Anything the screener doesn't cover (ETFs, mutual funds, bond funds, non-US
listings) goes in data/portfolio/manual_prices.csv — `ticker,price,date` —
which is never sent anywhere. A manual price wins over the market price when
its date is the same or newer. Both price files are git-ignored.

The screener is Nasdaq's public website endpoint, not an official API: if it
changes, --update fails loudly and manual prices keep working.

Usage:
    python scripts/prices.py --update      # download market prices (one request, no tickers)
    python scripts/prices.py               # value your holdings
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from portfolio_store import load_holdings  # noqa: E402

ROOT = Path(__file__).parent.parent
MARKET_PRICES_PATH = ROOT / "data" / "portfolio" / "market_prices.csv"
MANUAL_PRICES_PATH = ROOT / "data" / "portfolio" / "manual_prices.csv"

# Fixed URL: nothing in it depends on your holdings.
SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
SCREENER_HEADERS = {"User-Agent": "Mozilla/5.0 (FinRAG local price update)", "Accept": "application/json"}
SOURCE_NAME = "Nasdaq stock screener (last sale)"
STALE_AFTER_DAYS = 3  # older prices are flagged; weekends and holidays fit inside this


def normalize_ticker(ticker: str) -> str:
    """'BRK/B', 'brk.b' and 'BRK-B' are the same listing."""
    return re.sub(r"[/.]", "-", ticker.strip().upper())


def parse_price(text: str) -> float | None:
    try:
        return float(text.replace("$", "").replace(",", "").strip())
    except (AttributeError, ValueError):
        return None


def parse_screener(payload: dict) -> dict[str, float]:
    """Screener JSON -> {ticker: last sale}. Rows without a usable price are skipped."""
    rows = ((payload or {}).get("data") or {}).get("rows") or []
    out = {}
    for row in rows:
        price = parse_price(row.get("lastsale", ""))
        if row.get("symbol") and price and price > 0:
            out[normalize_ticker(row["symbol"])] = price
    return out


def fetch_market_prices(path: Path = MARKET_PRICES_PATH, get=None) -> int:
    """
    Download the whole-market screener (one request, no tickers) and save it
    locally. `get` is injectable for tests; defaults to requests.get.
    """
    if get is None:
        import requests
        get = requests.get
    resp = get(SCREENER_URL, headers=SCREENER_HEADERS, timeout=60)
    resp.raise_for_status()
    prices = parse_screener(resp.json())
    if len(prices) < 1000:  # the full market is ~7,000; far fewer means the endpoint changed
        raise RuntimeError(f"screener returned only {len(prices)} prices — endpoint may have changed; "
                           f"use {MANUAL_PRICES_PATH.name} meanwhile")
    fetched_at = datetime.now().isoformat(timespec="minutes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "price", "as_of", "source"])
        for ticker, price in sorted(prices.items()):
            w.writerow([ticker, f"{price:.4f}", fetched_at, SOURCE_NAME])
    return len(prices)


def load_prices(market_path: Path = MARKET_PRICES_PATH, manual_path: Path = MANUAL_PRICES_PATH) -> dict[str, dict]:
    """{ticker: {"price", "as_of", "source"}} — a manual price wins if its date is the same or newer."""
    prices: dict[str, dict] = {}
    if market_path.exists():
        with open(market_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prices[row["ticker"]] = {"price": float(row["price"]), "as_of": row["as_of"], "source": row["source"]}
    if manual_path.exists():
        with open(manual_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ticker, price = normalize_ticker(row.get("ticker", "")), parse_price(row.get("price", ""))
                if not ticker or price is None:
                    continue
                as_of = row.get("date", "").strip()
                current = prices.get(ticker)
                if current is None or as_of[:10] >= current["as_of"][:10]:
                    prices[ticker] = {"price": price, "as_of": as_of, "source": "manual price file"}
    return prices


def value_positions(holdings: list[dict], prices: dict[str, dict], today: date | None = None) -> list[dict]:
    """Market value and gain per position, computed here — never by the LLM."""
    today = today or date.today()
    valued = []
    for h in holdings:
        cost = h["quantity"] * h["avg_cost"]
        p = prices.get(normalize_ticker(h["ticker"]))
        row = {"ticker": h["ticker"], "quantity": h["quantity"], "avg_cost": h["avg_cost"], "cost_basis": cost,
               "price": None, "as_of": None, "source": None, "market_value": None, "gain": None,
               "gain_pct": None, "stale": False, "missing": p is None}
        if p is not None:
            value = h["quantity"] * p["price"]
            age = (today - date.fromisoformat(p["as_of"][:10])).days
            row.update(price=p["price"], as_of=p["as_of"], source=p["source"], market_value=value,
                       gain=value - cost, gain_pct=(value - cost) / cost * 100 if cost else None,
                       stale=age > STALE_AFTER_DAYS)
        valued.append(row)
    return valued


def summarize(valued: list[dict]) -> dict:
    priced = [v for v in valued if not v["missing"]]
    cost = sum(v["cost_basis"] for v in priced)
    value = sum(v["market_value"] for v in priced)
    return {"cost_basis": cost, "market_value": value, "gain": value - cost,
            "gain_pct": (value - cost) / cost * 100 if cost else None,
            "priced": len(priced), "missing": [v["ticker"] for v in valued if v["missing"]],
            "stale": [v["ticker"] for v in priced if v["stale"]],
            "as_of": min((v["as_of"] for v in priced), default=None)}


def _money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _signed(x: float) -> str:
    return ("+" if x >= 0 else "-") + f"${abs(x):,.2f}"


def format_valuation(valued: list[dict]) -> list[str]:
    """Plain-text valuation lines, shown verbatim by the CLI and the UI."""
    lines = []
    for v in valued:
        if v["missing"]:
            lines.append(f"{v['ticker']}: no price — add it to {MANUAL_PRICES_PATH.name} (ETFs/funds aren't in the screener)")
            continue
        stale = " ⚠ stale" if v["stale"] else ""
        lines.append(f"{v['ticker']}: {v['quantity']:g} × {_money(v['price'])} = {_money(v['market_value'])} "
                     f"(cost {_money(v['cost_basis'])}, {_signed(v['gain'])}, {v['gain_pct']:+.1f}%) "
                     f"— price as of {v['as_of']}{stale}")
    return lines


def summary_sentence(valued: list[dict]) -> str:
    """The deterministic answer to "how is my portfolio doing?" (no LLM involved)."""
    s = summarize(valued)
    if not s["priced"]:
        return ("There are no prices for your holdings yet — run `python scripts/prices.py --update` "
                f"(or add them to {MANUAL_PRICES_PATH.name}).")
    text = (f"Your priced holdings are worth {_money(s['market_value'])} against a cost basis of "
            f"{_money(s['cost_basis'])}: {_signed(s['gain'])} ({s['gain_pct']:+.1f}%), "
            f"using prices as of {s['as_of']}.")
    if s["missing"]:
        text += f" Not included (no price): {', '.join(s['missing'])}."
    if s["stale"]:
        text += f" Prices older than {STALE_AFTER_DAYS} days: {', '.join(s['stale'])} — run --update."
    return text + " This is computed from your holdings file and the last downloaded prices, not investment advice."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local price data for FinRAG portfolio questions.")
    parser.add_argument("--update", action="store_true",
                        help="Download whole-market prices (one request containing no tickers).")
    args = parser.parse_args()
    if args.update:
        n = fetch_market_prices()
        print(f"Saved {n:,} market prices to {MARKET_PRICES_PATH.relative_to(ROOT)}")
    valued = value_positions(load_holdings(), load_prices())
    print("\n".join(format_valuation(valued)))
    print("\n" + summary_sentence(valued))
