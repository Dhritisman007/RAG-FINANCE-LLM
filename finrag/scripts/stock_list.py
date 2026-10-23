"""
Day 19: Stock list for the "＋ Add" search — every NSE equity and ETF, plus US companies.

The lists are downloaded whole (no holdings are sent anywhere) and stored in
data/reference/stocks.csv, which is committed so the app works offline. Each row
carries the Yahoo Finance ticker the price feed uses (RELIANCE.NS, NIFTYBEES.NS,
AAPL), so a stock picked from the list always gets prices.

Sources:
  - NSE equities  https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
  - NSE ETFs      https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv
  - US companies  https://www.sec.gov/files/company_tickers.json (largest companies first)

Usage:
    python scripts/stock_list.py --update     # refresh data/reference/stocks.csv
    python scripts/stock_list.py RELI         # search the list, as the app does
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
STOCKS_PATH = ROOT / "data" / "reference" / "stocks.csv"
NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_ETF_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
NSE_HEADERS = {"User-Agent": "FinRAG/0.1 (personal research; local RAG project)"}
FIELDS = ["ticker", "symbol", "name", "exchange"]
EXCHANGE_LABELS = {"NSE": "NSE", "NSE ETF": "NSE · ETF", "US": "US"}


def parse_nse_equity(text: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("SYMBOL"):
            rows.append({"ticker": f"{r['SYMBOL'].upper()}.NS", "symbol": r["SYMBOL"].upper(),
                         "name": r.get("NAME OF COMPANY", ""), "exchange": "NSE"})
    return rows


def parse_nse_etf(text: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if r.get("Symbol"):
            underlying = r.get("Underlying Asset") or r.get("SecurityName", "")
            rows.append({"ticker": f"{r['Symbol'].upper()}.NS", "symbol": r["Symbol"].upper(),
                         "name": f"{underlying} ETF".strip(), "exchange": "NSE ETF"})
    return rows


def parse_sec(payload: dict) -> list[dict]:
    """SEC's list is ordered by company size; keep that order, one row per ticker."""
    rows, seen = [], set()
    for _, entry in sorted(payload.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
        ticker = str(entry.get("ticker", "")).strip().upper().replace(".", "-")  # Yahoo: BRK-B
        if ticker and ticker not in seen:
            seen.add(ticker)
            rows.append({"ticker": ticker, "symbol": ticker, "name": str(entry.get("title", "")).strip(),
                         "exchange": "US"})
    return rows


def update(path: Path = STOCKS_PATH, get=None) -> int:
    """Download all three lists and write stocks.csv (Indian listings first). Returns the row count."""
    if get is None:
        import requests
        get = requests.get
    sys.path.insert(0, str(Path(__file__).parent))
    from scrape_tier2_edgar import HEADERS as SEC_HEADERS  # the SEC asks for a contact address
    rows = []
    for url, parse in ((NSE_EQUITY_URL, parse_nse_equity), (NSE_ETF_URL, parse_nse_etf)):
        resp = get(url, headers=NSE_HEADERS, timeout=60)
        resp.raise_for_status()
        rows += parse(resp.text)
    resp = get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=60)
    resp.raise_for_status()
    rows += parse_sec(resp.json())
    if len(rows) < 5000:  # NSE ~2,500 + SEC ~10,000; far fewer means a source changed
        raise RuntimeError(f"only {len(rows)} stocks downloaded — a source may have changed; keeping the old list")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    return len(rows)


def load_stocks(path: Path = STOCKS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label(row: dict) -> str:
    """What the dropdown shows and matches against: symbol, company name and exchange."""
    return f"{row['symbol']} — {row['name']} · {EXCHANGE_LABELS.get(row['exchange'], row['exchange'])}"


def ticker_for(choice: str | None, by_label: dict[str, str]) -> str:
    """
    The Yahoo ticker for a dropdown choice. Something typed that isn't in the list is
    taken as a symbol: kept as typed if it has an exchange suffix (TCS.BO), else NSE.
    """
    choice = (choice or "").strip()
    if not choice:
        return ""
    if choice in by_label:
        return by_label[choice]
    symbol = choice.split(" — ")[0].strip().upper()
    return symbol if symbol.endswith((".NS", ".BO")) else f"{symbol}.NS"


def search(query: str, rows: list[dict], limit: int = 10) -> list[str]:
    """Case-insensitive match on symbol or name — symbol prefixes first (used by the CLI and tests)."""
    q = query.strip().lower()
    prefix = [r for r in rows if r["symbol"].lower().startswith(q)]
    rest = [r for r in rows if r not in prefix and (q in r["symbol"].lower() or q in r["name"].lower())]
    return [label(r) for r in (prefix + rest)[:limit]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock list for PortIQ's add-stock search.")
    parser.add_argument("--update", action="store_true", help="Download NSE and SEC lists into data/reference/")
    parser.add_argument("query", nargs="?", help="Search the list")
    args = parser.parse_args()
    if args.update:
        print(f"Saved {update():,} stocks to {STOCKS_PATH.relative_to(ROOT)}")
    if args.query:
        print("\n".join(search(args.query, load_stocks())))
