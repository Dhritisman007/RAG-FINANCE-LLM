"""
Portfolio store — the user's own holdings, looked up as structured data.

Holdings are NOT embedded into Chroma: ticker / quantity / cost basis are exact
facts, so they're read directly and injected into the prompt alongside retrieved
filing chunks (see rag_loop.py). Two kinds:
  - manual: data/portfolio/holdings.csv, edited in the UI
  - synced: the latest snapshot of each source (CSV export, Upstox, Dhan, Groww)
    in data/portfolio/portfolio.db, written by scripts/portfolio_sync.py. Each
    carries its source and as-of time, and format_position() states both, so an
    answer about a synced holding always says where and when it came from.

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


SOURCE_LABELS = {"manual": "your holdings file", "csv": "CSV import", "upstox": "Upstox", "dhan": "Dhan",
                 "groww": "Groww"}


def load_holdings(path: Path | None = None, snapshots_db: Path | None = None) -> list[dict]:
    """
    All holdings: the manual file plus the latest synced snapshot of every source
    (each dict has "source", and "as_of" for synced ones). With an explicit `path`,
    only that CSV file is read, as before.
    """
    if path is not None:
        return load_manual_holdings(path)
    return load_manual_holdings() + load_snapshot_holdings(snapshots_db)


def load_snapshot_holdings(db_path: Path | None = None) -> list[dict]:
    import portfolio_db
    out = []
    for snap in portfolio_db.latest_snapshots(db_path or portfolio_db.DB_PATH):
        for h in snap["holdings"]:
            out.append({"ticker": h["ticker"], "quantity": h["quantity"], "avg_cost": h["avg_cost"],
                        "purchase_date": "", "source": snap["source"], "as_of": snap["fetched_at"],
                        "exchange": h["exchange"], "isin": h["isin"], "currency": h["currency"],
                        "last_price": h["last_price"], "price_as_of": h["price_as_of"]})
    return out


def load_manual_holdings(path: Path = HOLDINGS_PATH) -> list[dict]:
    """Read holdings.csv -> [{ticker, quantity, avg_cost, purchase_date, source: "manual"}, ...]."""
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
                "source": "manual",
            })
    return holdings


# US (AAPL, BRK-B) and Indian NSE/BSE symbols with Yahoo suffixes (RELIANCE.NS, M&M.NS, BAJAJ-AUTO.BO)
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9&.\-]{0,19}$")
INDIA_SUFFIXES = (".NS", ".BO")


def validate_holdings(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Clean and check edited rows -> (holdings, errors). Blank rows are dropped.
    Each error names the row (1-based) so the UI can point at it.
    """
    import math
    from datetime import date

    def blank(v) -> bool:  # the table editor returns empty number cells as NaN, not None
        return v is None or v == "" or (isinstance(v, float) and math.isnan(v))

    def number(v):
        if blank(v):
            raise ValueError
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            raise ValueError
        return x

    holdings, errors, seen = [], [], set()
    for i, row in enumerate(rows, start=1):
        ticker = str(row.get("ticker") or "").strip().upper()
        qty, cost, when = row.get("quantity"), row.get("avg_cost"), row.get("purchase_date")
        if not ticker and blank(qty) and blank(cost):
            continue  # an empty row the editor added
        if not TICKER_RE.match(ticker):
            errors.append(f"Row {i}: '{ticker or '(blank)'}' isn't a valid ticker (e.g. RELIANCE.NS, TCS.BO, AAPL).")
        elif ticker in seen:
            errors.append(f"Row {i}: {ticker} appears twice — combine it into one row.")
        seen.add(ticker)
        try:
            qty = number(qty)
            if qty <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"Row {i}: shares must be a number greater than 0.")
        try:
            cost = number(cost)
            if cost < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"Row {i}: average cost must be a number of 0 or more.")
        if isinstance(when, date):
            when = when.isoformat()
        when = str(when or "").strip()[:10]
        if when:
            try:
                if date.fromisoformat(when) > date.today():
                    errors.append(f"Row {i}: purchase date {when} is in the future.")
            except ValueError:
                errors.append(f"Row {i}: purchase date must be YYYY-MM-DD.")
        if not errors or not errors[-1].startswith(f"Row {i}:"):
            holdings.append({"ticker": ticker, "quantity": qty, "avg_cost": cost, "purchase_date": when})
    return holdings, errors


def save_holdings(rows: list[dict], path: Path = HOLDINGS_PATH) -> list[str]:
    """
    Validate and write holdings.csv. Returns the errors; nothing is written
    unless every row is valid. Writes to a temp file first, so a crash can't
    leave a half-written holdings file.
    """
    holdings, errors = validate_holdings(rows)
    if errors:
        return errors
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "quantity", "avg_cost", "purchase_date"])
        for h in holdings:
            w.writerow([h["ticker"], f"{h['quantity']:g}", f"{h['avg_cost']:.2f}", h["purchase_date"]])
    tmp.replace(path)
    return []


def add_holding(row: dict, path: Path = HOLDINGS_PATH) -> list[str]:
    """
    Add one position and save. Buying more of a ticker already held merges the
    lots: shares add up, average cost becomes the weighted average, and the
    earlier purchase date is kept. Returns the errors (nothing saved if any).
    """
    new, errors = validate_holdings([row])
    if errors or not new:
        return [e.removeprefix("Row 1: ") for e in errors] or ["Enter a ticker, shares and average cost."]
    new = new[0]
    rows = load_holdings(path)
    held = next((h for h in rows if h["ticker"] == new["ticker"]), None)
    if held:
        total = held["quantity"] + new["quantity"]
        held["avg_cost"] = (held["quantity"] * held["avg_cost"] + new["quantity"] * new["avg_cost"]) / total
        held["quantity"] = total
        held["purchase_date"] = min(filter(None, [held["purchase_date"], new["purchase_date"]]), default="")
    else:
        rows.append(new)
    return save_holdings(rows, path)


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
    don't match ordinary words. Indian holdings also match without their
    exchange suffix ("my RELIANCE shares" -> RELIANCE.NS), or capitalised
    when 5+ letters long ("Wipro" -> WIPRO.NS).
    """
    if holdings is None:
        holdings = load_holdings()
    found = []
    for h in holdings:
        t = h["ticker"]
        forms = [t]
        if t.endswith(INDIA_SUFFIXES):
            base = t[:-3]
            forms += [base] + ([base.capitalize()] if len(base) >= 5 and base.isalpha() else [])
        if t not in found and any(re.search(rf"(?<![A-Za-z&]){re.escape(f)}(?![A-Za-z&])", query) for f in forms):
            found.append(t)  # once, even if several sources hold it
    return found


def _price(x: float, ticker: str) -> str:
    if not ticker.upper().endswith(INDIA_SUFFIXES):
        return f"${x:,.2f}"
    whole, frac = f"{x:.2f}".split(".")  # Indian grouping: ₹12,34,567.80
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "₹" + ",".join(([head] if head else []) + groups + [tail]) + "." + frac


def format_position(h: dict) -> str:
    qty = f"{h['quantity']:g}"
    cost_basis = h["quantity"] * h["avg_cost"]
    bought = f", purchased {h['purchase_date']}" if h["purchase_date"] else ""
    source = h.get("source", "manual")
    origin = (f" — from {SOURCE_LABELS.get(source, source)}, as of {h['as_of']}"
              if source != "manual" and h.get("as_of") else "")
    return (f"{h['ticker']}: {qty} shares @ avg cost {_price(h['avg_cost'], h['ticker'])} "
            f"(total cost basis {_price(cost_basis, h['ticker'])}){bought}{origin}")


if __name__ == "__main__":
    hs = load_holdings()
    if not hs:
        print(f"No holdings found at {HOLDINGS_PATH}")
    for h in hs:
        print(format_position(h))
