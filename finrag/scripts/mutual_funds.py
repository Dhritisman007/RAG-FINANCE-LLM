"""
Mutual funds: daily NAVs from AMFI, the industry body that publishes every scheme's NAV.

A fund holding is identified by its ISIN, as the ticker "MF:<ISIN>" (e.g.
MF:INF179K01XD8), so it sits beside stock tickers in holdings. It is valued in ₹ at
the latest NAV. AMFI publishes one NAV per scheme per business day (in the
evening), so there is no intraday price: "day change" is the latest NAV against
the previous one this cache saw.

Source (checked 29 Oct 2026; www.amfiindia.com redirects here):
    https://portal.amfiindia.com/spages/NAVAll.txt
    Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date
The whole file is downloaded (~1.5 MB, every scheme) — like the stock list, it
reveals nothing about which funds anyone holds. Fund ISINs are never sent to
Yahoo Finance.

    python scripts/mutual_funds.py --update        # refresh data/portfolio/amfi_nav.csv
    python scripts/mutual_funds.py INF179K01XD8     # look one up
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent.parent
NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
NAV_PATH = ROOT / "data" / "portfolio" / "amfi_nav.csv"   # public data, refreshed daily (git-ignored)
HEADERS = {"User-Agent": "FinRAG/0.1 (personal research; local RAG project)"}
PREFIX = "MF:"
SOURCE = "AMFI NAV"
FIELDS = ["isin", "scheme_code", "name", "plan", "option", "nav", "date", "prev_nav", "prev_date"]
REFRESH_AFTER_SEC = 6 * 3600     # NAVs change once a day
MIN_SCHEMES = 5000               # far fewer means AMFI changed the file; keep the old one


def is_mf(ticker: str) -> bool:
    return ticker.strip().upper().startswith(PREFIX)


def ticker_for(isin: str) -> str:
    return PREFIX + isin.strip().upper()


def isin_of(ticker: str) -> str:
    return ticker.strip().upper().removeprefix(PREFIX)


def _iso(text: str) -> str | None:
    try:
        return datetime.strptime(text.strip(), "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None


def parse_navall(text: str) -> dict[str, dict]:
    """NAVAll.txt -> {ISIN: row}. Both ISIN columns (growth/payout and reinvestment) map to the scheme's row."""
    out: dict[str, dict] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) != 8 or not parts[0].isdigit():   # headers, AMC names and blank lines
            continue
        code, isin_a, isin_b, name, plan, option, nav, day = parts
        try:
            value = float(nav)
        except ValueError:                               # "N.A." for suspended schemes
            continue
        date = _iso(day)
        if value <= 0 or not date:
            continue
        row = {"scheme_code": code, "name": name, "plan": plan, "option": option, "nav": value, "date": date}
        for isin in (isin_a, isin_b):
            if isin.startswith("INF") and len(isin) == 12:
                out[isin] = {"isin": isin, **row}
    return out


def load_navs(path: Path = NAV_PATH) -> dict[str, dict]:
    return dict(_load(path, path.stat().st_mtime if path.exists() else 0.0))


@lru_cache(maxsize=4)
def _load(path: Path, mtime: float) -> tuple[tuple[str, dict], ...]:  # re-read only when the file changes
    if not mtime:
        return ()
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["nav"] = float(r["nav"])
            r["prev_nav"] = float(r["prev_nav"]) if r.get("prev_nav") else None
            rows.append((r["isin"], r))
    return tuple(rows)


def update(path: Path = NAV_PATH, get=None) -> int:
    """Download today's NAVs; keep each scheme's previous NAV (for day change). Returns the scheme count."""
    if get is None:
        import requests
        get = requests.get
    resp = get(NAV_URL, headers=HEADERS, timeout=90)
    resp.raise_for_status()
    fresh = parse_navall(resp.text)
    if len(fresh) < MIN_SCHEMES:
        raise RuntimeError(f"only {len(fresh)} schemes in AMFI's file — its format may have changed; keeping the old NAVs")
    old = load_navs(path)
    for isin, row in fresh.items():
        before = old.get(isin)
        if before and before["date"] < row["date"]:      # a new NAV: yesterday's becomes the previous one
            row["prev_nav"], row["prev_date"] = before["nav"], before["date"]
        elif before:                                       # same day again: keep what we had as previous
            row["prev_nav"], row["prev_date"] = before.get("prev_nav"), before.get("prev_date")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(fresh.values())
    tmp.replace(path)
    return len(fresh)


def refresh_if_stale(path: Path = NAV_PATH, get=None) -> str | None:
    """Update the cache when it is older than REFRESH_AFTER_SEC. Returns an error message instead of raising."""
    if path.exists() and time.time() - path.stat().st_mtime < REFRESH_AFTER_SEC:
        return None
    try:
        update(path, get)
        return None
    except Exception as e:  # offline, AMFI down: the last NAVs are still used, with their date shown
        return f"couldn't refresh mutual fund NAVs ({type(e).__name__}); using those from the last download"


def quotes(tickers: list[str], path: Path = NAV_PATH) -> dict[str, dict]:
    """The same shape as prices.yahoo_quotes, for MF: tickers: {ticker: {price, prev_close, as_of, source}}."""
    navs = load_navs(path)
    out = {}
    for t in tickers:
        row = navs.get(isin_of(t)) if is_mf(t) else None
        if row:
            out[t.strip().upper()] = {"price": row["nav"], "prev_close": row.get("prev_nav"), "as_of": row["date"],
                                      "source": SOURCE}
    return out


def display_name(ticker_or_isin: str, path: Path = NAV_PATH) -> str:
    """'MF:INF179K01XD8' -> 'HDFC Corporate Bond Fund · Direct · Growth' (the ISIN if unknown)."""
    isin = isin_of(ticker_or_isin)
    row = load_navs(path).get(isin)
    if not row:
        return isin
    plan = row["plan"].replace(" Plan", "").strip()
    option = row["option"].replace(" Option", "").strip()
    return " · ".join(p for p in (row["name"], plan, option) if p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMFI mutual fund NAVs for PortIQ.")
    parser.add_argument("--update", action="store_true", help="Download today's NAVs into data/portfolio/")
    parser.add_argument("isin", nargs="?", help="Look up a scheme by ISIN")
    args = parser.parse_args()
    if args.update:
        print(f"Saved NAVs of {update():,} scheme ISINs to {NAV_PATH.relative_to(ROOT)}")
    if args.isin:
        row = load_navs().get(args.isin.strip().upper())
        print(f"{display_name(args.isin)}: ₹{row['nav']:,.4f} on {row['date']}" if row else "not found")
