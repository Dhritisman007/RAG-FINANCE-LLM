"""
Price data for portfolio questions ("How is my portfolio doing?").

Three sources, merged by date (newest wins; on the same date manual > Yahoo > Nasdaq):

1. Yahoo Finance via yfinance (default, Day 17) — the daily close for each of
   your holdings, including ETFs and mutual funds. PRIVACY: Yahoo sees the
   tickers you request (your holdings' tickers — never quantities or costs).
2. Nasdaq stock screener (Day 14, optional) — the whole US stock market in one
   request with no tickers in it, so it reveals nothing about your holdings;
   stocks only, and an unofficial website endpoint.
3. data/portfolio/manual_prices.csv (`ticker,price,date`) — never sent anywhere.

Indian listings use Yahoo's suffixes (RELIANCE.NS on NSE, RELIANCE.BO on BSE)
and are priced in ₹. Portfolio totals are in ₹; US holdings are converted at
the USD→INR rate (USDINR=X — a currency pair, so it reveals nothing about you).

Prices are only fetched when you ask (or while the UI's Live prices is on). Valuation is computed locally, by code —
the LLM never computes or restates position numbers. All price files are
git-ignored.

Usage:
    python scripts/prices.py --update          # Yahoo Finance closes for your holdings
    python scripts/prices.py --update-market   # Nasdaq whole-market download (no tickers sent)
    python scripts/prices.py                   # value your holdings
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
YAHOO_PRICES_PATH = ROOT / "data" / "portfolio" / "yahoo_prices.csv"
YAHOO_SOURCE = "Yahoo Finance close (yfinance)"

# Fixed URL: nothing in it depends on your holdings.
SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
SCREENER_HEADERS = {"User-Agent": "Mozilla/5.0 (FinRAG local price update)", "Accept": "application/json"}
SOURCE_NAME = "Nasdaq stock screener (last sale)"
STALE_AFTER_DAYS = 3  # older prices are flagged; weekends and holidays fit inside this
# Indian listings use Yahoo's exchange suffixes and are priced in rupees; totals are in ₹.
INDIA_SUFFIXES = (".NS", ".BO")  # NSE, BSE
EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO", "US": ""}
FX_TICKER = "USDINR=X"  # converts US holdings to ₹; a currency pair, so it reveals nothing about you


def normalize_ticker(ticker: str) -> str:
    """'BRK/B', 'brk.b' and 'BRK-B' are the same listing; 'reliance.ns' -> 'RELIANCE.NS'."""
    t = ticker.strip().upper()
    if t.endswith(INDIA_SUFFIXES) or t.endswith("=X"):
        return t
    return re.sub(r"[/.]", "-", t)


def with_exchange(symbol: str, exchange: str) -> str:
    """'reliance' on NSE -> 'RELIANCE.NS'. A symbol that already has a suffix is kept as typed."""
    s = symbol.strip().upper()
    return s if s.endswith(INDIA_SUFFIXES) else s + EXCHANGE_SUFFIX.get(exchange, "")


def currency(ticker: str) -> str:
    return "INR" if normalize_ticker(ticker).endswith(INDIA_SUFFIXES) else "USD"


def display_symbol(ticker: str) -> tuple[str, str]:
    """'RELIANCE.NS' -> ('RELIANCE', 'NSE'); 'AAPL' -> ('AAPL', 'US')."""
    t = ticker.upper()
    for exchange, suffix in EXCHANGE_SUFFIX.items():
        if suffix and t.endswith(suffix):
            return t[: -len(suffix)], exchange
    return t, "US"


def inr(x: float, decimals: int = 2) -> str:
    """Indian digit grouping: 1234567.8 -> '₹12,34,567.80' (decimals=0 -> '₹12,34,568')."""
    whole, _, frac = f"{abs(x):.{decimals}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ("-" if x < 0 else "") + "₹" + ",".join(groups + [tail]) + ("." + frac if frac else "")


def money(x: float, cur: str = "INR") -> str:
    return inr(x) if cur == "INR" else (f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}")


def signed_money(x: float, cur: str = "INR", decimals: int = 2) -> str:
    sign = "+" if x >= 0 else "-"
    return sign + (inr(abs(x), decimals) if cur == "INR" else money(abs(x), cur))


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


def yahoo_quotes(tickers: list[str], download=None) -> dict[str, dict]:
    """
    Latest price and previous close per ticker from Yahoo Finance (stocks, ETFs,
    funds): {ticker: {"price", "prev_close", "as_of"}}. While the market is open
    the latest daily bar is today's, so "price" is live (Yahoo's delay applies).
    `download` is injectable for tests; defaults to yfinance.download. Yahoo sees
    these tickers — never quantities or costs.
    """
    tickers = sorted({normalize_ticker(t) for t in tickers})
    if not tickers:
        return {}
    if download is None:
        import yfinance as yf
        download = yf.download
    data = download(tickers, period="7d", interval="1d", progress=False, auto_adjust=False, threads=False)
    close = data["Close"]
    quotes = {}
    for t in tickers:
        series = close[t].dropna() if t in close else []
        if len(series):
            quotes[t] = {"price": float(series.iloc[-1]), "as_of": series.index[-1].date().isoformat(),
                         "prev_close": float(series.iloc[-2]) if len(series) > 1 else None}
    return quotes


def save_yahoo_prices(quotes: dict[str, dict], path: Path = YAHOO_PRICES_PATH) -> None:
    """Merge quotes into the local Yahoo price file (earlier tickers' prices are kept)."""
    if not quotes:
        return
    existing = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = {r["ticker"]: r for r in csv.DictReader(f)}
    for t, q in quotes.items():
        existing[t] = {"ticker": t, "price": f"{q['price']:.4f}", "as_of": q["as_of"], "source": YAHOO_SOURCE}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "price", "as_of", "source"])
        w.writeheader()
        w.writerows(existing[t] for t in sorted(existing))


def with_fx(tickers: list[str]) -> list[str]:
    """Add the USD→INR pair when any holding is priced in dollars."""
    return list(tickers) + ([FX_TICKER] if any(currency(t) == "USD" for t in tickers) else [])


def fetch_yahoo_prices(tickers: list[str], path: Path = YAHOO_PRICES_PATH, download=None) -> tuple[int, list[str]]:
    """Fetch and save Yahoo prices (plus USD→INR if needed). Returns (holdings priced, holdings without a price)."""
    quotes = yahoo_quotes(with_fx(tickers), download)
    save_yahoo_prices(quotes, path)
    wanted = sorted({normalize_ticker(t) for t in tickers})
    return sum(t in quotes for t in wanted), [t for t in wanted if t not in quotes]


def yahoo_history(ticker: str, period: str = "1y", download=None) -> list[tuple[str, float]]:
    """Daily closes [(YYYY-MM-DD, close), ...] for one ticker, oldest first. Yahoo sees the ticker."""
    if download is None:
        import yfinance as yf
        download = yf.download
    t = normalize_ticker(ticker)
    data = download([t], period=period, interval="1d", progress=False, auto_adjust=False, threads=False)
    close = data["Close"][t].dropna()
    return [(d.date().isoformat(), float(v)) for d, v in close.items()]


def yahoo_histories(tickers: list[str], period: str = "1y", download=None) -> dict[str, list[tuple[str, float]]]:
    """Daily closes for several tickers in ONE request: {ticker: [(YYYY-MM-DD, close), ...]}.
    Index symbols like ^NSEI pass through unchanged. Yahoo sees these tickers."""
    wanted = sorted({t if t.startswith("^") else normalize_ticker(t) for t in tickers})
    if not wanted:
        return {}
    if download is None:
        import yfinance as yf
        download = yf.download
    data = download(wanted, period=period, interval="1d", progress=False, auto_adjust=False, threads=False)
    close = data["Close"]
    out = {}
    for t in wanted:
        if t in close:
            series = close[t].dropna()
            if len(series):
                out[t] = [(d.date().isoformat(), float(v)) for d, v in series.items()]
    return out


def price_stats(history: list[tuple[str, float]]) -> dict:
    """
    Descriptive statistics of a price history — what happened, not a forecast:
    52-week high/low, 1/3/12-month returns, annualised volatility, and the
    largest peak-to-trough fall.
    """
    import math
    closes = [c for _, c in history]
    if len(closes) < 2:
        return {}
    last = closes[-1]

    def ret(days: int):  # ~21 trading days a month
        return (last / closes[-days - 1] - 1) * 100 if len(closes) > days else None

    daily = [b / a - 1 for a, b in zip(closes, closes[1:])]
    mean = sum(daily) / len(daily)
    vol = math.sqrt(sum((x - mean) ** 2 for x in daily) / max(1, len(daily) - 1)) * math.sqrt(252) * 100
    peak, drawdown = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        drawdown = min(drawdown, (c / peak - 1) * 100)
    year = closes[-252:]
    return {"last": last, "high_52w": max(year), "low_52w": min(year), "ret_1m": ret(21), "ret_3m": ret(63),
            "ret_1y": ret(min(252, len(closes) - 1)), "volatility": vol, "max_drawdown": drawdown}


def load_prices(market_path: Path = MARKET_PRICES_PATH, manual_path: Path = MANUAL_PRICES_PATH,
                yahoo_path: Path = YAHOO_PRICES_PATH, snapshot_db: Path | None = None,
                use_snapshots: bool = True) -> dict[str, dict]:
    """
    {ticker: {"price", "as_of", "source"}} — newest date wins; same date: manual > Yahoo > broker > Nasdaq.
    Broker prices come from the latest synced snapshots (portfolio_sync.py); missing ones are NULL there,
    never 0, so they're simply absent here.
    """
    prices: dict[str, dict] = {}
    if use_snapshots:
        import portfolio_db
        from portfolio_store import source_label
        for snap in portfolio_db.latest_snapshots(snapshot_db or portfolio_db.DB_PATH):
            for h in snap["holdings"]:
                if h["last_price"] and h["price_as_of"]:
                    current = prices.get(h["ticker"])
                    if current is None or h["price_as_of"][:10] >= current["as_of"][:10]:
                        prices[h["ticker"]] = {"price": float(h["last_price"]), "as_of": h["price_as_of"],
                                               "source": f"{source_label(snap['source'])} last price"}
    for source_path in (market_path, yahoo_path):  # later sources win ties (over broker prices too)
        if source_path.exists():
            with open(source_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    current = prices.get(row["ticker"])
                    if current is None or row["as_of"][:10] >= current["as_of"][:10]:
                        prices[row["ticker"]] = {"price": float(row["price"]), "as_of": row["as_of"],
                                                 "source": row["source"]}
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
    """
    Market value and gain per position, computed here — never by the LLM.
    Each position is valued in its own currency, plus in ₹ for the totals (US
    holdings at the USD→INR rate; without one they're left out of the ₹ totals).
    """
    today = today or date.today()
    fx = prices.get(FX_TICKER, {}).get("price")
    valued = []
    for h in holdings:
        cost = h["quantity"] * h["avg_cost"]
        cur = currency(h["ticker"])
        rate = 1.0 if cur == "INR" else fx
        p = prices.get(normalize_ticker(h["ticker"]))
        row = {"ticker": h["ticker"], "currency": cur, "quantity": h["quantity"], "avg_cost": h["avg_cost"],
               "cost_basis": cost, "price": None, "as_of": None, "source": None, "market_value": None,
               "gain": None, "gain_pct": None, "stale": False, "missing": p is None,
               "value_inr": None, "cost_inr": None,
               "holding_source": h.get("source", "manual"), "holding_as_of": h.get("as_of")}
        if p is not None:
            value = h["quantity"] * p["price"]
            age = (today - date.fromisoformat(p["as_of"][:10])).days
            row.update(price=p["price"], as_of=p["as_of"], source=p["source"], market_value=value,
                       gain=value - cost, gain_pct=(value - cost) / cost * 100 if cost else None,
                       stale=age > STALE_AFTER_DAYS)
            if rate:
                row.update(value_inr=value * rate, cost_inr=cost * rate)
        valued.append(row)
    return valued


def summarize(valued: list[dict]) -> dict:
    """Totals in ₹ over the positions that have a price (and, for US holdings, an exchange rate)."""
    priced = [v for v in valued if v["value_inr"] is not None]
    cost = sum(v["cost_inr"] for v in priced)
    value = sum(v["value_inr"] for v in priced)
    return {"cost_basis": cost, "market_value": value, "gain": value - cost,
            "gain_pct": (value - cost) / cost * 100 if cost else None,
            "priced": len(priced), "missing": [v["ticker"] for v in valued if v["missing"]],
            "no_fx": [v["ticker"] for v in valued if not v["missing"] and v["value_inr"] is None],
            "stale": [v["ticker"] for v in priced if v["stale"]],
            "as_of": min((v["as_of"] for v in priced), default=None)}


def format_valuation(valued: list[dict]) -> list[str]:
    """Plain-text valuation lines, shown verbatim by the CLI and the UI."""
    lines = []
    for v in valued:
        if v["missing"]:
            lines.append(f"{v['ticker']}: no price — run Update prices, or add it to {MANUAL_PRICES_PATH.name}")
            continue
        stale = " ⚠ stale" if v["stale"] else ""
        cur = v["currency"]
        lines.append(f"{v['ticker']}: {v['quantity']:g} × {money(v['price'], cur)} = {money(v['market_value'], cur)} "
                     f"(cost {money(v['cost_basis'], cur)}, {signed_money(v['gain'], cur)}, {v['gain_pct']:+.1f}%) "
                     f"— price as of {v['as_of']}{stale}{_holding_origin(v)}")
    return lines


def _holding_origin(v: dict) -> str:
    """' · holding from Upstox, as of …' for synced holdings; nothing for the manual file."""
    if v.get("holding_source", "manual") == "manual" or not v.get("holding_as_of"):
        return ""
    from portfolio_store import source_label
    return f" · holding from {source_label(v['holding_source'])}, as of {v['holding_as_of']}"


def summary_sentence(valued: list[dict]) -> str:
    """The deterministic answer to "how is my portfolio doing?" (no LLM involved)."""
    s = summarize(valued)
    if not s["priced"]:
        return ("There are no prices for your holdings yet — run `python scripts/prices.py --update` "
                f"(or add them to {MANUAL_PRICES_PATH.name}).")
    text = (f"Your priced holdings are worth {inr(s['market_value'])} against a cost basis of "
            f"{inr(s['cost_basis'])}: {signed_money(s['gain'])} ({s['gain_pct']:+.1f}%), "
            f"using prices as of {s['as_of']}.")
    if any(v["currency"] == "USD" and v["value_inr"] is not None for v in valued):
        text += " US holdings are converted to rupees at the latest USD→INR rate."
    if s["missing"]:
        text += f" Not included (no price): {', '.join(s['missing'])}."
    if s["no_fx"]:
        text += f" Not included (no USD→INR rate yet — run --update): {', '.join(s['no_fx'])}."
    if s["stale"]:
        text += f" Prices older than {STALE_AFTER_DAYS} days: {', '.join(s['stale'])} — run --update."
    from portfolio_store import source_label
    synced = sorted({(v["holding_source"], v["holding_as_of"]) for v in valued
                     if v.get("holding_source", "manual") != "manual" and v.get("holding_as_of")})
    origin = "your holdings file" + "".join(f"; {source_label(s)} snapshot as of {a}" for s, a in synced)
    return text + f" This is computed from {origin} and the last downloaded prices, not investment advice."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local price data for FinRAG portfolio questions.")
    parser.add_argument("--update", action="store_true",
                        help="Yahoo Finance closes for your holdings (Yahoo sees the tickers).")
    parser.add_argument("--update-market", action="store_true",
                        help="Nasdaq whole-market download (one request containing no tickers).")
    args = parser.parse_args()
    if args.update:
        n, missing = fetch_yahoo_prices([h["ticker"] for h in load_holdings()])
        print(f"Saved {n} Yahoo Finance prices" + (f"; no price for {', '.join(missing)}" if missing else ""))
    if args.update_market:
        n = fetch_market_prices()
        print(f"Saved {n:,} market prices to {MARKET_PRICES_PATH.relative_to(ROOT)}")
    valued = value_positions(load_holdings(), load_prices())
    print("\n".join(format_valuation(valued)))
    print("\n" + summary_sentence(valued))
