"""
Day 19: Portfolio insights — computed by code from your holdings and Yahoo prices,
never by the language model.

All figures describe the past or the present (allocation, concentration, how
today's holdings would have done over the last year); nothing is forecast and
nothing here is a recommendation.
"""
from __future__ import annotations

import math
from datetime import date

BENCHMARKS = {"^NSEI": "Nifty 50", "^GSPC": "S&P 500"}
CONCENTRATION_FLAG = 25.0  # % of the portfolio in one stock that is called out


def allocation(rows: list[dict]) -> dict:
    """
    rows: priced holdings [{"symbol", "currency", "value_inr", "pnl_pct"}].
    Returns weights, India/US split, concentration and the best/worst performers.
    """
    priced = [r for r in rows if r.get("value_inr")]
    total = sum(r["value_inr"] for r in priced)
    if not priced or total <= 0:
        return {}
    by_symbol: dict[str, float] = {}  # one stock held in several places (manual + a broker) counts once
    for r in priced:
        by_symbol[r["symbol"]] = by_symbol.get(r["symbol"], 0.0) + r["value_inr"]
    weights = sorted(((s, v / total * 100) for s, v in by_symbol.items()), key=lambda x: -x[1])
    india = sum(r["value_inr"] for r in priced if r["currency"] == "INR") / total * 100
    by_return = sorted((r for r in priced if r.get("pnl_pct") is not None), key=lambda r: r["pnl_pct"])
    hhi = sum((w / 100) ** 2 for _, w in weights)
    return {
        "weights": weights,
        "largest": weights[0],
        "top3": sum(w for _, w in weights[:3]),
        "effective_holdings": 1 / hhi,  # how many equal-sized positions this concentration equals
        "india_pct": india, "us_pct": 100 - india,
        "best": (by_return[-1]["symbol"], by_return[-1]["pnl_pct"]) if by_return else None,
        "worst": (by_return[0]["symbol"], by_return[0]["pnl_pct"]) if by_return else None,
    }


def holding_periods(holdings: list[dict], today: date | None = None) -> dict[str, list[str]]:
    """Holdings grouped by how long ago they were bought (purchase date optional)."""
    today = today or date.today()
    groups = {"under 1 year": [], "1–3 years": [], "over 3 years": [], "date not given": []}
    for h in holdings:
        symbol = h["ticker"].split(".")[0]
        if not h.get("purchase_date"):
            groups["date not given"].append(symbol)
            continue
        years = (today - date.fromisoformat(h["purchase_date"])).days / 365.25
        groups["under 1 year" if years < 1 else "1–3 years" if years <= 3 else "over 3 years"].append(symbol)
    return groups


def portfolio_history(holdings: list[dict], histories: dict[str, list[tuple[str, float]]],
                      fx: dict[str, float]) -> list[tuple[str, float]]:
    """
    Value of TODAY's holdings on each past day both markets traded:
    sum(quantity × close × fx). fx maps a ticker to its ₹ rate (1 for Indian
    listings; today's USD→INR for US ones, so currency moves are left out).
    Holdings without history are skipped.
    """
    series = {}
    for h in holdings:
        hist = histories.get(h["ticker"])
        if hist and h["ticker"] in fx:
            series[h["ticker"]] = (h["quantity"] * fx[h["ticker"]], dict(hist))
    if not series:
        return []
    common = sorted(set.intersection(*(set(s[1]) for s in series.values())))
    return [(d, sum(qty_fx * s[d] for qty_fx, s in series.values())) for d in common]


def performance(values: list[tuple[str, float]]) -> dict:
    """Return, annualised volatility and largest peak-to-trough fall of a value series."""
    v = [x for _, x in values]
    if len(v) < 20:
        return {}
    daily = [b / a - 1 for a, b in zip(v, v[1:]) if a]
    mean = sum(daily) / len(daily)
    vol = math.sqrt(sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)) * math.sqrt(252) * 100
    peak, drawdown = v[0], 0.0
    for x in v:
        peak = max(peak, x)
        drawdown = min(drawdown, (x / peak - 1) * 100)
    return {"return_pct": (v[-1] / v[0] - 1) * 100, "volatility": vol, "max_drawdown": drawdown,
            "start": values[0][0], "end": values[-1][0]}


def headlines(alloc: dict, perf: dict, bench: dict[str, float], periods: dict[str, list[str]]) -> list[tuple[str, str]]:
    """
    Plain-English observations as (tone, text): tone is "good", "warn" or "info".
    Descriptive only — no "you should".
    """
    out: list[tuple[str, str]] = []
    if perf and bench:
        vs = ", ".join(f"{name} {ret:+.1f}%" for name, ret in bench.items())
        beat = [name for name, ret in bench.items() if perf["return_pct"] > ret]
        out.append(("good" if beat else "info",
                    f"Over the past year your current holdings returned {perf['return_pct']:+.1f}% "
                    f"(vs {vs}) — {'ahead of ' + ' and '.join(beat) if beat else 'behind both benchmarks'}."))
    if alloc:
        sym, pct = alloc["largest"]
        if pct >= CONCENTRATION_FLAG:
            out.append(("warn", f"{sym} is {pct:.0f}% of your portfolio — one holding drives much of your result."))
        else:
            out.append(("good", f"No single holding is more than {pct:.0f}% of your portfolio."))
        if alloc["us_pct"] >= 50:
            out.append(("info", f"{alloc['us_pct']:.0f}% of your money is in US stocks, so the dollar–rupee rate "
                                f"moves your rupee value too."))
        elif alloc["india_pct"] < 100:
            out.append(("info", f"{alloc['india_pct']:.0f}% India, {alloc['us_pct']:.0f}% US by value."))
    if perf and perf["max_drawdown"] <= -20:
        out.append(("warn", f"At its worst point in the past year this portfolio was {perf['max_drawdown']:.0f}% "
                            f"below its high."))
    recent = periods.get("under 1 year", [])
    if recent:
        out.append(("info", f"{len(recent)} holding(s) bought less than a year ago: {', '.join(recent)}."))
    return out
