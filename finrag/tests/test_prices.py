"""
Price module tests. No network: the screener request is captured, not sent.

    python tests/test_prices.py
"""
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import prices  # noqa: E402

HOLDINGS = [
    {"ticker": "AAPL", "quantity": 25.0, "avg_cost": 182.5, "purchase_date": "2024-03-15"},
    {"ticker": "BRK.B", "quantity": 2.0, "avg_cost": 400.0, "purchase_date": "2024-01-02"},
    {"ticker": "VBTLX", "quantity": 10.0, "avg_cost": 10.0, "purchase_date": "2024-01-02"},  # a fund: not in the screener
]
SCREENER = {"data": {"rows": [
    {"symbol": "AAPL", "lastsale": "$343.46"},
    {"symbol": "BRK/B", "lastsale": "$1,234.50"},
    {"symbol": "XYZ", "lastsale": "NA"},
] + [{"symbol": f"T{i}", "lastsale": "$1.00"} for i in range(1500)]}}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_fetch_never_sends_tickers() -> None:
    sent = []

    def fake_get(url, **kwargs):
        sent.append((url, kwargs))
        return FakeResponse(SCREENER)

    with tempfile.TemporaryDirectory() as tmp:
        n = prices.fetch_market_prices(Path(tmp) / "market.csv", get=fake_get)
    assert len(sent) == 1 and n == 1502  # one request; "NA" row skipped
    url, kwargs = sent[0]
    assert url == prices.SCREENER_URL and "params" not in kwargs and "data" not in kwargs
    request_text = url + str(kwargs)
    for h in HOLDINGS:
        assert h["ticker"].lower() not in request_text.lower(), f"{h['ticker']} leaked into the request"
    print("PASS: one whole-market request with a fixed URL — no holding ever appears in it")


def test_endpoint_change_fails_loudly() -> None:
    def tiny(url, **kwargs):
        return FakeResponse({"data": {"rows": [{"symbol": "AAPL", "lastsale": "$1"}]}})
    with tempfile.TemporaryDirectory() as tmp:
        try:
            prices.fetch_market_prices(Path(tmp) / "m.csv", get=tiny)
        except RuntimeError:
            print("PASS: a suspiciously small download raises instead of saving a broken price file")
            return
    raise AssertionError("expected RuntimeError")


def test_parse_normalize_and_manual_override() -> None:
    parsed = prices.parse_screener(SCREENER)
    assert parsed["AAPL"] == 343.46 and parsed["BRK-B"] == 1234.5 and "XYZ" not in parsed
    with tempfile.TemporaryDirectory() as tmp:
        market, manual = Path(tmp) / "market.csv", Path(tmp) / "manual.csv"
        market.write_text("ticker,price,as_of,source\nAAPL,343.46,2026-10-19T17:04,screener\n"
                          "BRK-B,1234.5,2026-10-19T17:04,screener\n")
        manual.write_text("ticker,price,date\nvbtlx,10.50,2026-10-18\nAAPL,300,2026-10-01\nBRK.B,1300,2026-10-19\n")
        p = prices.load_prices(market, manual, Path(tmp) / "no_yahoo.csv")  # never read real price files
    assert p["VBTLX"]["price"] == 10.5 and p["VBTLX"]["source"] == "manual price file"   # fund via manual file
    assert p["AAPL"]["price"] == 343.46                                                   # older manual loses
    assert p["BRK-B"]["price"] == 1300                                                    # same-day manual wins
    print("PASS: screener parsing, ticker normalization (BRK/B = BRK.B = BRK-B), manual overrides")


def test_valuation_math_stale_and_missing() -> None:
    p = {"AAPL": {"price": 343.46, "as_of": "2026-10-19T17:04", "source": "s"},
         "BRK-B": {"price": 1234.5, "as_of": "2026-10-10", "source": "s"},
         prices.FX_TICKER: {"price": 84.0, "as_of": "2026-10-19", "source": "s"}}
    v = prices.value_positions(HOLDINGS, p, today=date(2026, 10, 19))
    aapl, brk, fund = v
    assert round(aapl["market_value"], 2) == 8586.50 and round(aapl["gain"], 2) == 4024.00
    assert round(aapl["gain_pct"], 1) == 88.2 and not aapl["stale"] and aapl["currency"] == "USD"
    assert round(aapl["value_inr"], 2) == round(8586.50 * 84, 2)                 # US holdings in ₹ at the FX rate
    assert brk["stale"] and fund["missing"] and fund["market_value"] is None
    s = prices.summarize(v)
    assert round(s["market_value"], 2) == round((8586.50 + 2469.00) * 84, 2)
    assert s["missing"] == ["VBTLX"] and s["stale"] == ["BRK.B"]
    sentence = prices.summary_sentence(v)
    assert "Not included (no price): VBTLX" in sentence and "run --update" in sentence and "not investment advice" in sentence
    assert "₹9,28,662.00" in sentence and "converted to rupees" in sentence
    no_fx = prices.summarize(prices.value_positions(HOLDINGS, {k: x for k, x in p.items() if k != prices.FX_TICKER},
                                                    today=date(2026, 10, 19)))
    assert no_fx["priced"] == 0 and no_fx["no_fx"] == ["AAPL", "BRK.B"]           # never mixes $ into ₹ totals
    print("PASS: market value / gain math, ₹ totals via USD→INR, stale-price and missing-price reporting")


def test_indian_listings_and_rupees() -> None:
    assert prices.normalize_ticker("reliance.ns") == "RELIANCE.NS" and prices.normalize_ticker("m&m.bo") == "M&M.BO"
    assert prices.normalize_ticker("brk.b") == "BRK-B"                            # US class shares unchanged
    assert prices.with_exchange("tcs", "NSE") == "TCS.NS" and prices.with_exchange("tcs", "BSE") == "TCS.BO"
    assert prices.with_exchange("INFY.BO", "NSE") == "INFY.BO" and prices.with_exchange("aapl", "US") == "AAPL"
    assert prices.display_symbol("RELIANCE.NS") == ("RELIANCE", "NSE") and prices.display_symbol("AAPL") == ("AAPL", "US")
    assert prices.inr(1234567.8) == "₹12,34,567.80" and prices.inr(999) == "₹999.00" and prices.inr(-123456) == "-₹1,23,456.00"
    assert prices.signed_money(2500.5) == "+₹2,500.50" and prices.money(1234.5, "USD") == "$1,234.50"
    assert prices.inr(1098872.11, 0) == "₹10,98,872" and prices.signed_money(-549.9, decimals=0) == "-₹550"
    assert prices.with_fx(["TCS.NS"]) == ["TCS.NS"] and prices.with_fx(["TCS.NS", "AAPL"])[-1] == prices.FX_TICKER
    holdings = [{"ticker": "RELIANCE.NS", "quantity": 10, "avg_cost": 2400.0, "purchase_date": ""}]
    v = prices.value_positions(holdings, {"RELIANCE.NS": {"price": 2850.0, "as_of": "2026-10-19", "source": "s"}},
                               today=date(2026, 10, 19))
    assert v[0]["currency"] == "INR" and v[0]["value_inr"] == 28500.0           # no FX needed for ₹ listings
    line = prices.format_valuation(v)[0]
    assert "10 × ₹2,850.00 = ₹28,500.00" in line and "+₹4,500.00" in line, line
    print("PASS: NSE/BSE suffixes, ₹ with Indian digit grouping, rupee valuation without FX")


def test_yahoo_prices_and_source_precedence() -> None:
    import pandas as pd
    asked = []

    def fake_download(tickers, **kwargs):
        asked.append(list(tickers))
        idx = pd.to_datetime(["2026-10-16", "2026-10-19"])
        cols = pd.MultiIndex.from_product([["Close"], ["AAPL", "VBTLX", "GONE"]])
        return pd.DataFrame([[343.46, 9.32, float("nan")], [342.94, 9.29, float("nan")]], index=idx, columns=cols)

    with tempfile.TemporaryDirectory() as tmp:
        yahoo, market, manual = Path(tmp) / "y.csv", Path(tmp) / "m.csv", Path(tmp) / "man.csv"
        n, missing = prices.fetch_yahoo_prices(["aapl", "VBTLX", "GONE", "AAPL"], yahoo, download=fake_download)
        assert asked[0] == ["AAPL", "GONE", "USDINR=X", "VBTLX"]  # holdings (normalised, de-duplicated) + the FX pair
        assert n == 2 and missing == ["GONE"]
        q = prices.yahoo_quotes(["AAPL"], download=fake_download)["AAPL"]       # live panel: day change
        assert q == {"price": 342.94, "prev_close": 343.46, "as_of": "2026-10-19"}
        market.write_text("ticker,price,as_of,source\nAAPL,343.46,2026-10-19T17:04,screener\n")
        p = prices.load_prices(market, manual, yahoo)
        assert p["AAPL"]["price"] == 342.94 and p["AAPL"]["source"] == prices.YAHOO_SOURCE   # same date: Yahoo > Nasdaq
        assert p["VBTLX"]["price"] == 9.29                                                    # a fund Nasdaq doesn't list
        manual.write_text("ticker,price,date\nAAPL,350,2026-10-19\n")
        assert prices.load_prices(market, manual, yahoo)["AAPL"]["price"] == 350            # same date: manual wins
    print("PASS: Yahoo closes for holdings (stocks and funds); same-date precedence manual > Yahoo > Nasdaq")


def test_history_and_stats() -> None:
    import pandas as pd
    idx = pd.bdate_range("2025-10-01", periods=260)
    closes = [100 + i * 0.5 for i in range(200)] + [199.5 - i * 2 for i in range(60)]   # rise, then a fall

    def fake_download(tickers, **kwargs):
        assert tickers == ["RELIANCE.NS"] and kwargs["period"] == "1y"
        return pd.DataFrame({("Close", "RELIANCE.NS"): closes}, index=idx)

    h = prices.yahoo_history("reliance.ns", download=fake_download)
    assert len(h) == 260 and h[0] == ("2025-10-01", 100.0) and h[-1][1] == closes[-1]
    s = prices.price_stats(h)
    assert s["high_52w"] == 199.5 and s["low_52w"] == min(closes[-252:]) and s["last"] == closes[-1]
    assert round(s["max_drawdown"], 2) == round((closes[-1] / 199.5 - 1) * 100, 2)
    assert round(s["ret_1m"], 2) == round((closes[-1] / closes[-22] - 1) * 100, 2) and s["volatility"] > 0
    assert prices.price_stats([("2026-01-01", 1.0)]) == {}
    print("PASS: price history, 52-week range, returns, volatility and max drawdown")


def test_histories_one_request() -> None:
    import pandas as pd
    asked = []

    def fake_download(tickers, **kwargs):
        asked.append(list(tickers))
        idx = pd.bdate_range("2026-01-01", periods=3)
        cols = pd.MultiIndex.from_product([["Close"], ["TCS.NS", "^NSEI", "GONE"]])
        return pd.DataFrame([[3500, 25000, float("nan")], [3510, 25100, float("nan")], [3490, 24900, float("nan")]],
                            index=idx, columns=cols)

    h = prices.yahoo_histories(["tcs.ns", "^NSEI", "GONE", "TCS.NS"], download=fake_download)
    assert asked == [["GONE", "TCS.NS", "^NSEI"]]                  # one request, de-duplicated, index kept as is
    assert h["TCS.NS"][0] == ("2026-01-01", 3500.0) and len(h["^NSEI"]) == 3 and "GONE" not in h
    print("PASS: several histories (holdings + index benchmarks) in one request")


if __name__ == "__main__":
    test_fetch_never_sends_tickers()
    test_endpoint_change_fails_loudly()
    test_parse_normalize_and_manual_override()
    test_valuation_math_stale_and_missing()
    test_indian_listings_and_rupees()
    test_yahoo_prices_and_source_precedence()
    test_history_and_stats()
    test_histories_one_request()
    print("\nAll price tests passed.")
