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
        p = prices.load_prices(market, manual)
    assert p["VBTLX"]["price"] == 10.5 and p["VBTLX"]["source"] == "manual price file"   # fund via manual file
    assert p["AAPL"]["price"] == 343.46                                                   # older manual loses
    assert p["BRK-B"]["price"] == 1300                                                    # same-day manual wins
    print("PASS: screener parsing, ticker normalization (BRK/B = BRK.B = BRK-B), manual overrides")


def test_valuation_math_stale_and_missing() -> None:
    p = {"AAPL": {"price": 343.46, "as_of": "2026-10-19T17:04", "source": "s"},
         "BRK-B": {"price": 1234.5, "as_of": "2026-10-10", "source": "s"}}
    v = prices.value_positions(HOLDINGS, p, today=date(2026, 10, 19))
    aapl, brk, fund = v
    assert round(aapl["market_value"], 2) == 8586.50 and round(aapl["gain"], 2) == 4024.00
    assert round(aapl["gain_pct"], 1) == 88.2 and not aapl["stale"]
    assert brk["stale"] and fund["missing"] and fund["market_value"] is None
    s = prices.summarize(v)
    assert round(s["market_value"], 2) == 8586.50 + 2469.00 and s["missing"] == ["VBTLX"] and s["stale"] == ["BRK.B"]
    sentence = prices.summary_sentence(v)
    assert "Not included (no price): VBTLX" in sentence and "run --update" in sentence and "not investment advice" in sentence
    print("PASS: market value / gain math, stale-price and missing-price reporting")


if __name__ == "__main__":
    test_fetch_never_sends_tickers()
    test_endpoint_change_fails_loudly()
    test_parse_normalize_and_manual_override()
    test_valuation_math_stale_and_missing()
    print("\nAll price tests passed.")
