"""
Stock-list tests (the add-stock search). No network: the downloads are faked.

    python tests/test_stock_list.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import stock_list  # noqa: E402

NSE_EQUITY = ("SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING\n"
              "RELIANCE,Reliance Industries Limited,EQ,29-NOV-1995\n"
              "M&M,Mahindra & Mahindra Limited,EQ,08-NOV-1995\n"
              "BAJAJ-AUTO,Bajaj Auto Limited,EQ,26-MAY-2008\n")
NSE_ETF = ("Symbol,Underlying Asset,SecurityName,DateofListing\n"
           "NIFTYBEES,Nifty 50,NIPINDETFNIFTYBEES,08-Jan-02\n")
SEC = {"1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
       "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
       "2": {"cik_str": 1067983, "ticker": "BRK.B", "title": "BERKSHIRE HATHAWAY INC"},
       "3": {"cik_str": 1, "ticker": "AAPL", "title": "duplicate"}}


class Resp:
    def __init__(self, text="", payload=None):
        self.text, self.payload = text, payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_parsing_and_yahoo_tickers() -> None:
    eq = stock_list.parse_nse_equity(NSE_EQUITY)
    assert [r["ticker"] for r in eq] == ["RELIANCE.NS", "M&M.NS", "BAJAJ-AUTO.NS"]
    assert eq[0]["name"] == "Reliance Industries Limited" and eq[0]["exchange"] == "NSE"
    etf = stock_list.parse_nse_etf(NSE_ETF)
    assert etf == [{"ticker": "NIFTYBEES.NS", "symbol": "NIFTYBEES", "name": "Nifty 50 ETF", "exchange": "NSE ETF"}]
    us = stock_list.parse_sec(SEC)
    assert [r["ticker"] for r in us] == ["NVDA", "AAPL", "BRK-B"]   # size order kept, duplicate dropped, BRK.B -> BRK-B
    print("PASS: NSE equities/ETFs and SEC tickers parsed into Yahoo tickers (RELIANCE.NS, M&M.NS, BRK-B)")


def test_update_writes_list_and_refuses_tiny_downloads() -> None:
    def get(url, **kwargs):
        if url == stock_list.SEC_TICKERS_URL:
            assert "@" in kwargs["headers"]["User-Agent"]                 # SEC asks for a contact address
            return Resp(payload={str(i): {"ticker": f"T{i}", "title": f"Co {i}"} for i in range(6000)})
        assert "Mozilla" not in kwargs["headers"]["User-Agent"]          # honest label for NSE
        return Resp(NSE_EQUITY if url == stock_list.NSE_EQUITY_URL else NSE_ETF)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "stocks.csv"
        n = stock_list.update(path, get=get)
        rows = stock_list.load_stocks(path)
        assert n == len(rows) == 3 + 1 + 6000 and rows[0]["ticker"] == "RELIANCE.NS"   # Indian listings first
        before = path.read_text()
        try:
            stock_list.update(path, get=lambda url, **kw: Resp(NSE_EQUITY, payload={}) if "sec" in url
                              else Resp(NSE_EQUITY if "EQUITY" in url else NSE_ETF))
        except RuntimeError:
            pass
        else:
            raise AssertionError("a suspiciously small download must not replace the list")
        assert path.read_text() == before
    print("PASS: list written Indian-first; a tiny download (source changed) keeps the old list")


def test_choice_to_ticker_and_search() -> None:
    rows = stock_list.parse_nse_equity(NSE_EQUITY) + stock_list.parse_sec(SEC)
    labels = [stock_list.label(r) for r in rows]
    by_label = dict(zip(labels, (r["ticker"] for r in rows)))
    assert labels[0] == "RELIANCE — Reliance Industries Limited · NSE"
    assert stock_list.ticker_for(labels[0], by_label) == "RELIANCE.NS"
    assert stock_list.ticker_for("BRK-B — BERKSHIRE HATHAWAY INC · US", by_label) == "BRK-B"
    # typed, not in the list: NSE by default, a suffix is kept
    assert stock_list.ticker_for("tatamotors", by_label) == "TATAMOTORS.NS"
    assert stock_list.ticker_for("TCS.BO", by_label) == "TCS.BO"
    assert stock_list.ticker_for(None, by_label) == "" and stock_list.ticker_for("  ", by_label) == ""
    assert stock_list.search("mahindra", rows)[0].startswith("M&M — ")          # by company name
    assert stock_list.search("b", rows)[0].startswith("BAJAJ-AUTO")             # symbol prefixes first
    print("PASS: dropdown choice -> Yahoo ticker; typed symbols default to NSE; search by symbol or name")


if __name__ == "__main__":
    test_parsing_and_yahoo_tickers()
    test_update_writes_list_and_refuses_tiny_downloads()
    test_choice_to_ticker_and_search()
    print("\nAll stock-list tests passed.")
