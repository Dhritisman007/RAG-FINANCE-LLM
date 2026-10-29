"""
Mutual fund tests: AMFI NAV parsing and cache, pricing, and CAMS / KFintech statements. Offline.

    python tests/test_mutual_funds.py
"""
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from casparser.enums import CASFileType, FileType  # noqa: E402
from casparser.types import (CASData, Folio, InvestorInfo, Scheme, SchemeValuation,  # noqa: E402
                             StatementPeriod)

import mutual_funds as mf  # noqa: E402
import portfolio_store  # noqa: E402
import prices  # noqa: E402
from brokers.cas_adapter import holdings_from_cas  # noqa: E402

# Lines exactly as AMFI's NAVAll.txt has them (header, category, blank, fund house, schemes) — checked 29 Oct 2026
NAVALL = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Debt Scheme - Corporate Bond Fund)

HDFC Mutual Fund

118987;INF179K01XD8;-;HDFC Corporate Bond Fund;Direct Plan;Growth Option;{nav};{day}
113071;INF179K01DD0;INF179K01DE8;HDFC Corporate Bond Fund;Regular Plan;Quarterly IDCW Option;10.4322;{day}
999999;INF000X00001;-;A Suspended Fund;Direct Plan;Growth Option;N.A.;{day}
"""
GROWTH = "INF179K01XD8"


class Resp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def feed(nav="34.9214", day="29-Oct-2026", pad=6000):
    """The fixture plus `pad` filler schemes, so the file passes the too-small check like the real one (~14k)."""
    filler = "".join(f"{100000 + i};INF999Z{i:05d};-;Filler {i};Direct Plan;Growth Option;10;{day}\n" for i in range(pad))
    return lambda url, **kw: Resp(NAVALL.format(nav=nav, day=day) + filler)


def test_parse_navall() -> None:
    rows = mf.parse_navall(NAVALL.format(nav="34.9214", day="29-Oct-2026"))
    assert set(rows) == {GROWTH, "INF179K01DD0", "INF179K01DE8"}           # both ISIN columns; N.A. skipped
    assert rows[GROWTH] == {"isin": GROWTH, "scheme_code": "118987", "name": "HDFC Corporate Bond Fund",
                            "plan": "Direct Plan", "option": "Growth Option", "nav": 34.9214, "date": "2026-10-29"}
    assert rows["INF179K01DE8"]["scheme_code"] == "113071"
    print("PASS: AMFI's NAVAll.txt parsed (8 columns, both ISINs, headers / fund houses / N.A. skipped)")


def test_update_keeps_previous_nav_and_refuses_bad_files() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "amfi_nav.csv"
        assert mf.update(path, get=feed("34.9214", "28-Oct-2026")) > 6000
        mf.update(path, get=feed("35.0100", "29-Oct-2026"))                 # next day: yesterday becomes previous
        row = mf.load_navs(path)[GROWTH]
        assert (row["nav"], row["prev_nav"], row["date"], row["prev_date"]) == (35.01, 34.9214, "2026-10-29", "2026-10-28")
        mf.update(path, get=feed("35.0100", "29-Oct-2026"))                 # same day again: previous kept
        assert mf.load_navs(path)[GROWTH]["prev_nav"] == 34.9214
        q = mf.quotes(["mf:" + GROWTH.lower(), "RELIANCE.NS"], path)
        assert q == {"MF:" + GROWTH: {"price": 35.01, "prev_close": 34.9214, "as_of": "2026-10-29", "source": "AMFI NAV"}}
        assert mf.display_name("MF:" + GROWTH, path) == "HDFC Corporate Bond Fund · Direct · Growth"
        try:
            mf.update(path, get=feed(pad=0))                                 # AMFI changed its file: keep the old
        except RuntimeError as e:
            assert "keeping the old NAVs" in str(e)
        else:
            raise AssertionError("a tiny file must be refused")
        assert mf.load_navs(path)[GROWTH]["nav"] == 35.01
        assert "couldn't refresh" in mf.refresh_if_stale(Path(d) / "missing.csv", get=feed(pad=0))
    print("PASS: the NAV cache keeps the previous NAV for day change; a broken download keeps the old NAVs")


def test_funds_are_rupees_and_never_sent_to_yahoo() -> None:
    t = "MF:" + GROWTH
    assert prices.currency(t) == "INR" and prices.normalize_ticker("mf:" + GROWTH.lower()) == t
    assert prices.with_fx([t]) == [t]                                        # no USD→INR needed
    sent = []

    def download(tickers, **kw):
        sent.extend(tickers)
        raise RuntimeError("stop here")                                     # only what would be sent matters
    try:
        prices.yahoo_quotes([t, "TCS.NS"], download=download)
    except RuntimeError:
        pass
    try:
        prices.yahoo_histories([t, "^NSEI"], download=download)
    except RuntimeError:
        pass
    assert sent == ["TCS.NS", "^NSEI"], sent                                # the fund ISIN never left
    assert prices.yahoo_quotes([t], download=download) == {}
    with tempfile.TemporaryDirectory() as d:
        nav = Path(d) / "nav.csv"
        mf.update(nav, get=feed("35.0100", "29-Oct-2026"))
        px = prices.load_prices(Path(d) / "a.csv", Path(d) / "b.csv", Path(d) / "c.csv", use_snapshots=False,
                                nav_path=nav)
        assert px[t] == {"price": 35.01, "as_of": "2026-10-29", "source": "AMFI NAV"}
        v = prices.value_positions([{"ticker": t, "quantity": 100, "avg_cost": 30.0}], px, today=date(2026, 10, 29))[0]
        assert (v["currency"], v["value_inr"], v["cost_inr"], round(v["gain_pct"], 2)) == ("INR", 3501.0, 3000.0, 16.7)
    print("PASS: funds are valued in ₹ at AMFI's NAV and their ISINs are never sent to Yahoo")


def scheme(name, isin, units, nav, cost, folio_day="30-Sep-2026"):
    return Scheme(scheme=name, rta_code="X1", rta="CAMS", isin=isin, open=Decimal(0), close=Decimal(units),
                  close_calculated=Decimal(units), transactions=[],
                  valuation=SchemeValuation(date=folio_day, nav=Decimal(nav),
                                            cost=None if cost is None else Decimal(cost),
                                            value=Decimal(units) * Decimal(nav)))


def test_cams_statement_with_costs() -> None:
    data = CASData(
        statement_period=StatementPeriod(**{"from": "01-Apr-2026", "to": "30-Sep-2026"}),
        investor_info=InvestorInfo(name="Test Investor", email="t@example.com", address="x", mobile="0"),
        cas_type=CASFileType.DETAILED, file_type=FileType.CAMS,
        folios=[Folio(folio="1/11", amc="HDFC Mutual Fund", PAN="TESTP4321K", schemes=[
                    scheme("HDFC Corporate Bond Fund - Direct Growth", GROWTH, "100", "34.90", "3000"),
                    scheme("Redeemed Fund", "INF000R00001", "0", "10", "0"),
                    scheme("Old Fund", None, "5", "10", "40")]),
                Folio(folio="2/22", amc="HDFC Mutual Fund", schemes=[
                    scheme("HDFC Corporate Bond Fund - Direct Growth", GROWTH, "50", "34.90", "1800")])])
    r = holdings_from_cas(data)
    assert [(h.ticker, h.quantity, round(h.avg_cost, 4)) for h in r.holdings] == [("MF:" + GROWTH, 150.0, 32.0)]  # 4800/150
    assert r.prices[GROWTH].last_price == 34.9 and r.prices[GROWTH].as_of == "2026-09-30"
    assert r.issuer == "CAMS" and r.accounts == 2 and r.account_word == "folio(s)"
    assert r.merged == [f"{GROWTH} is in more than one folio — quantities added"]
    assert r.skipped == ["Old Fund: no ISIN on the statement, so it can't be priced"]    # redeemed: silently gone
    assert "2 folio(s)" in r.summary() and "TESTP4321K" not in r.summary()
    partial = holdings_from_cas(data.model_copy(update={"folios": [data.folios[0], Folio(
        folio="3/33", amc="x", schemes=[scheme("HDFC Corporate Bond Fund", GROWTH, "50", "34.90", None)])]}))
    assert partial.holdings[0].avg_cost is None                              # one lot's cost unknown -> unknown
    line = portfolio_store.format_position({"ticker": "MF:" + GROWTH, "quantity": 150.0, "avg_cost": 32.0,
                                            "purchase_date": "", "source": "cas:funds", "as_of": "2026-10-29T10:00"})
    assert "(mutual fund, MF:" in line and "150 units @ avg cost ₹32.00" in line and "CAS statement (funds)" in line, line
    print("PASS: a CAMS / KFintech statement becomes fund holdings with their cost (weighted across folios); "
          "answers name them in units and ₹")


if __name__ == "__main__":
    test_parse_navall()
    test_update_keeps_previous_nav_and_refuses_bad_files()
    test_funds_are_rupees_and_never_sent_to_yahoo()
    test_cams_statement_with_costs()
    print("\nAll mutual fund tests passed.")
