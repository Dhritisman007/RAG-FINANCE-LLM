"""
CAS statement import tests. Offline.

No real CAS is used (they hold a PAN, a name and an address). The statement is built
from casparser's own data model and run through casparser's real ISIN enrichment;
the PDF error paths use a genuinely encrypted PDF made here, read by real casparser.

    python tests/test_cas.py
"""
import json
import sqlite3
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from casparser.enums import FileType  # noqa: E402
from casparser.parsers import _enrich_demat_equities  # noqa: E402  (the step read_cas_pdf runs after parsing)
from casparser.types import (Bond, DematAccount, DematOwner, Equity, InvestorInfo, MutualFund,  # noqa: E402
                             NSDLCASData, StatementPeriod)

import portfolio_db  # noqa: E402
import portfolio_store  # noqa: E402
import portfolio_sync  # noqa: E402
import prices  # noqa: E402
from brokers import CredentialsMissing, IncorrectPassword, MalformedResponse, get_adapter  # noqa: E402
from brokers.cas_adapter import holdings_from_cas, read_statement  # noqa: E402
from net_guard import outbound_only  # noqa: E402

ROOT = Path(__file__).parent.parent
PAN, NAME, EMAIL, MOBILE = "TESTP4321K", "Test Investor Kumar", "investor@example.com", "9800000000"
ADDRESS, DP_ID, CLIENT_ID = "12 Test Lane, Pune 411001", "IN300999", "10000042"


def statement() -> NSDLCASData:
    owner = [DematOwner(name=NAME, PAN=PAN)]
    eq = lambda name, isin, qty, price: Equity(name=name, isin=isin, num_shares=Decimal(qty), price=Decimal(price),
                                               value=Decimal(qty) * Decimal(price))
    cdsl = DematAccount(name="Some Broker", type="CDSL Demat Account", dp_id=DP_ID, client_id=CLIENT_ID, folios=5,
                        balance=Decimal("43000"), owners=owner,
                        equities=[eq("RELIANCE INDUSTRIES LIMITED", "INE002A01018", "10", "1250.00"),
                                  eq("TATA CONSULTANCY SERVICES LTD", "INE467B01029", "5", "3245.60"),
                                  eq("NIPPON INDIA ETF NIFTY BEES", "INF204KB14I2", "20", "268.10"),
                                  eq("INFOSYS LIMITED", "INE009A01021", "0", "1500")],
                        mutual_funds=[MutualFund(name="SOME LIQUID FUND", isin="INF000000001", balance=Decimal("3"),
                                                 nav=Decimal("1000"), value=Decimal("3000"))],
                        bonds=[Bond(name="SOME BOND 2030", isin="INE000000002", num_bonds=Decimal("1"),
                                    value=Decimal("1000"))])
    nsdl = DematAccount(name="Other Broker", type="NSDL Demat Account", dp_id="IN301111", client_id="20000001",
                        folios=1, balance=Decimal("6250"), owners=owner,
                        equities=[eq("RELIANCE INDUSTRIES LIMITED", "INE002A01018", "5", "1250.00")], mutual_funds=[])
    data = NSDLCASData(accounts=[cdsl, nsdl], statement_period=StatementPeriod(**{"from": "01-Sep-2026", "to": "30-Sep-2026"}),
                       investor_info=InvestorInfo(name=NAME, email=EMAIL, address=ADDRESS, mobile=MOBILE),
                       file_type=FileType.CDSL)
    return _enrich_demat_equities(data)   # symbols from casparser's bundled ISIN database, as read_cas_pdf does


def fake_reader(expected_password=PAN):
    def read(pdf, password):
        assert password == expected_password
        return statement()
    return read


def encrypted_pdf(path: Path, password: str = PAN) -> Path:
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(612, 792)
    w.encrypt(password)
    with open(path, "wb") as f:
        w.write(f)
    return path


def test_statement_to_holdings() -> None:
    with outbound_only([]):                     # mapping uses the bundled ISIN DB — no network
        r = holdings_from_cas(statement(), costs={"INE467B01029": 3600.0}, listed={})
    got = {h.ticker: (h.quantity, h.avg_cost, h.isin) for h in r.holdings}
    assert got == {"RELIANCE.NS": (15.0, None, "INE002A01018"), "TCS.NS": (5.0, 3600.0, "INE467B01029"),
                   "MF:INF000000001": (3.0, None, "INF000000001")}, got           # a fund held in demat
    assert r.prices["RELIANCE"].last_price == 1250.0 and r.prices["TCS"].as_of == "2026-09-30"
    assert r.issuer == "CDSL" and r.accounts == 2 and r.statement_date == "2026-09-30"
    assert r.merged == ["RELIANCE is in more than one demat account — quantities added"]
    skipped = " | ".join(r.skipped)
    for expected in ("INF204KB14I2): no NSE symbol known", "INE009A01021) in account 1 (CDSL Demat Account): quantity 0",
                     "bonds aren't tracked yet"):
        assert expected in skipped, skipped
    summary = r.summary()
    assert "3 holding(s) imported from the CDSL statement for 01-Sep-2026 to 30-Sep-2026 (2 demat account(s))" in summary
    assert "average price unknown (a CAS doesn't include it): RELIANCE" in summary
    for private in (PAN, NAME, EMAIL, MOBILE, DP_ID, CLIENT_ID):
        assert private not in summary + skipped
    print("PASS: a statement becomes holdings — ISIN→NSE symbol, accounts merged, demat funds kept, bond/zero rows reported, "
          "average price unknown unless entered")


def test_etfs_from_nse_list() -> None:
    """An ETF maps through NSE's ETF list, whether the statement puts it under equities or mutual funds."""
    data = statement()
    data.accounts[1].mutual_funds.append(MutualFund(name="NIPPON INDIA ETF NIFTY BEES", isin="INF204KB14I2",
                                                    balance=Decimal("4"), nav=Decimal("268.10"), value=Decimal("1072.4")))
    r = holdings_from_cas(data, listed={"INF204KB14I2": "NIFTYBEES"})
    got = {h.ticker: h.quantity for h in r.holdings}
    assert got == {"RELIANCE.NS": 15.0, "TCS.NS": 5.0, "NIFTYBEES.NS": 24.0, "MF:INF000000001": 3.0}, got  # 20+4
    assert r.prices["NIFTYBEES"].last_price == 268.10
    assert not any("INF204KB14I2" in s for s in r.skipped)
    import stock_list
    rows = stock_list.parse_nse_etf("Symbol,Underlying Asset,SecurityName,DateofListing,MarketLot,ISINNumber,FaceValue\n"
                                    "NIFTYBEES,Nifty 50,NIPINDETFNIFTYBEES,08-Jan-02,1,INF204KB14I2,1\n")
    eq = stock_list.parse_nse_equity("SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, "
                                     "ISIN NUMBER, FACE VALUE\nTCS,Tata Consultancy Services Limited,EQ,25-AUG-2004,1,1,"
                                     "INE467B01029,1\n")
    assert rows[0]["isin"] == "INF204KB14I2" and eq[0]["isin"] == "INE467B01029"   # NSE's real header names
    renamed = holdings_from_cas(statement(), listed={"INE467B01029": "TCSNEW"})      # NSE's current symbol wins
    assert {h.symbol for h in renamed.holdings} == {"RELIANCE", "TCSNEW", "INF000000001"}
    print("PASS: ETFs map via NSE's ETF list (under equities or mutual funds); stock-list rows carry ISINs")


def test_sync_writes_no_personal_data() -> None:
    with tempfile.TemporaryDirectory() as d:
        db, pdf = Path(d) / "p.db", Path(d) / "statement.pdf"
        pdf.write_bytes(b"%PDF-1.4 stand-in; the reader is faked")
        adapter = get_adapter("cas", file=pdf, password=PAN, reader=fake_reader(), listed={})
        res = portfolio_sync.sync("cas", db_path=db, adapter=adapter, label="mom")
        assert res.ok and res.source == "cas:mom" and res.holdings_count == 3 and res.prices_status == "ok", res
        blob = db.read_bytes() + json.dumps(res.__dict__).encode() + repr(adapter).encode()
        for private in (PAN, NAME, EMAIL, MOBILE, ADDRESS, DP_ID, CLIENT_ID):
            assert private.encode() not in blob, f"{private} was stored"
        rows = dict(sqlite3.connect(db).execute("SELECT ticker, avg_cost FROM holdings").fetchall())
        assert rows == {"RELIANCE.NS": None, "TCS.NS": None, "MF:INF000000001": None}   # unknown, not 0
        held = portfolio_store.load_snapshot_holdings(db)
        line = portfolio_store.format_position(next(h for h in held if h["ticker"] == "TCS.NS"))
        assert "5 shares (average cost not known" in line and "from CAS statement (mom), as of" in line, line
    print("PASS: a CAS sync stores holdings only — no PAN, name, contact or account numbers; cost stays unknown")


def test_real_casparser_error_paths() -> None:
    with tempfile.TemporaryDirectory() as d, outbound_only([]):
        pdf = encrypted_pdf(Path(d) / "locked.pdf")
        for pdf_in, password, error, text in [
                (pdf, "WRONG9999Z", IncorrectPassword, "wrong password — a CAS opens with the first holder's PAN"),
                (pdf, PAN, MalformedResponse, "doesn't read as a CAS from NSDL, CDSL, CAMS or KFintech"),   # opens, but isn't a CAS
                (Path(d) / "x.pdf", PAN, MalformedResponse, "couldn't open this file as a PDF"),
                (pdf, "", CredentialsMissing, "password is needed")]:
            if pdf_in.name == "x.pdf":
                pdf_in.write_bytes(b"not a pdf at all")
            try:
                read_statement(pdf_in, password)
            except error as e:
                assert text in str(e), str(e)
                assert not password or password not in str(e)
            else:
                raise AssertionError(f"expected {error.__name__}")
    print("PASS: real casparser on an encrypted PDF: wrong password, not-a-CAS and not-a-PDF are typed errors "
          "that never repeat the password; no network")


def test_cli_reads_password_from_stdin() -> None:
    script = ROOT / "scripts" / "portfolio_sync.py"
    with tempfile.TemporaryDirectory() as d:
        db, pdf = Path(d) / "p.db", encrypted_pdf(Path(d) / "locked.pdf")
        run = lambda password, *extra: subprocess.run(
            [sys.executable, str(script), "--source", "cas", "--file", str(pdf), "--password-stdin", "--json",
             "--db", str(db), *extra], input=password + "\n", capture_output=True, text=True, cwd=ROOT, timeout=120)
        out = run("WRONG9999Z")
        res = json.loads(out.stdout.strip().splitlines()[-1])
        assert out.returncode == 1 and res["error_type"] == "IncorrectPassword", out.stdout + out.stderr
        out2 = run(PAN)
        assert json.loads(out2.stdout)["error_type"] == "MalformedResponse"
        log = " ".join(str(r) for r in sqlite3.connect(db).execute("SELECT * FROM sync_log"))
        for secret in ("WRONG9999Z", PAN):
            assert secret not in out.stdout + out.stderr + out2.stdout + out2.stderr + log
        bad = Path(d) / "costs.json"
        bad.write_text('{"INE002A01018": -5}')
        assert run(PAN, "--costs", str(bad)).returncode == 2               # refused before anything is read
    print("PASS: the CLI takes the CAS password on stdin (never argv); failures are logged without it")


def test_old_database_is_migrated() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript(portfolio_db.SCHEMA.replace("avg_cost REAL CHECK (avg_cost IS NULL OR avg_cost >= 0)",
                                                       "avg_cost REAL NOT NULL"))
        conn.execute("INSERT INTO snapshots VALUES (1, 'csv', '2026-10-01T10:00:00', 1, 'x')")
        conn.execute("INSERT INTO holdings VALUES (1,'TCS','NSE','TCS.NS',NULL,5,3600,'INR',NULL,NULL)")
        conn.commit()
        conn.close()
        conn = portfolio_db.connect(db)
        try:
            assert not [c for c in conn.execute("PRAGMA table_info(holdings)") if c["name"] == "avg_cost"][0]["notnull"]
            assert [tuple(r) for r in conn.execute("SELECT ticker, avg_cost FROM holdings")] == [("TCS.NS", 3600.0)]
            conn.execute("INSERT INTO holdings VALUES (1,'INFY','NSE','INFY.NS',NULL,2,NULL,'INR',NULL,NULL)")
        finally:
            conn.close()
        assert [s["holdings_count"] for s in portfolio_db.latest_snapshots(db)] == [1]   # still readable
    print("PASS: a database from before CAS import is migrated once (rows kept) so an unknown cost can be stored")


def test_partial_cost_valuation_is_honest() -> None:
    held = [{"ticker": "TCS.NS", "quantity": 5, "avg_cost": 3600.0, "source": "cas", "as_of": "2026-10-23T10:00:00"},
            {"ticker": "RELIANCE.NS", "quantity": 10, "avg_cost": None, "source": "cas", "as_of": "2026-10-23T10:00:00"}]
    px = {"TCS.NS": {"price": 3000.0, "as_of": "2026-10-23", "source": "test"},
          "RELIANCE.NS": {"price": 1200.0, "as_of": "2026-10-23", "source": "test"}}
    from datetime import date
    v = prices.value_positions(held, px, today=date(2026, 10, 23))
    s = prices.summarize(v)
    assert (s["market_value"], s["cost_basis"], s["gain"], s["no_cost"]) == (27000.0, 18000.0, -3000.0, ["RELIANCE.NS"])
    lines = prices.format_valuation(v)
    assert "(cost not known, so no gain/loss)" in lines[1] and "cost ₹18,000.00, -₹3,000.00, -16.7%" in lines[0], lines
    text = prices.summary_sentence(v)
    assert "against a cost basis" not in text                     # never worth (all) vs cost (some)
    assert "Those with a known average cost (₹18,000.00 invested) are at -₹3,000.00 (-16.7%)" in text, text
    assert "Average cost isn't known for RELIANCE.NS, so the gain/loss leaves them out" in text, text
    only_unknown = prices.summary_sentence(v[1:])
    assert "there is no gain/loss figure" in only_unknown and "₹0" not in only_unknown, only_unknown
    print("PASS: unknown average costs are left out of gain/loss and said so — never counted as zero")


if __name__ == "__main__":
    test_statement_to_holdings()
    test_etfs_from_nse_list()
    test_sync_writes_no_personal_data()
    test_real_casparser_error_paths()
    test_cli_reads_password_from_stdin()
    test_old_database_is_migrated()
    test_partial_cost_valuation_is_honest()
    print("\nAll CAS tests passed.")
