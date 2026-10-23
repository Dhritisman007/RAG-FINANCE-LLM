"""
Portfolio sync + snapshot tests. A temporary SQLite DB and fake broker sessions — offline.

    python tests/test_portfolio_sync.py
"""
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import portfolio_db  # noqa: E402
import portfolio_store  # noqa: E402
import portfolio_sync  # noqa: E402
from brokers import get_adapter  # noqa: E402
from brokers.base import BrokerAdapter, Holding, AuthStatus, Price  # noqa: E402
from test_brokers import FIX, ROOT, SPECS, TOKEN, FakeSession, fixture  # noqa: E402

MAPPING = ROOT / "config" / "broker_mappings" / "generic_example.json"


def rows(db, sql, *args):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def upstox(*replies):
    return get_adapter("upstox", token=TOKEN, session=FakeSession(*replies), sleep=lambda s: None)


def test_csv_sync_writes_snapshot_and_answers_cite_it() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "portfolio.db"
        res = portfolio_sync.sync("csv", db_path=db, file=FIX / "sample_export.csv", mapping=MAPPING)
        assert res.ok and res.holdings_count == 3 and res.prices_status == "not supported by this source"
        assert "SKIPPED" in res.report and "negative quantity" in res.report     # the import report is surfaced
        held = portfolio_store.load_snapshot_holdings(db)
        assert {h["ticker"] for h in held} == {"RELIANCE.NS", "TCS.NS", "AAPL"}
        assert all(h["source"] == "csv" and h["as_of"] == res.fetched_at for h in held)
        line = portfolio_store.format_position(next(h for h in held if h["ticker"] == "TCS.NS"))
        assert f"from CSV import, as of {res.fetched_at}" in line, line          # answers carry source + as_of
        assert rows(db, "SELECT status, holdings_count FROM sync_log") == [("ok", 3)]
    print("PASS: CSV sync writes a snapshot; every position from it states its source and as-of time")


def test_failed_sync_keeps_last_good_snapshot() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "portfolio.db"
        spec = SPECS["upstox"]
        good = portfolio_sync.sync("upstox", db_path=db,
                                   adapter=upstox((200, spec["holdings"]), (200, spec["prices"])))
        assert good.ok and good.prices_status == "ok"
        for failing in (upstox(spec["auth_error"]), upstox(*[ConnectionError("down")] * 4)):
            bad = portfolio_sync.sync("upstox", db_path=db, adapter=failing)
            assert not bad.ok and bad.error_type in ("AuthExpired", "BrokerUnavailable")
        snap = portfolio_db.latest_snapshots(db)
        assert len(snap) == 1 and snap[0]["fetched_at"] == good.fetched_at and snap[0]["holdings_count"] == 2
        log = rows(db, "SELECT status, error_type FROM sync_log ORDER BY id")
        assert log == [("ok", None), ("failed", "AuthExpired"), ("failed", "BrokerUnavailable")], log
    print("PASS: a failed sync (expired token, network down) is logged and the last good snapshot is untouched")


def test_partial_prices_are_null_never_zero() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "portfolio.db"
        spec = SPECS["dhan"]
        dhan = get_adapter("dhan", token=TOKEN, client_id="1", sleep=lambda s: None,
                           session=FakeSession((200, spec["holdings"]), (200, spec["prices"])))
        res = portfolio_sync.sync("dhan", db_path=db, adapter=dhan)
        assert res.ok and res.prices_status == "partial — unavailable for HDFC", res.prices_status
        got = dict(rows(db, "SELECT symbol, last_price FROM holdings"))
        assert got == {"HDFC": None, "TCS": 3245.6}, got                          # NULL, not 0
        # price API failing entirely: holdings still saved, every price unavailable
        dhan = get_adapter("dhan", token=TOKEN, client_id="1", sleep=lambda s: None,
                           session=FakeSession((200, spec["holdings"]), *[(500, None)] * 4))
        res = portfolio_sync.sync("dhan", db_path=db, adapter=dhan)
        assert res.ok and res.prices_status.startswith("unavailable — BrokerUnavailable")
        latest = portfolio_db.latest_snapshots(db)[0]
        assert latest["holdings_count"] == 2 and all(h["last_price"] is None for h in latest["holdings"])
        try:
            sqlite3.connect(db).execute("INSERT INTO holdings VALUES (1,'X','NSE','X.NS',NULL,1,1,'INR',0,NULL)")
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("a zero price must be impossible to store")
    print("PASS: missing prices are stored as unavailable (NULL — zero is rejected by the schema); holdings kept")


def test_dry_run_and_atomic_rollback() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "portfolio.db"
        res = portfolio_sync.sync("csv", db_path=db, dry_run=True, file=FIX / "sample_export.csv", mapping=MAPPING)
        assert res.ok and res.dry_run and not db.exists()                         # nothing written
        portfolio_sync.sync("csv", db_path=db, file=FIX / "sample_export.csv", mapping=MAPPING)
        before = portfolio_db.latest_snapshots(db)

        class Broken(BrokerAdapter):  # returns a price that violates the schema mid-write
            name, allowed_hosts = "csv", ()

            def fetch_holdings(self):
                return [Holding("ZZZ", "NSE", 1, 1, "INR", "csv"), Holding("YYY", "NSE", 1, 1, "INR", "csv")]

            def fetch_prices(self, symbols):
                return {"YYY": Price("YYY", -5.0, "2026-10-23T10:00:00")}

            def auth_status(self):
                return AuthStatus.OK
        try:
            portfolio_sync.sync("csv", db_path=db, adapter=Broken())
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("expected the write to fail")
        assert portfolio_db.latest_snapshots(db) == before                        # rolled back completely
    print("PASS: --dry-run writes nothing; a write that fails half-way rolls back and the previous snapshot survives")


def test_status_and_agent_integration() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "portfolio.db"
        spec = SPECS["upstox"]
        portfolio_sync.sync("upstox", db_path=db, adapter=upstox((200, spec["holdings"]), (200, spec["prices"])))
        st = {r["source"]: r for r in portfolio_sync.status(db_path=db, online=False)}
        assert set(st) == {"csv", "upstox", "dhan", "groww"} and st["upstox"]["snapshot"].startswith("2 holdings")
        assert st["upstox"]["last_sync"] == "ok" and st["dhan"]["snapshot"] == "none"
        # the agent reads synced holdings through the same functions as before
        real = portfolio_db.DB_PATH
        portfolio_db.DB_PATH = db
        try:
            import rag_loop as r
            import prices
            held = portfolio_store.find_held_tickers("What risks does my RELIANCE position face?",
                                                     portfolio_store.load_holdings())
            assert "RELIANCE.NS" in held
            d_, plan, positions = r.plan_query("What risks does my RELIANCE position face?")
            ups = [p for p in positions if p.get("source") == "upstox"]
            assert ups and "from Upstox, as of" in portfolio_store.format_position(ups[0])
            p = prices.load_prices(Path(d) / "x.csv", Path(d) / "y.csv", Path(d) / "z.csv")
            assert p["RELIANCE.NS"]["price"] == 1190.4 and p["RELIANCE.NS"]["source"] == "Upstox last price"
            v = prices.value_positions(ups, p)
            assert "holding from Upstox, as of" in prices.format_valuation(v)[0]
            assert "Upstox snapshot as of" in prices.summary_sentence(v)
        finally:
            portfolio_db.DB_PATH = real
    print("PASS: --status shows auth and snapshot age; the router, prompts and valuation read synced holdings with source + as-of")


if __name__ == "__main__":
    test_csv_sync_writes_snapshot_and_answers_cite_it()
    test_failed_sync_keeps_last_good_snapshot()
    test_partial_prices_are_null_never_zero()
    test_dry_run_and_atomic_rollback()
    test_status_and_agent_integration()
    print("\nAll portfolio sync tests passed.")
