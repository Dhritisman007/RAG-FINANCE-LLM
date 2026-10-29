"""
Hosted-mode isolation tests (the website): every user's data in their own folder,
nothing personal read without a signed-in user, no broker tokens. Offline.

    python tests/test_workspace.py
"""
import contextlib
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import portfolio_db  # noqa: E402
import portfolio_store  # noqa: E402
import portfolio_sync  # noqa: E402
import workspace  # noqa: E402
from brokers.credentials import get_secret  # noqa: E402
from test_cas import PAN, fake_reader  # noqa: E402

ROOT = Path(__file__).parent.parent


@contextlib.contextmanager
def hosted(data_dir: Path):
    old = {k: os.environ.get(k) for k in ("PORTIQ_MODE", "PORTIQ_DATA_DIR")}
    os.environ.update(PORTIQ_MODE="hosted", PORTIQ_DATA_DIR=str(data_dir))
    try:
        yield
    finally:
        workspace.use(None)
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_local_mode_is_unchanged() -> None:
    workspace.use(None)
    assert not workspace.hosted() and workspace.current() is None
    assert portfolio_store.holdings_path() == portfolio_store.HOLDINGS_PATH
    assert portfolio_db.db_path() == portfolio_db.DB_PATH and workspace.cache_scope() == "local"
    print("PASS: local mode keeps data/portfolio/ paths exactly as before")


def test_nobody_signed_in_reads_nothing() -> None:
    with tempfile.TemporaryDirectory() as d, hosted(Path(d)):
        workspace.use(None)
        for read in (portfolio_store.load_holdings, portfolio_store.load_manual_holdings, portfolio_db.latest_snapshots,
                     lambda: portfolio_store.add_holding({"ticker": "TCS.NS", "quantity": 1, "avg_cost": 1})):
            try:
                read()
            except workspace.NoWorkspace:
                pass
            else:
                raise AssertionError(f"{read} read personal data with nobody signed in")
        assert workspace.cache_scope() == "nobody"
        assert not any(Path(d).rglob("*"))                               # nothing was written either
    print("PASS: hosted mode with nobody signed in refuses to read or write any holdings (no shared fallback)")


def test_two_users_never_see_each_other() -> None:
    with tempfile.TemporaryDirectory() as d, hosted(Path(d)):
        a, b = workspace.for_user("https://accounts.google.com", "111"), workspace.for_user("https://accounts.google.com", "222")
        assert a.root != b.root and a.root.parent == b.root.parent == Path(d) / "users"
        assert "111" not in str(a.root) and len(a.user_key) == 32           # a one-way name, not the ID
        workspace.use(a)
        assert portfolio_store.add_holding({"ticker": "AAPL", "quantity": 2, "avg_cost": 150}) == []
        pdf = Path(d) / "s.pdf"
        pdf.write_bytes(b"%PDF stand-in")
        res = portfolio_sync.sync("cas", adapter=__import__("brokers").get_adapter(
            "cas", file=pdf, password=PAN, reader=fake_reader(), listed={}))   # no db_path: the user's own
        assert res.ok and a.db.exists() and not b.db.exists()
        workspace.use(b)
        assert portfolio_store.load_holdings() == [] and portfolio_db.latest_snapshots() == []
        assert portfolio_store.add_holding({"ticker": "TCS.NS", "quantity": 5, "avg_cost": 3600}) == []
        import rag_loop as r
        _, _, positions = r.plan_query("What risks does my RELIANCE position face?")
        assert not positions                                               # A's CAS holdings aren't B's
        workspace.use(a)
        tickers = {h["ticker"] for h in portfolio_store.load_holdings()}
        assert tickers == {"AAPL", "RELIANCE.NS", "TCS.NS", "MF:INF000000001"}, tickers   # A: manual AAPL + CAS
        assert [h["ticker"] for h in portfolio_store.load_manual_holdings()] == ["AAPL"]   # B's TCS isn't here
        _, _, positions = r.plan_query("What risks does my RELIANCE position face?")
        assert [p["ticker"] for p in positions] == ["RELIANCE.NS"]
        assert workspace.cache_scope() == a.user_key != b.user_key
    print("PASS: two users' holdings, snapshots and answer context stay apart; folders are named by a one-way hash")


def test_threads_start_as_nobody() -> None:
    """Streamlit runs each session in its own thread: a user set in one must not leak into another."""
    with tempfile.TemporaryDirectory() as d, hosted(Path(d)):
        workspace.use(workspace.for_user("g", "111"))
        seen = []

        def other_session():
            try:
                workspace.current()
                seen.append("leaked")
            except workspace.NoWorkspace:
                seen.append("nobody")
        t = threading.Thread(target=other_session)
        t.start()
        t.join()
        assert seen == ["nobody"] and workspace.current().user_key == workspace.user_key("g", "111")
    print("PASS: a user set in one session's thread is invisible in another (context variable, not a global)")


def test_consent_and_delete() -> None:
    with tempfile.TemporaryDirectory() as d, hosted(Path(d)):
        ws = workspace.for_user("g", "333")
        assert not ws.has_consented()
        ws.record_consent()
        assert ws.has_consented()
        real = workspace.NOTICE_VERSION
        workspace.NOTICE_VERSION = "2099-01-01"                             # notice changed: asked again
        try:
            assert not ws.has_consented()
        finally:
            workspace.NOTICE_VERSION = real
        workspace.use(ws)
        portfolio_store.add_holding({"ticker": "INFY.NS", "quantity": 1, "avg_cost": 1500})
        ws.delete_everything()
        assert not ws.root.exists() and portfolio_store.load_holdings() == []
        try:
            workspace.user_key("g", "")
        except workspace.NoWorkspace:
            pass
        else:
            raise AssertionError("an empty sign-in ID must not map to a folder")
    print("PASS: consent is recorded per notice version; Delete my data removes the whole folder")


def test_no_brokers_on_the_website() -> None:
    with tempfile.TemporaryDirectory() as d, hosted(Path(d)):
        env = Path(d) / ".env"
        env.write_text("UPSTOX_ACCESS_TOKEN=operator-token\n")
        assert get_secret("UPSTOX_ACCESS_TOKEN", env_path=env) is None    # the server's tokens are never used
        workspace.use(workspace.for_user("g", "444"))
        res = portfolio_sync.sync("upstox")
        assert not res.ok and res.error_type == "UnsupportedOperation" and "CAS statement" in res.message
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "portfolio_sync.py"), "--status"],
                             capture_output=True, text=True, cwd=ROOT, env={**os.environ}, timeout=60)
        assert out.returncode == 2 and "pass --db" in out.stderr         # never the shared DB
    print("PASS: on the website broker syncs are refused, .env tokens are ignored, and the CLI demands the user's DB")


if __name__ == "__main__":
    test_local_mode_is_unchanged()
    test_nobody_signed_in_reads_nothing()
    test_two_users_never_see_each_other()
    test_threads_start_as_nobody()
    test_consent_and_delete()
    test_no_brokers_on_the_website()
    test_local_mode_is_unchanged()      # and back to local afterwards
    print("\nAll workspace tests passed.")
