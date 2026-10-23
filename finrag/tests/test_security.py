"""
Security checks for portfolio ingestion. Offline.

  1. No token/secret-like string ever reaches logs, printed output, exception
     messages, sync_log rows or the portfolio DB — on every error path.
  2. During a sync the only reachable host is the adapter's own broker host.
  3. Query-time answering (route -> retrieve -> prompt -> citation check) makes
     no outbound connection at all (loopback, for the local model, is allowed).

    python tests/test_security.py
"""
import contextlib
import io
import logging
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import portfolio_sync  # noqa: E402
from brokers import BrokerError, get_adapter  # noqa: E402
from brokers.base import AuthStatus, BrokerAdapter  # noqa: E402
from net_guard import NetworkBlocked, outbound_only  # noqa: E402
from test_brokers import SPECS, FakeSession  # noqa: E402

SECRET = "sk-LIVE-9f3a7c2e81d4b6a0"
CLIENT_ID = "CID-55512345"
ROOT = Path(__file__).parent.parent


def error_paths(name):
    spec = SPECS[name]
    echo = {"status": "error", "errors": [{"errorCode": "X", "message": f"bad token {SECRET}"}]} if name == "upstox" else \
        {"errorCode": "DH-905", "errorMessage": f"token {SECRET} rejected"} if name == "dhan" else \
        {"status": "FAILURE", "error": {"code": "GA001", "message": f"token {SECRET} rejected"}}
    return [
        [spec["auth_error"]],                                  # expired token
        [spec["rate"]] * 4,                                    # rate limited to the end
        [ConnectionError(f"connect failed for Bearer {SECRET}")] * 4,   # a network error that mentions the header
        [(200, {"garbage": True})],                            # malformed payload
        [(400, echo)],                                         # the broker echoes the token in its message
        [(200, spec["holdings"]), ConnectionError("prices down")] + [ConnectionError("x")] * 3,  # price failure
    ]


def test_no_secrets_anywhere() -> None:
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    out, messages = io.StringIO(), []
    try:
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "portfolio.db"
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                for name in SPECS:
                    for replies in error_paths(name):
                        kwargs = {"token": SECRET} | ({"client_id": CLIENT_ID} if name == "dhan" else {})
                        adapter = get_adapter(name, session=FakeSession(*replies), sleep=lambda s: None, **kwargs)
                        res = portfolio_sync.sync(name, db_path=db, adapter=adapter)
                        messages += [res.message, res.prices_status, res.error_type or "", repr(adapter), str(res)]
                        try:
                            get_adapter(name, session=FakeSession(*replies), sleep=lambda s: None,
                                        **kwargs).fetch_holdings()
                        except BrokerError as e:
                            messages += [str(e), repr(e)]
                        except Exception as e:  # any other exception must not leak either
                            messages += [str(e), repr(e)]
                        print(res)
            blob = db.read_bytes()
            import sqlite3
            conn = sqlite3.connect(db)
            log_rows = " ".join(str(r) for r in conn.execute("SELECT * FROM sync_log"))
            n_failed = conn.execute("SELECT COUNT(*) FROM sync_log WHERE status='failed'").fetchone()[0]
            conn.close()
    finally:
        logging.getLogger().removeHandler(handler)
    everything = "\n".join(messages) + out.getvalue() + log.getvalue() + log_rows
    for secret in (SECRET, CLIENT_ID):
        assert secret not in everything, f"{secret} leaked: …{everything[everything.find(secret) - 80:][:160]}"
        assert secret.encode() not in blob, f"{secret} is stored in the DB"
    assert n_failed >= 12 and "***" in everything        # failures were logged, and echoed tokens were redacted
    print(f"PASS: no token in logs, output, exceptions, {n_failed} failed sync_log rows or the DB (echoed tokens redacted)")


def test_guard_blocks_everything_but_the_allowed_host() -> None:
    real_resolve, real_connect = socket.getaddrinfo, socket.socket.connect
    fake_ip = "192.0.2.10"   # TEST-NET: never routed; connect() below is faked anyway

    def fake_resolve(host, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (fake_ip, 443))]

    def fake_connect(self, address):
        raise ConnectionRefusedError("test: no real connection")
    socket.getaddrinfo, socket.socket.connect = fake_resolve, fake_connect
    try:
        import requests
        with outbound_only(["api.upstox.com"]) as guard:
            try:
                requests.get("https://api.upstox.com/v2/portfolio/long-term-holdings", timeout=2)
            except requests.ConnectionError:
                pass                      # allowed: it got as far as connect(), which the test refuses
            try:
                requests.get("https://example.com/", timeout=2)
            except NetworkBlocked:
                pass
            else:
                raise AssertionError("example.com should be blocked")
            s = socket.socket()
            try:
                s.connect(("203.0.113.5", 443))   # a raw IP nobody resolved
            except NetworkBlocked:
                pass
            else:
                raise AssertionError("raw IP connects should be blocked")
            finally:
                s.close()
        assert guard.contacted == ["api.upstox.com"] and "example.com" in guard.blocked and "203.0.113.5" in guard.blocked
    finally:
        socket.getaddrinfo, socket.socket.connect = real_resolve, real_connect
    assert socket.getaddrinfo is real_resolve                  # restored afterwards
    print("PASS: the guard allows only the listed host (resolve + connect); others and raw IPs are blocked and recorded")


def test_sync_enforces_adapter_hosts() -> None:
    class Sneaky(BrokerAdapter):
        name, allowed_hosts = "upstox", ("api.upstox.com",)

        def fetch_holdings(self):
            socket.getaddrinfo("telemetry.example.net", 443)   # not its broker host
            return []

        def auth_status(self):
            return AuthStatus.OK
    with tempfile.TemporaryDirectory() as d:
        res = portfolio_sync.sync("upstox", db_path=Path(d) / "p.db", adapter=Sneaky())
    assert not res.ok and res.error_type == "NetworkBlocked" and "telemetry.example.net" in res.message
    csv_hosts = get_adapter("csv").allowed_hosts
    assert csv_hosts == () and all(get_adapter(n, token="x").allowed_hosts == (f"api.{d}",)
                                   for n, d in (("upstox", "upstox.com"), ("dhan", "dhan.co"), ("groww", "groww.in")))
    print("PASS: a sync can reach only its adapter's broker host (CSV: none); anything else fails the sync")


QUERY_SCRIPT = r'''
import sys, io, contextlib
sys.path.insert(0, "scripts")
from net_guard import outbound_only
with outbound_only([]) as guard:                 # no external host at all; loopback allowed
    import rag_loop as r
    from citations import check_citations
    with contextlib.redirect_stdout(io.StringIO()):
        col = r.open_collection()
        if col is None:
            print("SKIP no-store"); raise SystemExit(0)
        for q in ["What is a SIP?", "What was TCS's consolidated revenue in FY 2026?", "How is my portfolio doing?"]:
            d, plan, pos = r.plan_query(q)
            cs = r.retrieve_for_plan(q, col, plan)
            prompt = r.build_prompt(q, cs, positions=pos, route=d.route)
            check_citations("Answer [1].", cs, r.get_embedder().encode_texts, extra_sources=[q])
print("BLOCKED", sorted(set(guard.blocked)))
print("CONTACTED", sorted(set(guard.contacted)))
'''


def test_query_time_makes_no_outbound_connection() -> None:
    """Fresh process, so loading the embedding model and the store happens inside the guard."""
    out = subprocess.run([sys.executable, "-c", QUERY_SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=600)
    text = out.stdout
    if "SKIP no-store" in text:
        print("SKIP: no vector store on this machine — query-time network test not run")
        return
    assert out.returncode == 0, out.stderr[-2000:]
    assert "BLOCKED []" in text and "CONTACTED []" in text, text + out.stderr[-1500:]
    from llm_backends import MLX_HOST, OLLAMA_HOST
    assert OLLAMA_HOST.startswith("http://127.0.0.1") and MLX_HOST.startswith("http://127.0.0.1")
    print("PASS: answering (routing, retrieval, prompt, citation check) made no outbound connection; models are on 127.0.0.1")


def test_gitignore_covers_private_files() -> None:
    private = [".env", ".env.local", "data/portfolio/portfolio.db", "data/portfolio/holdings.csv",
               "data/portfolio/zerodha_export.csv", "data/portfolio/imports/upstox.csv",
               "config/broker_mappings/my_broker.local.json", "config/broker_mappings/real.json"]
    public = ["config/broker_mappings/generic_example.json", ".env.example"]
    ignored = subprocess.run(["git", "check-ignore", "--no-index", *private, *public], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert set(private) <= set(ignored), set(private) - set(ignored)
    assert not set(public) & set(ignored), set(public) & set(ignored)
    print("PASS: .gitignore covers .env, the portfolio DB, holdings/export CSVs and real mapping files")


if __name__ == "__main__":
    test_no_secrets_anywhere()
    test_guard_blocks_everything_but_the_allowed_host()
    test_sync_enforces_adapter_hosts()
    test_gitignore_covers_private_files()
    test_query_time_makes_no_outbound_connection()
    print("\nAll security tests passed.")
