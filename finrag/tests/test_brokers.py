"""
Adapter contract tests, run against every registered adapter. Offline: broker
adapters get a fake HTTP session replaying fixture JSON (tests/fixtures/brokers,
built from the official docs' example responses); CSV uses a fixture export.

    python tests/test_brokers.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from brokers import (AuthExpired, AuthStatus, BrokerUnavailable, CredentialsMissing, Holding,  # noqa: E402
                     MalformedResponse, RateLimited, get_adapter, list_adapters)
from brokers.csv_adapter import parse_rows  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "brokers"
ROOT = Path(__file__).parent.parent
TOKEN = "test-token-not-real"


def fixture(name):
    return json.loads((FIX / name).read_text())


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.content = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())

    def json(self):
        if isinstance(self._body, bytes):
            raise ValueError("not JSON")
        return self._body


class FakeSession:
    """Replays (status, body) pairs; an Exception instance is raised instead of answering."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(*reply)


SPECS = {  # per broker adapter: fixtures, constructor kwargs, and the expected parse
    "upstox": {"holdings": fixture("upstox_holdings.json"), "prices": fixture("upstox_ltp.json"),
               "auth_error": (401, fixture("upstox_error_401.json")), "rate": (429, fixture("upstox_error_429.json")),
               "kwargs": {"token": TOKEN}, "host": "https://api.upstox.com",
               "expect": [("YESBANK", "NSE", 36.0, 18.75, "YESBANK.NS"), ("RELIANCE", "NSE", 10.0, 1250.0, "RELIANCE.NS")],
               "prices_expect": {"YESBANK": 17.15, "RELIANCE": 1190.4}},
    "dhan": {"holdings": fixture("dhan_holdings.json"), "prices": fixture("dhan_ltp.json"),
             "auth_error": (400, fixture("dhan_error_auth.json")), "rate": (429, fixture("dhan_error_429.json")),
             "kwargs": {"token": TOKEN, "client_id": "1000000001"}, "host": "https://api.dhan.co",
             "expect": [("HDFC", "NSE", 1000.0, 2655.0, "HDFC.NS"), ("TCS", "NSE", 5.0, 3600.0, "TCS.NS")],
             "prices_expect": {"TCS": 3245.6}},
    "groww": {"holdings": fixture("groww_holdings.json"), "prices": fixture("groww_ltp.json"),
              "auth_error": (403, fixture("groww_error_auth.json")), "rate": (429, {"status": "FAILURE"}),
              "kwargs": {"token": TOKEN}, "host": "https://api.groww.in",
              "expect": [("RELIANCE", "NSE", 10.0, 100.0, "RELIANCE.NS"), ("INFY", "NSE", 4.0, 1500.5, "INFY.NS")],
              "prices_expect": {"RELIANCE": 2334.2, "INFY": 1612.75}},
}


def make(name, *replies, sleeps=None):
    session = FakeSession(*replies)
    adapter = get_adapter(name, session=session, sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
                          **SPECS[name]["kwargs"])
    return adapter, session


def test_registry() -> None:
    assert list_adapters() == ["csv", "cas", "upstox", "dhan", "groww"]
    for name in list_adapters():
        a = get_adapter(name) if name in ("csv", "cas") else get_adapter(name, env_path=Path("/nonexistent/.env"))
        assert a.name == name and TOKEN not in repr(a)
    try:
        get_adapter("zerodha")
    except ValueError as e:
        assert "unknown portfolio source" in str(e)
    else:
        raise AssertionError
    print("PASS: registry lists csv, cas, upstox, dhan, groww; unknown names are refused")


def test_holdings_and_prices_parse() -> None:
    for name, spec in SPECS.items():
        adapter, session = make(name, (200, spec["holdings"]), (200, spec["prices"]))
        got = adapter.fetch_holdings()
        assert all(isinstance(h, Holding) and h.source == name and h.currency == "INR" for h in got), name
        assert [(h.symbol, h.exchange, h.quantity, h.avg_cost, h.ticker) for h in got] == spec["expect"], (name, got)
        method, url, kw = session.calls[0]
        assert method == "GET" and url.startswith(spec["host"]), (name, url)
        prices = adapter.fetch_prices([h.symbol for h in got])
        assert {s: p.last_price for s, p in prices.items()} == spec["prices_expect"], (name, prices)
        assert all(p.as_of for p in prices.values())
        assert all(session.calls[i][1].startswith(spec["host"]) for i in range(len(session.calls)))
    # documented request details
    up, s = make("upstox", (200, SPECS["upstox"]["holdings"]), (200, SPECS["upstox"]["prices"]))
    up.fetch_prices([h.symbol for h in up.fetch_holdings()])
    assert s.calls[0][1] == "https://api.upstox.com/v2/portfolio/long-term-holdings"
    assert s.calls[0][2]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert s.calls[1][1] == "https://api.upstox.com/v3/market-quote/ltp"
    assert s.calls[1][2]["params"]["instrument_key"] == "NSE_EQ|INE528G01035,NSE_EQ|INE002A01018"
    dh, s = make("dhan", (200, SPECS["dhan"]["holdings"]), (200, SPECS["dhan"]["prices"]))
    dh.fetch_prices([h.symbol for h in dh.fetch_holdings()])
    assert s.calls[0][2]["headers"]["access-token"] == TOKEN and s.calls[1][0] == "POST"
    assert s.calls[1][2]["json"] == {"NSE_EQ": [1330, 11536]} and s.calls[1][2]["headers"]["client-id"] == "1000000001"
    gw, s = make("groww", (200, SPECS["groww"]["holdings"]), (200, SPECS["groww"]["prices"]))
    gw.fetch_prices([h.symbol for h in gw.fetch_holdings()])
    assert s.calls[0][2]["headers"]["X-API-VERSION"] == "1.0"
    assert s.calls[1][2]["params"] == {"segment": "CASH", "exchange_symbols": "NSE_RELIANCE,NSE_INFY"}
    print("PASS: every broker adapter parses its documented holdings and LTP payloads, on its own host only")


def test_expired_token_is_auth_expired() -> None:
    for name, spec in SPECS.items():
        adapter, _ = make(name, spec["auth_error"])
        try:
            adapter.fetch_holdings()
        except AuthExpired as e:
            assert "re" in str(e).lower() and ".env" in str(e) and TOKEN not in str(e), (name, str(e))
        else:
            raise AssertionError(f"{name}: expected AuthExpired")
        adapter, _ = make(name, spec["auth_error"])
        assert adapter.auth_status() == AuthStatus.EXPIRED, name
        ok, _ = make(name, (200, spec["holdings"]))
        assert ok.auth_status() == AuthStatus.OK, name
    print("PASS: a rejected token raises AuthExpired with a re-authenticate message (and auth_status says expired)")


def test_rate_limit_backoff_then_rate_limited() -> None:
    for name, spec in SPECS.items():
        sleeps = []
        adapter, session = make(name, spec["rate"], spec["rate"], (200, spec["holdings"]), sleeps=sleeps)
        assert len(adapter.fetch_holdings()) == 2 and sleeps == [1.0, 2.0], (name, sleeps)   # backoff, then success
        sleeps = []
        adapter, _ = make(name, *[spec["rate"]] * 4, sleeps=sleeps)
        try:
            adapter.fetch_holdings()
        except RateLimited as e:
            assert "gave up after 4 attempts" in str(e) and sleeps == [1.0, 2.0, 4.0], (name, str(e), sleeps)
        else:
            raise AssertionError(f"{name}: expected RateLimited")
    print("PASS: rate limits back off (1 s, 2 s, 4 s) and then raise RateLimited")


def test_malformed_and_network_errors() -> None:
    for name, spec in SPECS.items():
        for bad in ({"unexpected": True}, [{"no_symbol_here": 1}], b"<html>gateway</html>"):
            adapter, _ = make(name, (200, bad))
            try:
                adapter.fetch_holdings()
            except MalformedResponse as e:
                assert name in str(e), (name, str(e))
            else:
                raise AssertionError(f"{name}: {bad!r} should be MalformedResponse")
        adapter, _ = make(name, *[ConnectionError("dns")] * 4)
        try:
            adapter.fetch_holdings()
        except BrokerUnavailable as e:
            assert "network error" in str(e)
        else:
            raise AssertionError(f"{name}: expected BrokerUnavailable")
        adapter, _ = make(name, (502, None), (503, None), (500, None), (500, None))
        try:
            adapter.fetch_holdings()
        except BrokerUnavailable:
            pass
        else:
            raise AssertionError(f"{name}: 5xx should end as BrokerUnavailable")
    print("PASS: malformed payloads -> MalformedResponse naming the broker; network/5xx -> BrokerUnavailable after retries")


def test_missing_credentials() -> None:
    with tempfile.TemporaryDirectory() as d:
        env = Path(d) / ".env"
        env.write_text("SOMETHING_ELSE=1\n")
        for name in SPECS:
            adapter = get_adapter(name, env_path=env)
            assert adapter.auth_status() == AuthStatus.MISSING, name
            try:
                adapter.fetch_holdings()
            except CredentialsMissing as e:
                assert ".env" in str(e)
            else:
                raise AssertionError(name)
        env.write_text(f"UPSTOX_ACCESS_TOKEN={TOKEN}\n")
        assert get_adapter("upstox", env_path=env)._token == TOKEN   # read from .env only
    print("PASS: missing credentials -> CredentialsMissing / auth_status missing; tokens are read from .env")


def test_csv_import_validates_and_reports() -> None:
    mapping = json.loads((ROOT / "config" / "broker_mappings" / "generic_example.json").read_text())
    rep = parse_rows((FIX / "sample_export.csv").read_text(), mapping)
    got = {(h.symbol, h.exchange): (h.quantity, round(h.avg_cost, 2), h.ticker) for h in rep.holdings}
    assert got == {("RELIANCE", "NSE"): (20.0, 1300.0, "RELIANCE.NS"), ("TCS", "NSE"): (5.0, 3600.0, "TCS.NS"),
                   ("AAPL", "US"): (4.0, 182.5, "AAPL")}, got
    reasons = dict(rep.skipped)
    assert "negative quantity -3" in reasons[4] and "'n/a' is not a number" in reasons[5]
    assert "isn't a valid trading symbol" in reasons[6] and reasons[9] == "blank row"
    assert "exchange 'LSE'" in reasons[10]
    assert rep.merged and rep.merged[0][0] == 8                      # second RELIANCE lot merged, not dropped
    assert len(rep.holdings) + len(rep.skipped) + len(rep.merged) == 9   # every data line accounted for
    bad = dict(mapping, columns={**mapping["columns"], "quantity": "Qty."})
    try:
        parse_rows((FIX / "sample_export.csv").read_text(), bad)
    except MalformedResponse as e:
        assert "no column for quantity ('Qty.')" in str(e) and "Symbol" in str(e)
    else:
        raise AssertionError
    csv_adapter = get_adapter("csv", file=FIX / "sample_export.csv",
                              mapping=ROOT / "config" / "broker_mappings" / "generic_example.json")
    assert csv_adapter.auth_status() == AuthStatus.OK and len(csv_adapter.fetch_holdings()) == 3
    assert csv_adapter.fetch_prices(["TCS"]) == {} and csv_adapter.allowed_hosts == ()
    print("PASS: CSV import maps columns, rejects negative/non-numeric/invalid rows with reasons, merges lots, never drops")


if __name__ == "__main__":
    test_registry()
    test_holdings_and_prices_parse()
    test_expired_token_is_auth_expired()
    test_rate_limit_backoff_then_rate_limited()
    test_malformed_and_network_errors()
    test_missing_credentials()
    test_csv_import_validates_and_reports()
    print("\nAll broker tests passed.")
