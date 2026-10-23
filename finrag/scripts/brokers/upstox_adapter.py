"""
Upstox — read-only, bring-your-own-credentials.

Verified from the official docs (upstox.com/developer/api-documentation, Oct 2026):
  - Holdings: GET https://api.upstox.com/v2/portfolio/long-term-holdings,
    headers Accept/Content-Type application/json + Authorization: Bearer <token>.
    {"status": "success", "data": [{isin, trading_symbol, exchange, quantity,
    average_price, last_price, close_price, instrument_token ("NSE_EQ|INE…"),
    t1_quantity, …}]}
  - LTP: GET https://api.upstox.com/v3/market-quote/ltp?instrument_key=k1,k2
    (the v2 endpoint is marked deprecated). {"status": "success", "data":
    {"NSE_EQ:RELIANCE": {"last_price", "instrument_token", "ltq", "volume", "cp"}}}
  - Auth: OAuth authorization-code flow; the access token is valid until 3:30 AM
    the next day. The 1-year "Analytics Token" is read-only and can call Portfolio
    GET APIs — but only from a whitelisted static IP.
  - Errors: {"status": "error", "errors": [{"errorCode", "message", "propertyPath",
    "invalidValue"}]} (SDK models Problem / ApiGatewayErrorResponse).
    401 UDAPI100050 = invalid token; 429 UDAPI10005 = too many requests.
  - Rate limit for standard APIs: 50/s, 500/min, 2,000 per 30 min.
  - Sandbox: order APIs only — no holdings, so tests use the docs' example payloads.

TODO (not stated in the docs): whether `quantity` already includes `t1_quantity`
(we use `quantity` as "the total holding qty", as documented); the v3 LTP limit on
instruments per call (we send at most 50); whether tokens are JWTs with an expiry
we could check offline (we don't rely on it).
"""
from __future__ import annotations

from datetime import datetime

from .base import (AuthExpired, AuthStatus, BrokerAdapter, BrokerUnavailable, CredentialsMissing, Holding,
                   MalformedResponse, Price, RateLimited)
from .credentials import get_secret
from .http import BrokerHTTP

HOST = "api.upstox.com"
HOLDINGS_PATH = "/v2/portfolio/long-term-holdings"
LTP_PATH = "/v3/market-quote/ltp"
LTP_BATCH = 50  # TODO: the documented v3 maximum isn't stated; 50 is conservative
REAUTH = ("Upstox rejected the access token (tokens expire at 3:30 AM each day). Generate a new one "
          "(developer apps page, or the OAuth flow) and update UPSTOX_ACCESS_TOKEN in finrag/.env.")


def _codes(body) -> list[str]:
    if isinstance(body, dict) and isinstance(body.get("errors"), list):
        return [str(e.get("errorCode") or e.get("error_code") or "") for e in body["errors"] if isinstance(e, dict)]
    return []


def classify(status: int, body):
    codes = _codes(body)
    if status == 401 or "UDAPI100050" in codes:
        return AuthExpired(REAUTH)
    if status == 429 or "UDAPI10005" in codes:
        return RateLimited("upstox: too many requests (UDAPI10005)")
    if status >= 500:
        return BrokerUnavailable(f"upstox: server error HTTP {status}")
    if status >= 400 or (isinstance(body, dict) and body.get("status") == "error"):
        msg = "; ".join(str(e.get("message", "")) for e in body.get("errors", []) if isinstance(e, dict)) \
            if isinstance(body, dict) else ""
        return MalformedResponse(f"upstox: request failed (HTTP {status}{', ' + ', '.join(codes) if codes else ''}"
                                 f"{': ' + msg if msg else ''})")
    return None


class UpstoxAdapter(BrokerAdapter):
    name = "upstox"
    allowed_hosts = (HOST,)

    def __init__(self, token: str | None = None, session=None, sleep=None, env_path=None):
        self._token = token if token is not None else get_secret("UPSTOX_ACCESS_TOKEN", env_path)
        self._session, self._sleep = session, sleep
        self._keys: dict[str, str] = {}  # symbol -> instrument key (from holdings), for LTP

    def _http(self) -> BrokerHTTP:
        if not self._token:
            raise CredentialsMissing("UPSTOX_ACCESS_TOKEN is not set in finrag/.env.", self.name)
        kwargs = {"sleep": self._sleep} if self._sleep else {}
        return BrokerHTTP(f"https://{HOST}", {"Accept": "application/json", "Content-Type": "application/json",
                                              "Authorization": f"Bearer {self._token}"},
                          self.name, classify, session=self._session, secrets=(self._token,), **kwargs)

    def fetch_holdings(self) -> list[Holding]:
        body = self._http().request("GET", HOLDINGS_PATH)
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise MalformedResponse("upstox: holdings response has no 'data' list", self.name)
        holdings = []
        for i, r in enumerate(rows):
            try:
                h = Holding(symbol=str(r["trading_symbol"]).strip().upper(), exchange=str(r["exchange"]).upper(),
                            quantity=float(r["quantity"]), avg_cost=float(r["average_price"]), currency="INR",
                            source=self.name, isin=r.get("isin"), broker_ref=r.get("instrument_token"))
            except (KeyError, TypeError, ValueError) as e:
                raise MalformedResponse(f"upstox: holding #{i + 1} is missing or has a bad field ({e})", self.name)
            if h.broker_ref:
                self._keys[h.symbol] = h.broker_ref
            holdings.append(h)
        return holdings

    def fetch_prices(self, symbols: list[str]) -> dict[str, Price]:
        """LTP for symbols seen in the last fetch_holdings (their instrument keys come from it)."""
        wanted = {self._keys[s]: s for s in symbols if s in self._keys}
        prices: dict[str, Price] = {}
        now = datetime.now().isoformat(timespec="seconds")  # the LTP response carries no timestamp
        keys = list(wanted)
        for i in range(0, len(keys), LTP_BATCH):
            body = self._http().request("GET", LTP_PATH, params={"instrument_key": ",".join(keys[i:i + LTP_BATCH])})
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, dict):
                raise MalformedResponse("upstox: LTP response has no 'data' object", self.name)
            for quote in data.values():
                sym = wanted.get(quote.get("instrument_token")) if isinstance(quote, dict) else None
                price = quote.get("last_price") if isinstance(quote, dict) else None
                if sym and isinstance(price, (int, float)) and price > 0:
                    prices[sym] = Price(sym, float(price), now)
        return prices

    def auth_status(self) -> AuthStatus:
        """Online check with the (read-only) holdings endpoint."""
        if not self._token:
            return AuthStatus.MISSING
        try:
            self._http().request("GET", HOLDINGS_PATH)
        except AuthExpired:
            return AuthStatus.EXPIRED
        return AuthStatus.OK
