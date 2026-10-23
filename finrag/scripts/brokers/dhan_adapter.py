"""
Dhan — read-only, bring-your-own-credentials.

Verified from the official docs (dhanhq.co/docs/v2, Oct 2026):
  - Holdings: GET https://api.dhan.co/v2/holdings, headers Content-Type +
    access-token. Response is a top-level JSON array: [{exchange ("ALL" in the
    example), tradingSymbol, securityId, isin, totalQty, dpQty, t1Qty,
    availableQty, collateralQty, avgCostPrice}].
  - LTP: POST https://api.dhan.co/v2/marketfeed/ltp, headers access-token +
    client-id, body {"NSE_EQ": [securityId, …]} (up to 1,000 instruments,
    1 request/second). Response {"data": {"NSE_EQ": {"11536": {"last_price": …}}},
    "status": "success"}.
  - Auth: access tokens valid 24 hours (renewable while active); generated on
    web.dhan.co, via the TOTP endpoint, or the API-key consent flow. Partners have
    a documented consent flow (generate-consent -> consent-login -> consume-consent).
  - Errors: {"errorType", "errorCode", "errorMessage"}; DH-901 = client ID or
    access token invalid/expired; DH-904 = rate limit. Non-trading APIs: 20/s.

TODO (not stated in the docs): the HTTP status codes that accompany DH-901 /
DH-904 (we classify by errorCode, and also treat 401/403 as auth and 429 as rate
limit); what exchange "ALL" means for pricing (we price on NSE_EQ by securityId);
whether the market-feed API needs a separate data subscription — if it fails,
prices are marked unavailable and holdings are kept.
"""
from __future__ import annotations

from datetime import datetime

from .base import (AuthExpired, AuthStatus, BrokerAdapter, BrokerUnavailable, CredentialsMissing, Holding,
                   MalformedResponse, Price, RateLimited, UnsupportedOperation)
from .credentials import get_secret
from .http import BrokerHTTP

HOST = "api.dhan.co"
HOLDINGS_PATH = "/v2/holdings"
LTP_PATH = "/v2/marketfeed/ltp"
LTP_BATCH = 1000
REAUTH = ("Dhan rejected the access token (DH-901 — tokens are valid 24 hours). Generate a new one on "
          "web.dhan.co ('Access DhanHQ APIs') and update DHAN_ACCESS_TOKEN in finrag/.env.")


def classify(status: int, body):
    code = str(body.get("errorCode", "")) if isinstance(body, dict) else ""
    if code == "DH-901" or status in (401, 403):
        return AuthExpired(REAUTH)
    if code == "DH-904" or status == 429:
        return RateLimited("dhan: too many requests (DH-904)")
    if status >= 500:
        return BrokerUnavailable(f"dhan: server error HTTP {status}")
    if status >= 400 or code:
        msg = body.get("errorMessage", "") if isinstance(body, dict) else ""
        return MalformedResponse(f"dhan: request failed (HTTP {status}{', ' + code if code else ''}"
                                 f"{': ' + str(msg) if msg else ''})")
    return None


class DhanAdapter(BrokerAdapter):
    name = "dhan"
    allowed_hosts = (HOST,)

    def __init__(self, token: str | None = None, client_id: str | None = None, session=None, sleep=None,
                 env_path=None):
        self._token = token if token is not None else get_secret("DHAN_ACCESS_TOKEN", env_path)
        self._client_id = client_id if client_id is not None else get_secret("DHAN_CLIENT_ID", env_path)
        self._session, self._sleep = session, sleep
        self._ids: dict[str, str] = {}  # symbol -> securityId

    def _http(self, with_client_id: bool = False) -> BrokerHTTP:
        if not self._token:
            raise CredentialsMissing("DHAN_ACCESS_TOKEN is not set in finrag/.env.", self.name)
        headers = {"Content-Type": "application/json", "Accept": "application/json", "access-token": self._token}
        if with_client_id:
            headers["client-id"] = self._client_id
        kwargs = {"sleep": self._sleep} if self._sleep else {}
        return BrokerHTTP(f"https://{HOST}", headers, self.name, classify, session=self._session,
                          secrets=(self._token, self._client_id or ""), **kwargs)

    def fetch_holdings(self) -> list[Holding]:
        rows = self._http().request("GET", HOLDINGS_PATH)
        if not isinstance(rows, list):
            raise MalformedResponse("dhan: holdings response is not a list", self.name)
        holdings = []
        for i, r in enumerate(rows):
            try:
                exchange = str(r.get("exchange", "")).upper()
                h = Holding(symbol=str(r["tradingSymbol"]).strip().upper(),
                            exchange="BSE" if exchange == "BSE" else "NSE",  # "ALL" = both; priced on NSE
                            quantity=float(r["totalQty"]), avg_cost=float(r["avgCostPrice"]), currency="INR",
                            source=self.name, isin=r.get("isin"), broker_ref=str(r.get("securityId") or "") or None)
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                raise MalformedResponse(f"dhan: holding #{i + 1} is missing or has a bad field ({e})", self.name)
            if h.broker_ref:
                self._ids[h.symbol] = h.broker_ref
            holdings.append(h)
        return holdings

    def fetch_prices(self, symbols: list[str]) -> dict[str, Price]:
        if not self._client_id:
            raise UnsupportedOperation("dhan: prices need DHAN_CLIENT_ID in finrag/.env", self.name)
        wanted = {self._ids[s]: s for s in symbols if s in self._ids}
        ids = list(wanted)
        prices: dict[str, Price] = {}
        now = datetime.now().isoformat(timespec="seconds")
        for i in range(0, len(ids), LTP_BATCH):
            body = self._http(with_client_id=True).request(
                "POST", LTP_PATH, json_body={"NSE_EQ": [int(x) for x in ids[i:i + LTP_BATCH]]})
            seg = (body.get("data") or {}).get("NSE_EQ") if isinstance(body, dict) else None
            if not isinstance(seg, dict):
                raise MalformedResponse("dhan: LTP response has no data.NSE_EQ object", self.name)
            for sec_id, quote in seg.items():
                price = quote.get("last_price") if isinstance(quote, dict) else None
                if str(sec_id) in wanted and isinstance(price, (int, float)) and price > 0:
                    prices[wanted[str(sec_id)]] = Price(wanted[str(sec_id)], float(price), now)
        return prices

    def auth_status(self) -> AuthStatus:
        if not self._token:
            return AuthStatus.MISSING
        try:
            self._http().request("GET", HOLDINGS_PATH)
        except AuthExpired:
            return AuthStatus.EXPIRED
        return AuthStatus.OK
