"""
Groww — read-only, bring-your-own-credentials.

Verified from the official docs (groww.in/trade-api/docs, Oct 2026):
  - Holdings: GET https://api.groww.in/v1/holdings/user, headers Accept:
    application/json, Authorization: Bearer <token>, X-API-VERSION: 1.0.
    {"status": "SUCCESS", "payload": {"holdings": [{isin, trading_symbol,
    quantity, average_price, pledge_quantity, t1_quantity, demat_free_quantity, …}]}}
  - LTP: GET https://api.groww.in/v1/live-data/ltp?segment=CASH&exchange_symbols=
    NSE_RELIANCE,… (up to 50 per call). {"status": "SUCCESS", "payload":
    {"NSE_RELIANCE": 2334.2}}.
  - Auth: a token generated on groww.in (Trading APIs) "expires daily at 6:00 AM";
    or API key + secret / TOTP via POST /v1/token/api/access (150/day).
  - Errors: HTTP 40x/50x with {"status": "FAILURE", "error": {"code", "message",
    "metadata"}}; GA005 = "User not authorised to perform this operation".
    Non-trading APIs 20/s, 500/min; live data 10/s, 300/min.

TODO (not stated in the docs): the HTTP status for an expired token and for rate
limiting (we treat 401/403 and GA005 as re-authenticate, 429 as rate limit); the
holdings payload has no exchange field (demat holdings are exchange-agnostic), so
holdings are priced on NSE.
"""
from __future__ import annotations

from datetime import datetime

from .base import (AuthExpired, AuthStatus, BrokerAdapter, BrokerUnavailable, CredentialsMissing, Holding,
                   MalformedResponse, Price, RateLimited)
from .credentials import get_secret
from .http import BrokerHTTP

HOST = "api.groww.in"
HOLDINGS_PATH = "/v1/holdings/user"
LTP_PATH = "/v1/live-data/ltp"
LTP_BATCH = 50
REAUTH = ("Groww rejected the access token (tokens expire daily at 6:00 AM). Generate a new one on groww.in "
          "(Profile → Trading APIs) and update GROWW_ACCESS_TOKEN in finrag/.env.")


def classify(status: int, body):
    err = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else {}
    code = str(err.get("code", ""))
    if status in (401, 403) or code == "GA005":
        return AuthExpired(REAUTH)
    if status == 429:
        return RateLimited("groww: too many requests")
    if status >= 500:
        return BrokerUnavailable(f"groww: server error HTTP {status}")
    if status >= 400 or (isinstance(body, dict) and body.get("status") == "FAILURE"):
        return MalformedResponse(f"groww: request failed (HTTP {status}{', ' + code if code else ''}"
                                 f"{': ' + str(err.get('message')) if err.get('message') else ''})")
    return None


class GrowwAdapter(BrokerAdapter):
    name = "groww"
    allowed_hosts = (HOST,)

    def __init__(self, token: str | None = None, session=None, sleep=None, env_path=None):
        self._token = token if token is not None else get_secret("GROWW_ACCESS_TOKEN", env_path)
        self._session, self._sleep = session, sleep

    def _http(self) -> BrokerHTTP:
        if not self._token:
            raise CredentialsMissing("GROWW_ACCESS_TOKEN is not set in finrag/.env.", self.name)
        kwargs = {"sleep": self._sleep} if self._sleep else {}
        return BrokerHTTP(f"https://{HOST}", {"Accept": "application/json", "X-API-VERSION": "1.0",
                                              "Authorization": f"Bearer {self._token}"},
                          self.name, classify, session=self._session, secrets=(self._token,), **kwargs)

    def fetch_holdings(self) -> list[Holding]:
        body = self._http().request("GET", HOLDINGS_PATH)
        payload = body.get("payload") if isinstance(body, dict) else None
        rows = payload.get("holdings") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise MalformedResponse("groww: holdings response has no payload.holdings list", self.name)
        holdings = []
        for i, r in enumerate(rows):
            try:
                holdings.append(Holding(symbol=str(r["trading_symbol"]).strip().upper(), exchange="NSE",
                                        quantity=float(r["quantity"]), avg_cost=float(r["average_price"]),
                                        currency="INR", source=self.name, isin=r.get("isin")))
            except (KeyError, TypeError, ValueError) as e:
                raise MalformedResponse(f"groww: holding #{i + 1} is missing or has a bad field ({e})", self.name)
        return holdings

    def fetch_prices(self, symbols: list[str]) -> dict[str, Price]:
        prices: dict[str, Price] = {}
        now = datetime.now().isoformat(timespec="seconds")
        for i in range(0, len(symbols), LTP_BATCH):
            batch = symbols[i:i + LTP_BATCH]
            body = self._http().request("GET", LTP_PATH, params={
                "segment": "CASH", "exchange_symbols": ",".join(f"NSE_{s}" for s in batch)})
            payload = body.get("payload") if isinstance(body, dict) else None
            if not isinstance(payload, dict):
                raise MalformedResponse("groww: LTP response has no payload object", self.name)
            for s in batch:
                price = payload.get(f"NSE_{s}")
                if isinstance(price, (int, float)) and price > 0:
                    prices[s] = Price(s, float(price), now)
        return prices

    def auth_status(self) -> AuthStatus:
        if not self._token:
            return AuthStatus.MISSING
        try:
            self._http().request("GET", HOLDINGS_PATH)
        except AuthExpired:
            return AuthStatus.EXPIRED
        return AuthStatus.OK
