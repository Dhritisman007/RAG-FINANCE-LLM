"""
Broker-agnostic portfolio ingestion — the adapter contract.

Every source of holdings (a CSV / broker export, or a broker's read-only API used
with the user's OWN credentials) implements BrokerAdapter. Nothing here talks to
the network; adapters do, and only to the hosts they declare in `allowed_hosts`
(portfolio_sync enforces that during a sync).

Errors are typed so a caller can tell "re-authenticate" (AuthExpired /
CredentialsMissing) from "try later" (RateLimited, BrokerUnavailable) from "this
source can't do that" (UnsupportedOperation) from "the broker sent something we
don't understand" (MalformedResponse). Messages never contain credentials.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class AuthStatus(str, Enum):
    OK = "ok"
    EXPIRED = "expired"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"


# ── Errors ───────────────────────────────────────────────────────────────────
class BrokerError(Exception):
    """Base class; `source` names the adapter."""

    def __init__(self, message: str, source: str = ""):
        super().__init__(message)
        self.source = source


class AuthError(BrokerError):
    """The user has to (re-)authenticate."""


class AuthExpired(AuthError):
    """The token was rejected — usually a daily token that has expired."""


class CredentialsMissing(AuthError):
    """No credentials configured in .env."""


class IncorrectPassword(AuthError):
    """A file's password was wrong (a CAS statement opens with the first holder's PAN)."""


class RateLimited(BrokerError):
    """The broker kept rate-limiting after retries with backoff."""


class BrokerUnavailable(BrokerError):
    """Network failure, timeout or a broker-side (5xx) error."""


class UnsupportedOperation(BrokerError):
    """The source can't do this (e.g. no price API, or the API is unconfirmed)."""


class MalformedResponse(BrokerError):
    """The broker returned a payload that doesn't match its documented shape."""


# ── Data ─────────────────────────────────────────────────────────────────────
EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


@dataclass(frozen=True)
class Holding:
    symbol: str          # the broker's trading symbol, e.g. RELIANCE
    exchange: str        # NSE, BSE, US, …
    quantity: float
    avg_cost: float | None  # None when the source doesn't say (a CAS statement has no purchase price)
    currency: str        # INR, USD
    source: str          # adapter name: csv, upstox, dhan, groww
    isin: str | None = None
    broker_ref: str | None = None   # the broker's instrument id (Upstox instrument key, Dhan securityId)

    @property
    def ticker(self) -> str:
        """The ticker the rest of FinRAG uses (Yahoo form): RELIANCE on NSE -> RELIANCE.NS; a fund -> MF:<ISIN>."""
        if self.exchange.upper() == "MF":   # mutual fund: symbol is the ISIN, priced from AMFI's NAVs
            return "MF:" + self.symbol.upper()
        return self.symbol.upper() + EXCHANGE_SUFFIX.get(self.exchange.upper(), "")


@dataclass(frozen=True)
class Price:
    symbol: str
    last_price: float
    as_of: str           # ISO 8601; when the broker reported it (or when it was fetched, if the API gives no time)


# ── Contract ─────────────────────────────────────────────────────────────────
class BrokerAdapter(ABC):
    name: str = ""
    # Hosts this adapter may contact. portfolio_sync blocks every other outbound connection.
    allowed_hosts: tuple[str, ...] = ()

    @abstractmethod
    def fetch_holdings(self) -> list[Holding]:
        """Current holdings. Raises a BrokerError subclass on failure — never returns partial data silently."""

    def fetch_prices(self, symbols: list[str]) -> dict[str, Price]:
        """Last traded prices by symbol. {} if this source has no price API."""
        return {}

    @abstractmethod
    def auth_status(self) -> AuthStatus:
        """ok | expired | missing | unsupported."""

    def __repr__(self) -> str:  # never include credentials
        return f"<{type(self).__name__} {self.name}>"
