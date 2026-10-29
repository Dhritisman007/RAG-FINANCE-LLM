"""
CAS statement import — the NSDL / CDSL Consolidated Account Statement (monthly e-CAS PDF).

One statement lists every demat account under the investor's PAN, whichever brokers
they use, so this covers brokers with no API for other apps (Groww, for one). The PDF
is parsed on this machine by casparser (MIT, github.com/codereverser/casparser); no
network call is made (tests/test_cas.py runs it under the network guard).

What a CAS has, and doesn't:
  - each equity's ISIN, quantity and the market price on the statement date;
  - NOT what you paid — avg_cost stays unknown (NULL) unless the user enters it;
  - no trading symbol: it comes from NSE's own equity and ETF lists (their ISINs are in
    data/reference/stocks.csv — refresh with stock_list.py --update), falling back to
    casparser's bundled ISIN database. ETFs count whether the statement lists them under
    equities or mutual funds. BSE-only shares, other mutual funds, bonds and NPS are
    listed as skipped.

The password (the first holder's PAN in capital letters, per NSDL and CDSL) is held in
memory only: never logged, stored, or put on a command line. Only holdings are kept —
the investor's name, PAN, address, e-mail, mobile and account numbers are never written
anywhere, and error messages never repeat the password.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from casparser import read_cas_pdf
from casparser.exceptions import IncorrectPasswordError, ParserException
from casparser.types import CASData, NSDLCASData

from .base import (AuthStatus, BrokerAdapter, CredentialsMissing, Holding, IncorrectPassword, MalformedResponse,
                   Price)

PASSWORD_HINT = "the first holder's PAN in capital letters, e.g. ABCDE1234F"
STATEMENT_DATE_FORMATS = ("%d-%b-%Y", "%d/%b/%Y", "%d-%m-%Y", "%d/%m/%Y")  # NSDL: 30-Sep-2026; CDSL varies


@dataclass
class CasReport:
    holdings: list[Holding] = field(default_factory=list)
    prices: dict[str, Price] = field(default_factory=dict)   # statement-date prices, by symbol
    issuer: str = ""                                         # NSDL or CDSL
    period: tuple[str, str] = ("", "")                       # as printed on the statement
    statement_date: str | None = None                        # ISO date the prices are as of
    accounts: int = 0
    account_word: str = "demat account(s)"                   # or "folio(s)" for a mutual-fund statement
    skipped: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{len(self.holdings)} holding(s) imported from the {self.issuer} statement for "
                 f"{self.period[0]} to {self.period[1]} ({self.accounts} {self.account_word})"]
        lines += [f"  merged — {why}" for why in self.merged]
        lines += [f"  SKIPPED — {why}" for why in self.skipped]
        no_cost = [h.symbol for h in self.holdings if h.avg_cost is None]
        if no_cost:
            lines.append(f"  average price unknown (a CAS doesn't include it): {', '.join(no_cost)}")
        lines += [f"  warning — {why}" for why in self.warnings]
        return "\n".join(lines)


def read_statement(pdf, password: str | None, reader=read_cas_pdf) -> NSDLCASData | CASData:
    """Open and parse a CAS. `pdf` is a path or a binary file object; `reader` is swappable for tests."""
    if not password:
        raise CredentialsMissing(f"cas: the statement's password is needed ({PASSWORD_HINT})", "cas")
    try:
        data = reader(pdf, password)
    except IncorrectPasswordError:
        raise IncorrectPassword(f"cas: wrong password — a CAS opens with {PASSWORD_HINT}", "cas") from None
    except ParserException as e:
        if str(e).startswith("Unhandled error while opening PDF"):  # casparser's wrapper for PDFium load errors
            raise MalformedResponse("cas: couldn't open this file as a PDF — is it a CAS from NSDL, CDSL, CAMS or "
                                    "KFintech?", "cas") from None
        raise MalformedResponse(f"cas: this doesn't read as a CAS from NSDL, CDSL, CAMS or KFintech "
                                f"({str(e).replace(password, '***')})", "cas") from None
    except Exception as e:  # anything else from the PDF library: name it, never echo its text
        raise MalformedResponse(f"cas: couldn't open this file as a PDF statement ({type(e).__name__})", "cas") from None
    return data


def _iso_date(text: str) -> str | None:
    for fmt in STATEMENT_DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def nse_isins() -> dict[str, str]:
    """ISIN -> NSE symbol for NSE equities and ETFs, from data/reference/stocks.csv (stock_list.py --update)."""
    import stock_list  # scripts/ is on sys.path wherever brokers is imported
    return {r["isin"]: r["symbol"] for r in stock_list.load_stocks() if r.get("isin")}


def _value_date(value) -> str | None:
    """A valuation date from casparser: a date, or text such as 30-Sep-2026."""
    return value.isoformat() if hasattr(value, "isoformat") else _iso_date(str(value or ""))


def holdings_from_cas(data: NSDLCASData | CASData, costs: dict[str, float] | None = None,
                      listed: dict[str, str] | None = None) -> CasReport:
    """
    Statement -> holdings (pure apart from reading the stock list when `listed` is None).
      - NSDL / CDSL (demat): shares and ETFs by NSE symbol; mutual fund units held in demat as funds.
      - CAMS / KFintech (mutual funds): every scheme with units, with the cost the statement reports.
    `costs` maps ISIN -> average buy price the user entered (it wins over the statement's); `listed`
    maps ISIN -> NSE symbol and fills in what casparser's ISIN database lacks — ETFs, which a demat
    statement may list under equities or under mutual funds depending on its layout.
    """
    costs = costs or {}
    funds_statement = isinstance(data, CASData)
    listed = {} if funds_statement else (nse_isins() if listed is None else listed)
    report = CasReport(issuer=str(getattr(data.file_type, "value", data.file_type)),
                       period=(data.statement_period.from_, data.statement_period.to),
                       statement_date=_iso_date(data.statement_period.to),
                       accounts=len(data.folios if funds_statement else data.accounts),
                       account_word="folio(s)" if funds_statement else "demat account(s)",
                       warnings=list(data.parse_warnings))
    if report.statement_date is None and not funds_statement:
        report.warnings.append(f"statement date {data.statement_period.to!r} not recognised — its prices aren't kept")
    by_isin: dict[str, int] = {}

    def add(symbol: str, exchange: str, qty: float, isin: str, price, cost: float | None = None,
            price_date: str | None = None) -> None:
        """cost: average buy price per unit from the statement, if it has one."""
        symbol = symbol.upper()
        cost = costs.get(isin, cost)
        if isin in by_isin:
            i = by_isin[isin]
            prev = report.holdings[i]
            total = prev.quantity + qty
            avg = ((prev.quantity * prev.avg_cost + qty * cost) / total
                   if prev.avg_cost is not None and cost is not None else None)   # unknown if any part is
            report.holdings[i] = Holding(prev.symbol, prev.exchange, total, costs.get(isin, avg),
                                         prev.currency, prev.source, prev.isin)
            note = f"{prev.symbol} is in more than one {'folio' if funds_statement else 'demat account'} — " \
                   "quantities added"
            if note not in report.merged:
                report.merged.append(note)
            return
        by_isin[isin] = len(report.holdings)
        report.holdings.append(Holding(symbol=symbol, exchange=exchange, quantity=qty, avg_cost=cost,
                                       currency="INR", source="cas", isin=isin))
        when = price_date or report.statement_date
        if when and price and price > 0:
            report.prices[symbol] = Price(symbol, float(price), when)

    def per_unit(total, units) -> float | None:
        return float(total) / units if total is not None and units > 0 else None

    if funds_statement:
        for folio in data.folios:
            for s in folio.schemes:
                units = float(s.close)
                if units <= 0:
                    continue                                   # redeemed in full: nothing held
                if not s.isin:
                    report.skipped.append(f"{s.scheme}: no ISIN on the statement, so it can't be priced")
                    continue
                add(s.isin, "MF", units, s.isin, s.valuation.nav, per_unit(s.valuation.cost, units),
                    _value_date(s.valuation.date))
        return report

    for n, account in enumerate(data.accounts, 1):
        where = f"account {n} ({account.type})"   # never the DP / client ID
        for eq in account.equities:
            what = f"{eq.name or eq.isin} ({eq.isin})"
            qty = float(eq.num_shares)
            if qty <= 0:
                report.skipped.append(f"{what} in {where}: quantity {qty:g}")
                continue
            # NSE's own list first: it is current (casparser's database still has pre-rename symbols such as
            # ICICIPRULI for ICICILIFE); casparser's is the fallback for ISINs the list doesn't carry
            symbol, exchange = (listed[eq.isin], "NSE") if eq.isin in listed else (eq.symbol, (eq.exchange or "").upper())
            if not symbol or exchange not in ("NSE", "BSE"):
                report.skipped.append(f"{what}: no NSE symbol known for this ISIN (it isn't in NSE's equity or ETF "
                                      "lists, e.g. a BSE-only share) — add it with ＋ Add to track it")
                continue
            add(symbol, exchange, qty, eq.isin, eq.price)
        for mf in account.mutual_funds:
            units = float(mf.balance)
            if units <= 0:
                continue
            if mf.isin in listed:                              # an exchange-traded fund held in demat
                add(listed[mf.isin], "NSE", units, mf.isin, mf.nav)
            else:                                              # a mutual fund held in demat
                cost = float(mf.avg_cost) if mf.avg_cost is not None else per_unit(mf.total_cost, units)
                add(mf.isin, "MF", units, mf.isin, mf.nav, cost)
        report.skipped += [f"{bd.name or bd.isin} ({bd.isin}): bonds aren't tracked yet" for bd in account.bonds]
    if data.nps:
        report.skipped.append("NPS holdings aren't tracked yet")
    return report


def load_costs(path: Path) -> dict[str, float]:
    """{ISIN: average buy price} from a JSON file (what the import screen writes)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        costs = {str(k): float(v) for k, v in raw.items()}
    except (OSError, ValueError, TypeError, AttributeError) as e:
        raise MalformedResponse(f"cas: can't read average prices from {path} ({e})", "cas")
    bad = [k for k, v in costs.items() if v < 0]
    if bad:
        raise MalformedResponse(f"cas: negative average price for {', '.join(bad)}", "cas")
    return costs


class CasAdapter(BrokerAdapter):
    name = "cas"
    allowed_hosts = ()  # parsed on this machine — no network at all

    def __init__(self, file: str | Path | None = None, password: str | None = None,
                 costs: dict[str, float] | None = None, reader=read_cas_pdf, listed: dict[str, str] | None = None):
        self.file = Path(file) if file else None
        self._password = password
        self.costs = costs or {}
        self.reader = reader
        self.listed = listed
        self.last_report: CasReport | None = None

    def fetch_holdings(self) -> list[Holding]:
        if not self.file or not self.file.exists():
            raise CredentialsMissing(f"cas: no statement file{' at ' + str(self.file) if self.file else ''} "
                                     "(pass --file)", self.name)
        data = read_statement(self.file, self._password, self.reader)
        self.last_report = holdings_from_cas(data, self.costs, self.listed)
        return self.last_report.holdings

    def fetch_prices(self, symbols: list[str]) -> dict[str, Price]:
        """The statement's own prices (as of its date) — kept as a fallback; live prices are newer."""
        prices = self.last_report.prices if self.last_report else {}
        return {s: prices[s] for s in symbols if s in prices}

    def auth_status(self) -> AuthStatus:
        return AuthStatus.OK if self.file and self.file.exists() and self._password else AuthStatus.MISSING
