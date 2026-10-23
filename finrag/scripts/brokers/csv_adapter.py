"""
CSV / broker-export import — works for everyone, no broker approval needed.

A JSON column mapping tells the importer which of the export's headers hold our
fields, so any broker's "download holdings" file can be read:

    {
      "name": "generic",
      "columns": {"symbol": "Symbol", "quantity": "Quantity", "avg_cost": "Average Price",
                  "exchange": "Exchange", "isin": "ISIN", "currency": "Currency"},
      "defaults": {"exchange": "NSE", "currency": "INR"},
      "skip_lines": 0
    }

symbol, quantity and avg_cost are required columns; the others may come from
`defaults`. Every row is validated and every row that isn't imported is reported
with its line number and reason — nothing is dropped silently. Repeated
symbols (several lots of one stock) are merged at their weighted average cost
and reported as merged.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .base import AuthStatus, BrokerAdapter, CredentialsMissing, Holding, MalformedResponse

REQUIRED = ("symbol", "quantity", "avg_cost")
OPTIONAL = ("exchange", "isin", "currency")
EXCHANGES = {"NSE", "BSE", "US", "NASDAQ", "NYSE"}
SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&.\-]{0,19}$")
NUMBER_CLEAN_RE = re.compile(r"[₹$,\s]")


@dataclass
class ImportReport:
    holdings: list[Holding] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)   # (line number, reason)
    merged: list[tuple[int, str]] = field(default_factory=list)    # (line number, note)

    def summary(self) -> str:
        lines = [f"{len(self.holdings)} holding(s) imported"]
        lines += [f"  line {n}: merged — {why}" for n, why in self.merged]
        lines += [f"  line {n}: SKIPPED — {why}" for n, why in self.skipped]
        return "\n".join(lines)


def load_mapping(path: Path) -> dict:
    try:
        mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise MalformedResponse(f"csv: can't read mapping {path} ({e})", "csv")
    cols = mapping.get("columns", {})
    missing = [f for f in REQUIRED if not cols.get(f)]
    if missing:
        raise MalformedResponse(f"csv: mapping {path} has no column for {', '.join(missing)}", "csv")
    return mapping


def _number(raw: str, what: str) -> float:
    text = NUMBER_CLEAN_RE.sub("", raw or "")
    if text.startswith("(") and text.endswith(")"):  # accounting negative
        text = "-" + text[1:-1]
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{what} {raw!r} is not a number")


def parse_rows(text: str, mapping: dict) -> ImportReport:
    """Validate an export's rows against the mapping (pure — no files, no network)."""
    report = ImportReport()
    cols, defaults = mapping["columns"], mapping.get("defaults", {})
    lines = text.splitlines()[int(mapping.get("skip_lines", 0)):]
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=mapping.get("delimiter", ","))
    headers = [h.strip() for h in (reader.fieldnames or [])]
    absent = [f"{f} ({cols[f]!r})" for f in REQUIRED + OPTIONAL if cols.get(f) and cols[f] not in headers]
    if any(f.split(" ")[0] in REQUIRED for f in absent):
        raise MalformedResponse(f"csv: the file has no column for {', '.join(absent)}; its headers are "
                                f"{', '.join(headers) or '(none)'}", "csv")
    first_line = int(mapping.get("skip_lines", 0)) + 2  # header is line 1 after skipped lines
    seen: dict[tuple[str, str], int] = {}
    for offset, row in enumerate(reader):
        n = first_line + offset
        row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        if not any(row.values()):
            report.skipped.append((n, "blank row"))
            continue
        get = lambda f: row.get(cols[f], "") if cols.get(f) else ""
        symbol = get("symbol").upper()
        exchange = (get("exchange") or defaults.get("exchange", "")).upper()
        currency = (get("currency") or defaults.get("currency", "INR")).upper()
        try:
            if not SYMBOL_RE.match(symbol):
                raise ValueError(f"symbol {symbol or '(blank)'!r} isn't a valid trading symbol")
            if exchange not in EXCHANGES:
                raise ValueError(f"exchange {exchange or '(blank)'!r} isn't one of {', '.join(sorted(EXCHANGES))}")
            qty = _number(get("quantity"), "quantity")
            if qty < 0:
                raise ValueError(f"negative quantity {qty:g} (short positions aren't supported)")
            if qty == 0:
                raise ValueError("quantity is 0")
            cost = _number(get("avg_cost"), "average cost")
            if cost < 0:
                raise ValueError(f"negative average cost {cost:g}")
        except ValueError as e:
            report.skipped.append((n, str(e)))
            continue
        exch = "US" if exchange in ("NASDAQ", "NYSE") else exchange
        h = Holding(symbol=symbol, exchange=exch, quantity=qty, avg_cost=cost, currency=currency,
                    source="csv", isin=get("isin") or None)
        key = (symbol, exch)
        if key in seen:
            prev = report.holdings[seen[key]]
            total = prev.quantity + qty
            report.holdings[seen[key]] = Holding(symbol, exch, total, (prev.quantity * prev.avg_cost + qty * cost) / total,
                                                 currency, "csv", prev.isin or h.isin)
            report.merged.append((n, f"{symbol} also on an earlier line — quantities added, average cost weighted"))
        else:
            seen[key] = len(report.holdings)
            report.holdings.append(h)
    return report


class CsvAdapter(BrokerAdapter):
    name = "csv"
    allowed_hosts = ()  # a file import never touches the network

    def __init__(self, file: str | Path | None = None, mapping: str | Path | None = None):
        self.file = Path(file) if file else None
        self.mapping_path = Path(mapping) if mapping else None
        self.last_report: ImportReport | None = None

    def fetch_holdings(self) -> list[Holding]:
        if not self.file or not self.file.exists():
            raise CredentialsMissing(f"csv: no export file{' at ' + str(self.file) if self.file else ''} "
                                     "(pass --file)", self.name)
        if not self.mapping_path:
            raise CredentialsMissing("csv: no column mapping (pass --mapping, e.g. "
                                     "config/broker_mappings/generic_example.json)", self.name)
        self.last_report = parse_rows(self.file.read_text(encoding="utf-8-sig"), load_mapping(self.mapping_path))
        return self.last_report.holdings

    def auth_status(self) -> AuthStatus:
        return AuthStatus.OK if self.file and self.file.exists() and self.mapping_path else AuthStatus.MISSING
