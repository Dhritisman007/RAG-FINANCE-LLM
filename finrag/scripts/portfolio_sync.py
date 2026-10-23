"""
Portfolio sync — pull holdings (and prices, where the source has them) from one
source into an atomic SQLite snapshot that the agent reads.

    python scripts/portfolio_sync.py --source csv --file export.csv --mapping config/broker_mappings/generic_example.json
    python scripts/portfolio_sync.py --source cas --file statement.pdf   # NSDL/CDSL CAS; asks for the PDF password
    python scripts/portfolio_sync.py --source upstox            # uses UPSTOX_ACCESS_TOKEN from finrag/.env
    python scripts/portfolio_sync.py --source dhan --dry-run    # fetch and show, write nothing
    python scripts/portfolio_sync.py --status                   # auth state and snapshot age per source

Rules:
  - The network is limited to the adapter's own broker host while a sync runs
    (net_guard); a CSV or CAS import makes no network call at all.
  - A failed sync writes a sync_log row and leaves the last good snapshot untouched.
  - If prices fail, holdings are kept and prices are stored as unavailable (NULL), never 0.
  - No credential is printed, logged, or stored.
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import portfolio_db  # noqa: E402
from brokers import BrokerAdapter, BrokerError, get_adapter, list_adapters  # noqa: E402
from net_guard import NetworkBlocked, outbound_only  # noqa: E402


@dataclass
class SyncResult:
    source: str
    ok: bool
    fetched_at: str
    holdings_count: int = 0
    prices_status: str = ""
    error_type: str | None = None
    message: str = ""
    report: str = ""        # CSV import report (skipped / merged rows)
    dry_run: bool = False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sync(source: str, dry_run: bool = False, db_path: Path = portfolio_db.DB_PATH, adapter=None,
         label: str | None = None, **adapter_kwargs) -> SyncResult:
    """Sync one source. A CSV or CAS import can carry a label ("zerodha", "mom") so several files
    coexist as separate snapshots ("csv:zerodha"); brokers are always stored under their own name."""
    adapter = adapter or get_adapter(source, **adapter_kwargs)
    if label and source in ("csv", "cas"):
        source = f"{source}:{label}"
    started = _now()
    with outbound_only(adapter.allowed_hosts):
        try:
            holdings = adapter.fetch_holdings()
        except (BrokerError, NetworkBlocked) as e:  # NetworkBlocked: an adapter tried a host it didn't declare
            result = SyncResult(source, False, started, error_type=type(e).__name__, message=str(e))
            if not dry_run:
                conn = portfolio_db.connect(db_path)
                try:
                    portfolio_db.log_sync(conn, source, started, _now(), "failed", result.error_type, result.message)
                finally:
                    conn.close()
            return result
        prices, prices_status = {}, "not supported by this source"
        if holdings:
            try:
                prices = adapter.fetch_prices([h.symbol for h in holdings])
                missing = [h.symbol for h in holdings if h.symbol not in prices]
                if prices:
                    prices_status = "ok" if not missing else f"partial — unavailable for {', '.join(missing)}"
                elif type(adapter).fetch_prices is not BrokerAdapter.fetch_prices:  # it has a price API
                    prices_status = "unavailable — the source returned no prices"
            except (BrokerError, NetworkBlocked) as e:
                prices, prices_status = {}, f"unavailable — {type(e).__name__}: {e}"
    report = adapter.last_report.summary() if getattr(adapter, "last_report", None) else ""
    result = SyncResult(source, True, started, len(holdings), prices_status, report=report, dry_run=dry_run)
    if dry_run:
        result.message = "dry run — nothing written"
        return result
    conn = portfolio_db.connect(db_path)
    try:
        portfolio_db.write_snapshot(conn, source, started, holdings, prices, prices_status)
        portfolio_db.log_sync(conn, source, started, _now(), "ok", holdings_count=len(holdings),
                              message=f"prices: {prices_status}")
    finally:
        conn.close()
    return result


def status(db_path: Path = portfolio_db.DB_PATH, online: bool = True, sources: list[str] | None = None) -> list[dict]:
    """Per source: auth state (online check for brokers with credentials) and snapshot age."""
    snaps = {s["source"]: s for s in portfolio_db.latest_snapshots(db_path)}
    rows = []
    for name in sources or list_adapters():
        adapter = get_adapter(name)
        if name in ("csv", "cas"):
            auth = "n/a (file import)"
        elif not online:
            auth = "not checked"
        else:
            try:
                with outbound_only(adapter.allowed_hosts):
                    auth = adapter.auth_status().value
            except BrokerError as e:
                auth = f"unknown ({type(e).__name__})"
        snap = snaps.get(name)
        age = ""
        if snap:
            delta = datetime.now() - datetime.fromisoformat(snap["fetched_at"])
            hours = delta.total_seconds() / 3600
            age = f"{hours:.1f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"
        last = portfolio_db.last_sync(db_path, name)
        rows.append({"source": name, "auth": auth,
                     "snapshot": f"{snap['holdings_count']} holdings, {age} ({snap['fetched_at']})" if snap else "none",
                     "last_sync": f"{last['status']}{' — ' + last['error_type'] if last and last['error_type'] else ''}"
                                  if last else "never"})
    if not sources:  # labelled imports ("csv:zerodha", "cas:mom") are snapshot sources without their own adapter
        for name, snap in snaps.items():
            if name.startswith(("csv:", "cas:")):
                rows.append({"source": name, "auth": "n/a (file import)",
                             "snapshot": f"{snap['holdings_count']} holdings ({snap['fetched_at']})",
                             "last_sync": (portfolio_db.last_sync(db_path, name) or {}).get("status", "never")})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync holdings from a CSV export or a broker (read-only).")
    parser.add_argument("--source", choices=list_adapters())
    parser.add_argument("--file", help="CSV export (csv source) or CAS statement PDF (cas source)")
    parser.add_argument("--mapping", help="Column-mapping JSON (csv source)")
    parser.add_argument("--password-stdin", action="store_true",
                        help="Read the CAS password from the first line of stdin (otherwise it is asked for)")
    parser.add_argument("--costs", type=Path, help="JSON {ISIN: average buy price} (cas source; a CAS has none)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and show, write nothing")
    parser.add_argument("--status", action="store_true", help="Auth state and snapshot age per source")
    parser.add_argument("--label", help="Name for a CSV import (e.g. zerodha): keeps several exports apart")
    parser.add_argument("--remove", metavar="SOURCE", help="Delete a source's snapshots (e.g. upstox, csv:zerodha)")
    parser.add_argument("--json", action="store_true", help="Machine-readable output (used by the app)")
    parser.add_argument("--db", type=Path, default=portfolio_db.DB_PATH, help=argparse.SUPPRESS)  # tests
    args = parser.parse_args()
    if args.remove:
        n = portfolio_db.delete_source(args.db, args.remove)
        print(json.dumps({"removed": n, "source": args.remove}) if args.json else f"Removed {n} snapshot(s) of {args.remove}")
        return
    if args.status:
        rows = status(args.db, sources=[args.source] if args.source else None)
        if args.json:
            print(json.dumps(rows))
            return
        for row in rows:
            print(f"{row['source']:7}  auth: {row['auth']:18}  snapshot: {row['snapshot']}  last sync: {row['last_sync']}")
        return
    if not args.source:
        parser.error("--source is required (or use --status)")
    kwargs = {"file": args.file, "mapping": args.mapping} if args.source == "csv" else {}
    if args.source == "cas":  # the password never goes on the command line (it is the holder's PAN)
        from brokers.base import BrokerError as _BrokerError
        from brokers.cas_adapter import PASSWORD_HINT, load_costs
        password = (sys.stdin.readline().rstrip("\r\n") if args.password_stdin else
                    getpass.getpass(f"CAS password ({PASSWORD_HINT}): ") if sys.stdin.isatty() else None)
        try:
            costs = load_costs(args.costs) if args.costs else {}
        except _BrokerError as e:
            parser.error(str(e))
        kwargs = {"file": args.file, "password": password, "costs": costs}
    result = sync(args.source, dry_run=args.dry_run, db_path=args.db, label=args.label, **kwargs)
    if args.json:
        print(json.dumps(asdict(result)))
        sys.exit(0 if result.ok else 1)
    if result.report:
        print(result.report)
    if result.ok:
        print(f"{result.source}: {result.holdings_count} holding(s) · prices: {result.prices_status} · "
              f"as of {result.fetched_at}{' · ' + result.message if result.message else ''}")
    else:
        print(f"{result.source}: sync FAILED ({result.error_type}) — {result.message}\n"
              "The previous snapshot (if any) is unchanged.")
        sys.exit(1)


if __name__ == "__main__":
    main()
