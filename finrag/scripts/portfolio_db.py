"""
Portfolio snapshots in SQLite (data/portfolio/portfolio.db, git-ignored).

    snapshots(id, source, fetched_at, holdings_count, prices_status)
    holdings(snapshot_id, symbol, exchange, ticker, isin, quantity, avg_cost, currency,
             last_price, price_as_of)       -- last_price NULL when unavailable, never 0
    sync_log(id, source, started_at, finished_at, status, error_type, message, holdings_count)

A snapshot is written in one transaction, so a failed or interrupted sync leaves
the previous good snapshot exactly as it was. No credentials are ever stored.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "portfolio" / "portfolio.db"
KEEP_SNAPSHOTS = 10  # per source

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, fetched_at TEXT NOT NULL,
    holdings_count INTEGER NOT NULL, prices_status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS holdings (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL, exchange TEXT NOT NULL, ticker TEXT NOT NULL, isin TEXT,
    quantity REAL NOT NULL, avg_cost REAL NOT NULL, currency TEXT NOT NULL,
    last_price REAL CHECK (last_price IS NULL OR last_price > 0), price_as_of TEXT);
CREATE TABLE IF NOT EXISTS sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, started_at TEXT NOT NULL,
    finished_at TEXT, status TEXT NOT NULL, error_type TEXT, message TEXT, holdings_count INTEGER);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # explicit BEGIN/COMMIT below
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def write_snapshot(conn: sqlite3.Connection, source: str, fetched_at: str, holdings, prices: dict,
                   prices_status: str) -> int:
    """Insert a snapshot and its holdings atomically; prune old snapshots of that source."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute("INSERT INTO snapshots (source, fetched_at, holdings_count, prices_status) VALUES (?,?,?,?)",
                           (source, fetched_at, len(holdings), prices_status))
        sid = cur.lastrowid
        for h in holdings:
            p = prices.get(h.symbol)
            conn.execute("INSERT INTO holdings VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (sid, h.symbol, h.exchange, h.ticker, h.isin, h.quantity, h.avg_cost, h.currency,
                          p.last_price if p else None, p.as_of if p else None))
        conn.execute("DELETE FROM snapshots WHERE source = ? AND id NOT IN "
                     "(SELECT id FROM snapshots WHERE source = ? ORDER BY id DESC LIMIT ?)",
                     (source, source, KEEP_SNAPSHOTS))
        conn.execute("COMMIT")
        return sid
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def log_sync(conn: sqlite3.Connection, source: str, started_at: str, finished_at: str, status: str,
             error_type: str | None = None, message: str | None = None, holdings_count: int | None = None) -> None:
    conn.execute("INSERT INTO sync_log (source, started_at, finished_at, status, error_type, message, holdings_count) "
                 "VALUES (?,?,?,?,?,?,?)", (source, started_at, finished_at, status, error_type, message, holdings_count))


def delete_source(path: Path, source: str) -> int:
    """Remove every snapshot of one source (its holdings go with them); sync_log is kept. Returns the count."""
    if not path.exists():
        return 0
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        n = conn.execute("DELETE FROM snapshots WHERE source = ?", (source,)).rowcount
        conn.execute("COMMIT")
        return n
    finally:
        conn.close()


def latest_snapshots(path: Path = DB_PATH) -> list[dict]:
    """The newest snapshot of every source: [{source, fetched_at, prices_status, holdings: [row dicts]}]."""
    if not path.exists():
        return []
    conn = connect(path)
    try:
        snaps = conn.execute("SELECT * FROM snapshots WHERE id IN (SELECT MAX(id) FROM snapshots GROUP BY source) "
                             "ORDER BY source").fetchall()
        return [{**dict(s), "holdings": [dict(h) for h in conn.execute(
            "SELECT * FROM holdings WHERE snapshot_id = ? ORDER BY ticker", (s["id"],))]} for s in snaps]
    finally:
        conn.close()


def last_sync(path: Path = DB_PATH, source: str | None = None) -> dict | None:
    if not path.exists():
        return None
    conn = connect(path)
    try:
        row = conn.execute("SELECT * FROM sync_log WHERE (? IS NULL OR source = ?) ORDER BY id DESC LIMIT 1",
                           (source, source)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
