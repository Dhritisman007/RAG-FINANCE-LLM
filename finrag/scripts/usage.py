"""
Daily question limits and counts for the website (hosted mode). Local mode has no limits.

The website runs on free tiers (Groq's free API: a shared daily quota for all users), so:
  - each user gets PORTIQ_DAILY_QUESTIONS answers a day (default 20);
  - the whole site gets PORTIQ_DAILY_QUESTIONS_TOTAL a day (default 900 — below Groq's
    free ~1,000 requests/day on its larger models, so PortIQ says "come back tomorrow"
    before the provider starts refusing);
  - only questions that reach the AI model count: a cached repeat, or a question with no
    matching passages, is free. A failed answer is refunded.
Days reset at midnight India time. Upgrading = raising the two numbers (no code change).

Stored per day: the user's one-way key (workspace.user_key) and counts — never question
text. Kept 90 days. `python scripts/usage.py` prints the last two weeks, the evidence for
whether a paid plan is needed.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import workspace

IST = ZoneInfo("Asia/Kolkata")
KEEP_DAYS = 90
EVENTS = ("answered", "user_limit", "site_limit", "provider_limit", "provider_busy")
SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (day TEXT NOT NULL, user_key TEXT NOT NULL, n INTEGER NOT NULL,
                                      PRIMARY KEY (day, user_key));
CREATE TABLE IF NOT EXISTS events (day TEXT NOT NULL, kind TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (day, kind));
"""


def user_limit() -> int:
    return int(os.environ.get("PORTIQ_DAILY_QUESTIONS", "20"))


def site_limit() -> int:
    return int(os.environ.get("PORTIQ_DAILY_QUESTIONS_TOTAL", "900"))


def today() -> str:
    return datetime.now(IST).date().isoformat()


def db_path() -> Path:
    return workspace.data_root() / "usage.db"


def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=10)
    conn.executescript(SCHEMA)
    return conn


@dataclass
class Allowance:
    ok: bool
    left: int | None = None          # this user's questions left today (None: no limit — local mode)
    reason: str | None = None        # "user_limit" or "site_limit" when not ok


def _note(conn: sqlite3.Connection, kind: str, day: str, n: int = 1) -> None:
    conn.execute("INSERT INTO events VALUES (?,?,?) ON CONFLICT(day, kind) DO UPDATE SET n = n + ?", (day, kind, n, n))


def reserve(user_key: str, path: Path | None = None) -> Allowance:
    """Take one question for this user, if both limits allow it (atomic). Local mode: always allowed."""
    if not workspace.hosted():
        return Allowance(True)
    day = today()
    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        used = conn.execute("SELECT COALESCE(SUM(n), 0) FROM questions WHERE day = ?", (day,)).fetchone()[0]
        mine = conn.execute("SELECT COALESCE(SUM(n), 0) FROM questions WHERE day = ? AND user_key = ?",
                            (day, user_key)).fetchone()[0]
        reason = "site_limit" if used >= site_limit() else "user_limit" if mine >= user_limit() else None
        if reason:
            _note(conn, reason, day)
        else:
            conn.execute("INSERT INTO questions VALUES (?,?,1) ON CONFLICT(day, user_key) DO UPDATE SET n = n + 1",
                         (day, user_key))
            _note(conn, "answered", day)
            mine += 1
            cutoff = (datetime.now(IST).date() - timedelta(days=KEEP_DAYS)).isoformat()
            conn.execute("DELETE FROM questions WHERE day < ?", (cutoff,))
            conn.execute("DELETE FROM events WHERE day < ?", (cutoff,))
        conn.execute("COMMIT")
        return Allowance(reason is None, max(0, user_limit() - mine), reason)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def refund(user_key: str, path: Path | None = None) -> None:
    """The answer failed (provider error): give the question back."""
    if not workspace.hosted():
        return
    day = today()
    conn = _connect(path)
    try:
        conn.execute("UPDATE questions SET n = MAX(0, n - 1) WHERE day = ? AND user_key = ?", (day, user_key))
        conn.execute("UPDATE events SET n = MAX(0, n - 1) WHERE day = ? AND kind = 'answered'", (day,))
    finally:
        conn.close()


def note(kind: str, path: Path | None = None) -> None:
    """Count an event (e.g. the provider refused for quota) for the daily report."""
    if not workspace.hosted():
        return
    assert kind in EVENTS, kind
    conn = _connect(path)
    try:
        _note(conn, kind, today())
    finally:
        conn.close()


def left_today(user_key: str, path: Path | None = None) -> tuple[int, bool]:
    """(questions this user has left today, whether the whole site is out for today)."""
    path = path or db_path()
    if not path.exists():
        return user_limit(), False
    conn = _connect(path)
    try:
        day = today()
        used = conn.execute("SELECT COALESCE(SUM(n), 0) FROM questions WHERE day = ?", (day,)).fetchone()[0]
        mine = conn.execute("SELECT COALESCE(SUM(n), 0) FROM questions WHERE day = ? AND user_key = ?",
                            (day, user_key)).fetchone()[0]
        return max(0, user_limit() - mine), used >= site_limit()
    finally:
        conn.close()


def forget(user_key: str, path: Path | None = None) -> None:
    """Delete my data: this user's counts go too (the site's daily totals stay)."""
    path = path or db_path()
    if path.exists():
        conn = _connect(path)
        try:
            conn.execute("DELETE FROM questions WHERE user_key = ?", (user_key,))
        finally:
            conn.close()


def report(days: int = 14, path: Path | None = None) -> list[dict]:
    """Per day, newest first: answered, distinct users, and how often each limit was hit."""
    path = path or db_path()
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        rows = []
        for (day,) in conn.execute("SELECT DISTINCT day FROM events UNION SELECT DISTINCT day FROM questions "
                                   "ORDER BY 1 DESC LIMIT ?", (days,)):
            ev = dict(conn.execute("SELECT kind, n FROM events WHERE day = ?", (day,)).fetchall())
            users = conn.execute("SELECT COUNT(*) FROM questions WHERE day = ? AND n > 0", (day,)).fetchone()[0]
            rows.append({"day": day, "users": users, **{k: ev.get(k, 0) for k in EVENTS}})
        return rows
    finally:
        conn.close()


if __name__ == "__main__":
    rows = report()
    if not rows:
        print(f"No usage recorded yet ({db_path()}).")
    print(f"{'day':10}  {'answered':>8}  {'users':>5}  {'user cap':>8}  {'site cap':>8}  {'AI quota':>8}  {'AI busy':>7}")
    for r in rows:
        print(f"{r['day']:10}  {r['answered']:>8}  {r['users']:>5}  {r['user_limit']:>8}  {r['site_limit']:>8}  "
              f"{r['provider_limit']:>8}  {r['provider_busy']:>7}")
    print(f"Limits: {user_limit()} per user, {site_limit()} for the whole site per day "
          "(PORTIQ_DAILY_QUESTIONS / PORTIQ_DAILY_QUESTIONS_TOTAL). Often hitting 'site cap' or 'AI quota' = time to upgrade.")
