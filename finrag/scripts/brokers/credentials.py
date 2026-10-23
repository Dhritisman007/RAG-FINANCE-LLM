"""
Broker credentials come from finrag/.env ONLY (git-ignored) — never from the
command line (shell history), never logged, never written to the portfolio DB.

    UPSTOX_ACCESS_TOKEN=...      # daily token, expires 3:30 AM the next day
    DHAN_ACCESS_TOKEN=...        # valid 24 hours
    DHAN_CLIENT_ID=...           # needed for Dhan's price API
    GROWW_ACCESS_TOKEN=...       # expires daily at 6:00 AM
"""
from __future__ import annotations

from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def get_secret(name: str, env_path: Path | None = None) -> str | None:
    """The value of `name` in .env, or None. Reads the file each time (no caching, no os.environ)."""
    path = env_path or ENV_PATH
    if not path.exists():
        return None
    from dotenv import dotenv_values
    value = dotenv_values(path).get(name)
    return value.strip() if value and value.strip() else None
