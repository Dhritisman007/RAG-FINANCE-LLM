"""
Keeps website users' files when the host's own disk doesn't.

Free hosting (Streamlit Community Cloud) wipes its disk on every restart and wake-up, so
in hosted mode each user's folder (scripts/workspace.py) — holdings, imports, snapshots,
consent — and the site's usage counts are mirrored to a Postgres database (e.g. Neon's
free tier). The app keeps working with plain files exactly as before; this module only
copies them:
  - push(owner, folder): upload files that changed since the last push, delete ones removed;
  - pull(owner, folder): restore the folder (on sign-in, when the disk was wiped);
  - forget(owner): Delete my data.

Every file is encrypted before it leaves the server (Fernet, AES-128 + HMAC) with
PORTIQ_DATA_KEY, a secret only the site holds — the database provider sees ciphertext and
the one-way user key, never holdings. Without PORTIQ_DATABASE_URL this is a no-op (local
disk only). With a URL but no key it refuses to start: nothing is ever stored unencrypted.

    PORTIQ_DATABASE_URL=postgresql://...   PORTIQ_DATA_KEY=<Fernet key>
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"   # make a key
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

SYNC_FILE = ".portiq_sync.json"   # this server's note of what it last pushed (never uploaded)
_push_lock = threading.Lock()   # two sessions saving at once must not land out of order
SCHEMA = """CREATE TABLE IF NOT EXISTS portiq_files (
    owner TEXT NOT NULL, path TEXT NOT NULL, content BYTEA NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (owner, path))"""


class StoreMisconfigured(RuntimeError):
    """A database is configured without an encryption key — refuse rather than store plaintext."""


def enabled() -> bool:
    return bool(os.environ.get("PORTIQ_DATABASE_URL", "").strip())


def _cipher():
    from cryptography.fernet import Fernet
    key = os.environ.get("PORTIQ_DATA_KEY", "").strip()
    if not key:
        raise StoreMisconfigured("PORTIQ_DATABASE_URL is set but PORTIQ_DATA_KEY isn't — PortIQ never stores users' "
                                 "files unencrypted. Make a key: python -c \"from cryptography.fernet import Fernet; "
                                 "print(Fernet.generate_key().decode())\"")
    return Fernet(key.encode())


def _connect():
    import psycopg
    conn = psycopg.connect(os.environ["PORTIQ_DATABASE_URL"].strip(), connect_timeout=15, autocommit=True)
    conn.execute(SCHEMA)
    return conn


def _local_files(folder: Path) -> dict[str, bytes]:
    if not folder.exists():
        return {}
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != SYNC_FILE and not p.name.startswith("._")}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def push(owner: str, folder: Path) -> int:
    """Upload what changed since the last push (and delete what was removed). Returns the files written."""
    if not enabled():
        return 0
    with _push_lock:
        return _push(owner, folder)


def _push(owner: str, folder: Path) -> int:
    cipher = _cipher()
    files = _local_files(folder)
    note = folder / SYNC_FILE
    try:
        last = json.loads(note.read_text())
    except (OSError, ValueError):
        last = {}
    now = {path: _digest(data) for path, data in files.items()}
    changed = [p for p in files if last.get(p) != now[p]]
    removed = [p for p in last if p not in files]
    if not changed and not removed:
        return 0
    with _connect() as conn:
        with conn.transaction():
            for path in changed:
                conn.execute("INSERT INTO portiq_files (owner, path, content) VALUES (%s, %s, %s) "
                             "ON CONFLICT (owner, path) DO UPDATE SET content = EXCLUDED.content, updated_at = now()",
                             (owner, path, cipher.encrypt(files[path])))
            for path in removed:
                conn.execute("DELETE FROM portiq_files WHERE owner = %s AND path = %s", (owner, path))
    if folder.exists():
        note.write_text(json.dumps(now))
    return len(changed)


def pull(owner: str, folder: Path) -> int:
    """Restore `owner`'s files into `folder` (overwriting). Returns the number restored."""
    if not enabled():
        return 0
    cipher = _cipher()
    with _connect() as conn:
        rows = conn.execute("SELECT path, content FROM portiq_files WHERE owner = %s", (owner,)).fetchall()
    restored = {}
    for path, blob in rows:
        target = (folder / path).resolve()
        if folder.resolve() not in target.parents:     # a stored path must stay inside the folder
            continue
        data = cipher.decrypt(bytes(blob))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        restored[path] = _digest(data)
    if restored:
        (folder / SYNC_FILE).write_text(json.dumps(restored))
    return len(restored)


def forget(owner: str) -> None:
    """Delete my data: every stored file of this owner."""
    if enabled():
        with _connect() as conn:
            conn.execute("DELETE FROM portiq_files WHERE owner = %s", (owner,))
