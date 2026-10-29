"""
Whose portfolio this request is about.

Local mode (the default): one person, data in data/portfolio/ exactly as before —
nothing here changes it.

Hosted mode (PORTIQ_MODE=hosted, the website): many people share one server
process, so every script run says whose data it handles with `use(workspace)`,
and every default path (holdings file, snapshot DB, imports, saved column
mappings) resolves through `current()`. Each signed-in user gets a private folder
named by a hash of their sign-in ID. If code reaches for personal data with no
user set, `current()` raises instead of falling back to a shared file — a missing
`use()` fails loudly rather than mixing two people's holdings.

A context variable, not a module global: Streamlit runs each session's script in
its own thread, and a global set for one user would be seen by every other.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
NOTICE_VERSION = "2026-10-29"   # bump when the privacy notice changes: everyone is asked again


class NoWorkspace(RuntimeError):
    """Hosted mode, but this request has no signed-in user — personal data must not be touched."""


def hosted() -> bool:
    return os.environ.get("PORTIQ_MODE", "").strip().lower() == "hosted"


def data_root() -> Path:
    """Where hosted users' folders live (a Hugging Face Space mounts persistent storage at /data)."""
    return Path(os.environ.get("PORTIQ_DATA_DIR", "/data/portiq"))


@dataclass(frozen=True)
class Workspace:
    root: Path
    user_key: str            # hash of the sign-in ID — never the e-mail or the ID itself

    @property
    def holdings(self) -> Path:
        return self.root / "holdings.csv"

    @property
    def db(self) -> Path:
        return self.root / "portfolio.db"

    @property
    def imports(self) -> Path:
        return self.root / "imports"

    @property
    def mappings(self) -> Path:
        return self.root / "mappings"

    @property
    def consent_file(self) -> Path:
        return self.root / "consent.json"

    def has_consented(self) -> bool:
        try:
            return json.loads(self.consent_file.read_text())["notice"] == NOTICE_VERSION
        except (OSError, ValueError, KeyError):
            return False

    def record_consent(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.consent_file.write_text(json.dumps({"notice": NOTICE_VERSION,
                                                 "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))

    def delete_everything(self) -> None:
        """Delete my data: the whole folder — holdings, imports, snapshots, mappings, consent."""
        if self.root.exists():
            shutil.rmtree(self.root)


def user_key(issuer: str, subject: str) -> str:
    """Stable, non-reversible folder name for one sign-in identity (issuer + subject, as OIDC defines it)."""
    if not subject:
        raise NoWorkspace("sign-in returned no user ID")
    return hashlib.sha256(f"{issuer}\n{subject}".encode()).hexdigest()[:32]


def for_user(issuer: str, subject: str) -> Workspace:
    key = user_key(issuer, subject)
    return Workspace(data_root() / "users" / key, key)


_current: ContextVar[Workspace | None] = ContextVar("portiq_workspace", default=None)


def use(ws: Workspace | None) -> None:
    """Set whose data this script run handles. Called at the top of every run (None = nobody yet)."""
    _current.set(ws)


def current() -> Workspace | None:
    """The workspace of this run; None in local mode (callers then use their local paths)."""
    ws = _current.get()
    if ws is None and hosted():
        raise NoWorkspace("hosted mode: no signed-in user for this request, so no personal data is read")
    return ws


def cache_scope() -> str:
    """For caches shared across sessions (answers): never let one user's entry serve another."""
    ws = _current.get()
    return ws.user_key if ws else ("nobody" if hosted() else "local")
