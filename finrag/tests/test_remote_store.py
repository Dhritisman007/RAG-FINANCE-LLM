"""
Remote store tests: users' files mirrored to Postgres, encrypted. Needs a Postgres to talk to:

    PORTIQ_TEST_DATABASE_URL=postgresql://127.0.0.1:5432/portiq_test python tests/test_remote_store.py

Without it the Postgres tests are skipped (the no-database and missing-key checks still run).
"""
import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import remote_store  # noqa: E402

TEST_DB = os.environ.get("PORTIQ_TEST_DATABASE_URL", "")


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: v for k, v in values.items() if v is not None})
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def key() -> str:
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


def test_off_without_a_database_and_never_unencrypted() -> None:
    with tempfile.TemporaryDirectory() as d, env(PORTIQ_DATABASE_URL=None, PORTIQ_DATA_KEY=None):
        (Path(d) / "holdings.csv").write_text("x")
        assert not remote_store.enabled() and remote_store.push("u", Path(d)) == 0 and remote_store.pull("u", Path(d)) == 0
    with tempfile.TemporaryDirectory() as d, env(PORTIQ_DATABASE_URL="postgresql://nowhere/x", PORTIQ_DATA_KEY=None):
        (Path(d) / "holdings.csv").write_text("x")
        try:
            remote_store.push("u", Path(d))
        except remote_store.StoreMisconfigured as e:
            assert "never stores users' files unencrypted" in str(e)
        else:
            raise AssertionError("a database without a key must be refused before anything is sent")
    print("PASS: no database = local disk only; a database without PORTIQ_DATA_KEY is refused (never plaintext)")


def test_roundtrip_encrypted_and_per_user() -> None:
    if not TEST_DB:
        print("SKIP: set PORTIQ_TEST_DATABASE_URL to run the Postgres tests")
        return
    import psycopg
    k = key()
    with tempfile.TemporaryDirectory() as d, env(PORTIQ_DATABASE_URL=TEST_DB, PORTIQ_DATA_KEY=k):
        with psycopg.connect(TEST_DB, autocommit=True) as c:
            c.execute(remote_store.SCHEMA)
            c.execute("DELETE FROM portiq_files WHERE owner IN ('alice', 'bob')")
        a, b = Path(d) / "alice", Path(d) / "bob"
        (a / "imports").mkdir(parents=True)
        (a / "holdings.csv").write_text("ticker,quantity\nRELIANCE.NS,10\n")
        (a / "imports" / "zerodha.csv").write_text("Instrument,Qty\nTCS,5\n")
        (a / "consent.json").write_text('{"notice": "x"}')
        b.mkdir()
        (b / "holdings.csv").write_text("ticker,quantity\nINFY.NS,2\n")
        assert remote_store.push("alice", a) == 3 and remote_store.push("bob", b) == 1
        assert remote_store.push("alice", a) == 0                             # nothing changed: nothing sent
        (a / "holdings.csv").write_text("ticker,quantity\nRELIANCE.NS,12\n")
        (a / "imports" / "zerodha.csv").unlink()
        assert remote_store.push("alice", a) == 1                             # one changed, one deleted remotely
        with psycopg.connect(TEST_DB) as c:
            rows = c.execute("SELECT owner, path, content FROM portiq_files WHERE owner IN ('alice','bob')").fetchall()
        assert sorted((o, p) for o, p, _ in rows) == [("alice", "consent.json"), ("alice", "holdings.csv"),
                                                      ("bob", "holdings.csv")]
        assert not any(b"RELIANCE" in bytes(c) or b"INFY" in bytes(c) or b"ticker" in bytes(c) for _, _, c in rows)
        shutil.rmtree(a)                                                      # the host wiped its disk
        assert remote_store.pull("alice", a) == 2
        assert (a / "holdings.csv").read_text() == "ticker,quantity\nRELIANCE.NS,12\n" and not (a / "imports").exists()
        assert remote_store.push("alice", a) == 0                             # restored = already in sync
        with env(PORTIQ_DATA_KEY=key()):                                     # someone with another key
            try:
                remote_store.pull("alice", Path(d) / "thief")
            except Exception as e:
                assert type(e).__name__ == "InvalidToken"
            else:
                raise AssertionError("another key must not decrypt the files")
        with psycopg.connect(TEST_DB, autocommit=True) as c:                 # a stored path can't escape the folder
            c.execute("INSERT INTO portiq_files (owner, path, content) VALUES ('bob', '../escape.txt', %s)",
                      (remote_store._cipher().encrypt(b"x"),))
        assert remote_store.pull("bob", Path(d) / "bob2") == 1 and not (Path(d) / "escape.txt").exists()
        remote_store.forget("alice")
        remote_store.forget("bob")
        assert remote_store.pull("alice", Path(d) / "gone") == 0
    print("PASS: per-user files mirrored to Postgres encrypted (no plaintext stored), only changes sent, "
          "restored after a wipe, unreadable with another key, paths can't escape, forget deletes")


def test_usage_counts_survive_a_wipe() -> None:
    if not TEST_DB:
        print("SKIP: usage mirror (no PORTIQ_TEST_DATABASE_URL)")
        return
    import usage
    import psycopg
    with tempfile.TemporaryDirectory() as d, env(PORTIQ_MODE="hosted", PORTIQ_DATA_DIR=d, PORTIQ_DATABASE_URL=TEST_DB,
                                                PORTIQ_DATA_KEY=key(), PORTIQ_DAILY_QUESTIONS="5"):
        with psycopg.connect(TEST_DB, autocommit=True) as c:
            c.execute(remote_store.SCHEMA)
            c.execute("DELETE FROM portiq_files WHERE owner = %s", (usage.SITE_OWNER,))
        assert usage.reserve("carol").ok and usage.reserve("carol").ok
        shutil.rmtree(Path(d) / "site")                                       # restart: disk wiped
        assert usage.left_today("carol") == (3, False)                        # counts came back from Postgres
        with psycopg.connect(TEST_DB, autocommit=True) as c:
            c.execute("DELETE FROM portiq_files WHERE owner = %s", (usage.SITE_OWNER,))
    print("PASS: the daily question counts survive a disk wipe (restored from the remote store)")


if __name__ == "__main__":
    test_off_without_a_database_and_never_unencrypted()
    test_roundtrip_encrypted_and_per_user()
    test_usage_counts_survive_a_wipe()
    print("\nAll remote store tests passed.")
