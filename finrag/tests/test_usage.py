"""
Daily question limits (website mode). Offline.

    python tests/test_usage.py
"""
import contextlib
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import usage  # noqa: E402
import workspace  # noqa: E402


@contextlib.contextmanager
def hosted(data_dir: Path, per_user: int, site: int):
    keys = ("PORTIQ_MODE", "PORTIQ_DATA_DIR", "PORTIQ_DAILY_QUESTIONS", "PORTIQ_DAILY_QUESTIONS_TOTAL")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(PORTIQ_MODE="hosted", PORTIQ_DATA_DIR=str(data_dir), PORTIQ_DAILY_QUESTIONS=str(per_user),
                      PORTIQ_DAILY_QUESTIONS_TOTAL=str(site))
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_local_mode_has_no_limits() -> None:
    with tempfile.TemporaryDirectory() as d:
        os.environ["PORTIQ_DATA_DIR"] = d
        try:
            assert all(usage.reserve("me").ok for _ in range(100)) and usage.reserve("me").left is None
            assert not any(Path(d).iterdir())                          # nothing recorded locally
        finally:
            os.environ.pop("PORTIQ_DATA_DIR")
    print("PASS: the local app has no question limits and records nothing")


def test_user_and_site_limits() -> None:
    with tempfile.TemporaryDirectory() as d, hosted(Path(d), per_user=3, site=5):
        a = [usage.reserve("alice") for _ in range(4)]
        assert [x.ok for x in a] == [True, True, True, False] and [x.left for x in a] == [2, 1, 0, 0]
        assert a[-1].reason == "user_limit" and usage.left_today("alice") == (0, False)
        b = [usage.reserve("bob") for _ in range(3)]                  # site: 3 + 2 = 5, then full
        assert [x.ok for x in b] == [True, True, False] and b[-1].reason == "site_limit"
        assert usage.left_today("carol") == (3, True)                  # a fresh user, but the site is out
        usage.refund("bob")                                            # a failed answer gives it back
        assert usage.reserve("carol").ok                               # site back at 4 + 1
        usage.note("provider_limit")
        row = usage.report()[0]
        assert (row["answered"], row["users"], row["user_limit"], row["site_limit"], row["provider_limit"]) \
            == (5, 3, 1, 1, 1), row
        real_today = usage.today
        usage.today = lambda: "2099-01-01"                             # midnight (India time) passed
        try:
            assert usage.reserve("alice").ok and usage.left_today("alice") == (2, False)
        finally:
            usage.today = real_today
        usage.forget("alice")                                          # Delete my data
        assert usage.left_today("alice")[0] == 3
        assert "alice" not in Path(usage.db_path()).read_bytes().decode("latin-1")
    print("PASS: per-user and site-wide daily caps, refunds, reset at midnight, report, forget")


def test_simultaneous_questions_cannot_overshoot() -> None:
    with tempfile.TemporaryDirectory() as d, hosted(Path(d), per_user=5, site=1000):
        results = []
        threads = [threading.Thread(target=lambda: results.append(usage.reserve("alice").ok)) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count(True) == 5 and results.count(False) == 15, results
    print("PASS: 20 simultaneous questions with 5 left: exactly 5 go through (atomic reserve)")


if __name__ == "__main__":
    test_local_mode_has_no_limits()
    test_user_and_site_limits()
    test_simultaneous_questions_cannot_overshoot()
    print("\nAll usage tests passed.")
