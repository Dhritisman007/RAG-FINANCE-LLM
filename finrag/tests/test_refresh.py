"""
Filing-refresh tests. No network: the SEC lookup is replaced by a fake.

    python tests/test_refresh.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import refresh_filings as rf  # noqa: E402
import scrape_tier2_edgar as edgar  # noqa: E402

edgar.REQUEST_DELAY_SEC = 0  # no politeness delay against a fake

COMPANIES = [{"ticker": "JPM", "cik": "0000019617", "filings": ["10-K", "10-Q"]},
             {"ticker": "AAPL", "cik": "0000320193", "filings": ["10-K", "10-Q"]}]
SEC_NOW = {  # what the SEC reports as each company's current latest 10-K + 2 10-Qs
    "0000019617": [{"form": "10-K", "filingDate": "2026-02-13"}, {"form": "10-Q", "filingDate": "2026-08-06"},
                   {"form": "10-Q", "filingDate": "2026-05-01"}],
    "0000320193": [{"form": "10-K", "filingDate": "2025-10-31"}, {"form": "10-Q", "filingDate": "2026-07-31"},
                   {"form": "10-Q", "filingDate": "2026-05-01"}],
}
ON_DISK = {"tier2_secedgar_JPM-10K-2026-02-13", "tier2_secedgar_JPM-10Q-2026-05-01", "tier2_secedgar_JPM-10Q-2025-11-04",
           "tier2_secedgar_AAPL-10K-2025-10-31", "tier2_secedgar_AAPL-10Q-2026-07-31", "tier2_secedgar_AAPL-10Q-2026-05-01"}


def test_only_filings_missing_from_disk_are_new() -> None:
    new = rf.find_new_filings(COMPANIES, ON_DISK, get_targets=lambda cik, forms: SEC_NOW[cik])
    assert [(t, f["form"], f["filingDate"]) for t, f in new] == [("JPM", "10-Q", "2026-08-06")], new
    print("PASS: only the JPM 10-Q filed 2026-08-06 is new; up-to-date AAPL needs nothing")


def test_slug_matches_scraper_naming() -> None:
    assert rf.filing_slug("JPM", {"form": "10-Q", "filingDate": "2026-08-06"}) == "tier2_secedgar_JPM-10Q-2026-08-06"
    print("PASS: refresh uses the scraper's file naming, so existing filings are recognised")


def test_older_filings_are_kept() -> None:
    # The 2025-11-04 JPM 10-Q is no longer among the SEC's latest two, but nothing deletes it:
    # the refresh only ever adds files (golden questions cite specific filings).
    assert not hasattr(rf, "prune") and "unlink" not in Path(rf.__file__).read_text()
    print("PASS: refresh never deletes superseded filings")


if __name__ == "__main__":
    test_only_filings_missing_from_disk_are_new()
    test_slug_matches_scraper_naming()
    test_older_filings_are_kept()
    print("\nAll refresh tests passed.")
