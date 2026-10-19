"""
Day 16: Refresh SEC filings — pick up new 10-Ks / 10-Qs as companies file them.

For every company in config/corpus_manifest.yaml, asks the SEC which filings
are now its latest 10-K and two latest 10-Qs (the manifest's rule) and compares
that with data/raw/tier2. Only filings not already on disk are downloaded, with
the same extractor as scrape_tier2_edgar.py.

Older filings are KEPT: the golden evaluation set cites specific filings, and
answers stay reproducible. Over time a company accumulates history (a new 10-Q
doesn't delete the previous one); retrieval ranks by relevance, and each
passage's source label carries its filing date.

Usage:
    python scripts/refresh_filings.py              # dry run: list what's new, download nothing
    python scripts/refresh_filings.py --apply      # download new filings
    python scripts/refresh_filings.py --apply --rebuild   # ... then chunk, enrich, rebuild the store (~25 min)
    python scripts/refresh_filings.py --ticker AAPL       # one company
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scrape_tier2_edgar as edgar  # noqa: E402

ROOT = Path(__file__).parent.parent
PIPELINE = ["chunk_documents.py", "enrich_metadata.py", "build_vectorstore.py"]


def filing_slug(ticker: str, filing: dict) -> str:
    """Same file naming as scrape_tier2_edgar.save_doc()."""
    return f"tier2_secedgar_{ticker}-{filing['form'].replace('-', '')}-{filing['filingDate']}"


def on_disk(out_dir: Path = edgar.OUT_DIR) -> set[str]:
    return {p.stem for p in out_dir.glob("tier2_secedgar_*.txt") if not p.name.startswith("._")}


def find_new_filings(companies: list[dict], existing: set[str], get_targets=None) -> list[tuple[str, dict]]:
    """[(ticker, filing), ...] for each company's current target filings that aren't on disk yet."""
    get_targets = get_targets or edgar.get_target_filings
    new = []
    for company in companies:
        targets = get_targets(company["cik"], company["filings"])
        new += [(company["ticker"], f) for f in targets if filing_slug(company["ticker"], f) not in existing]
        time.sleep(edgar.REQUEST_DELAY_SEC)
    return new


def download(new: list[tuple[str, dict]]) -> int:
    saved = 0
    for ticker, filing in new:
        url = edgar.build_document_url(filing)
        html = edgar.fetch_html(url)
        if html is None:
            continue
        edgar.save_doc(ticker, filing, url, edgar.extract_filing_text(html))
        saved += (edgar.OUT_DIR / f"{filing_slug(ticker, filing)}.txt").exists()
        time.sleep(edgar.REQUEST_DELAY_SEC)
    return saved


def rebuild() -> None:
    for script in PIPELINE:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC filings newer than those on disk.")
    parser.add_argument("--apply", action="store_true", help="Download new filings (default: dry run).")
    parser.add_argument("--rebuild", action="store_true", help="After --apply, rebuild the vector store.")
    parser.add_argument("--ticker", default=None, help="Only this company.")
    args = parser.parse_args()

    companies = edgar.load_manifest()["tier2_filings"]["companies"]
    if args.ticker:
        companies = [c for c in companies if c["ticker"].upper() == args.ticker.upper()]
        if not companies:
            sys.exit(f"{args.ticker} is not in the manifest")

    print(f"Checking {len(companies)} companies against the SEC…")
    new = find_new_filings(companies, on_disk())
    if not new:
        print("Up to date — no new filings.")
        return
    print(f"{len(new)} new filing(s):")
    for ticker, f in new:
        print(f"  {ticker:6} {f['form']:5} filed {f['filingDate']}")
    if not args.apply:
        print("\nDry run — nothing downloaded. Re-run with --apply (and --rebuild to update the store).")
        return

    saved = download(new)
    print(f"\nDownloaded {saved} of {len(new)} new filing(s).")
    if saved and args.rebuild:
        rebuild()
    elif saved:
        print("Rebuild the store to search them:  python scripts/refresh_filings.py --rebuild … or run "
              + " → ".join(PIPELINE))


if __name__ == "__main__":
    main()
