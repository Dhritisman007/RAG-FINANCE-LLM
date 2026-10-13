"""
Day 3: Tier 2 (SEC EDGAR filings) scraper.

For each company in config/corpus_manifest.yaml:
  1. Fetch the filing index (data.sec.gov/submissions/CIK##########.json)
  2. Filter for the latest 10-K and latest 2 10-Q filings
  3. Build the real document URL and fetch the filing HTML
  4. Extract clean text, save as .txt + .meta.json (same schema as Tier 1)

Usage:
    python scripts/scrape_tier2_edgar.py
    python scripts/scrape_tier2_edgar.py --ticker AAPL   # single company, for testing
    python scripts/scrape_tier2_edgar.py --reextract     # re-fetch + re-extract the
                                                         # filings already on disk

IMPORTANT: SEC requires a descriptive User-Agent with a real contact email
on every request, or you will get blocked. Edit CONTACT_EMAIL below.
"""
import argparse
import json
import re
import time
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# EDGAR iXBRL documents are XHTML; the lxml HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).parent.parent
MANIFEST_PATH = ROOT / "config" / "corpus_manifest.yaml"
OUT_DIR = ROOT / "data" / "raw" / "tier2"

CONTACT_EMAIL = "dhritismansarma@gmail.com"  # <-- EDIT THIS before running
HEADERS = {"User-Agent": f"FinRAG-Coach-Bot/0.1 ({CONTACT_EMAIL})"}

REQUEST_DELAY_SEC = 0.5  # SEC allows up to ~10 req/sec but be conservative


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def fetch_json(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [FETCH FAILED] {url} -> {e}")
        return None


def fetch_html(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  [FETCH FAILED] {url} -> {e}")
        return None


def get_target_filings(cik: str, wanted_forms: list[str]) -> list[dict]:
    """
    Query the SEC submissions API and return the latest N filings per form type.
    CIK must be zero-padded to 10 digits in the URL.
    """
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    data = fetch_json(url)
    if data is None:
        return []

    recent = data["filings"]["recent"]
    n = len(recent["form"])
    filings = []
    for i in range(n):
        form = recent["form"][i]
        if form in wanted_forms:
            filings.append({
                "form": form,
                "accessionNumber": recent["accessionNumber"][i],
                "primaryDocument": recent["primaryDocument"][i],
                "filingDate": recent["filingDate"][i],
                "cik": cik,
            })

    # keep latest 1x 10-K and latest 2x 10-Q, matching the manifest's rule
    result = []
    result += [f for f in filings if f["form"] == "10-K"][:1]
    result += [f for f in filings if f["form"] == "10-Q"][:2]
    return result


def build_document_url(filing: dict) -> str:
    accession_nodash = filing["accessionNumber"].replace("-", "")
    cik_no_leading_zeros = str(int(filing["cik"]))
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_nodash}/{filing['primaryDocument']}"
    )


def extract_filing_text(html: str) -> str:
    """
    SEC filing HTML is large and mostly just document body — strip scripts/styles,
    keep headings/paragraphs/table rows so section structure survives for Day 4 chunking.
    """
    soup = BeautifulSoup(html, "lxml")
    # ix:header is the hidden machine-readable XBRL block (contexts, units,
    # dei facts) — not filing prose, and tens of thousands of tokens of noise.
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()

    # Emit each piece of text exactly once:
    #   - tables as one flattened line (and nothing nested inside them)
    #   - otherwise only the innermost block elements; <span> is inline, so it
    #     is never emitted on its own. EDGAR's iXBRL wraps every paragraph as
    #     <div><span>text</span></div> — emitting both doubled ~23% of lines.
    lines = []
    for el in soup.find_all(BLOCK_TAGS + ["table"]):
        if el.find_parent("table"):
            continue
        if el.name != "table" and el.find(BLOCK_TAGS + ["table"]):
            continue
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 3:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append(f"\n## {text}\n")
        else:
            lines.append(text)

    lines = drop_page_furniture(lines)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "div"]
FURNITURE_MAX_CHARS = 80
FURNITURE_MIN_REPEATS = 5


def drop_page_furniture(lines: list[str]) -> list[str]:
    """
    Remove running headers/footers ("Table of Contents",
    "Apple Inc. | 2025 Form 10-K | 6") — short lines that repeat many times
    once page numbers/digits are ignored — and collapse consecutive repeats.
    """
    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip())

    counts = Counter(norm(l) for l in lines if len(l.strip()) <= FURNITURE_MAX_CHARS)
    furniture = {k for k, n in counts.items() if n >= FURNITURE_MIN_REPEATS}

    kept = []
    for line in lines:
        if len(line.strip()) <= FURNITURE_MAX_CHARS and norm(line) in furniture:
            continue
        if kept and line == kept[-1]:
            continue
        kept.append(line)
    return kept


def save_doc(ticker: str, filing: dict, url: str, text: str) -> None:
    if len(text) < 500:
        print(f"  [SKIP] extracted text too short ({len(text)} chars): {url}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{ticker}-{filing['form'].replace('-', '')}-{filing['filingDate']}"
    txt_path = OUT_DIR / f"tier2_secedgar_{slug}.txt"
    meta_path = OUT_DIR / f"tier2_secedgar_{slug}.meta.json"

    txt_path.write_text(text, encoding="utf-8")

    metadata = {
        "source": "sec_edgar",
        "tier": 2,
        "doc_type": "sec_filing",
        "title": f"{ticker} {filing['form']} filed {filing['filingDate']}",
        "url": url,
        "date_fetched": date.today().isoformat(),
        "date_published": filing["filingDate"],
        "company_ticker": ticker,
        "arxiv_id": None,
        "char_count": len(text),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  [SAVED] {txt_path.name} ({len(text)} chars)")


def run(single_ticker: str | None = None) -> None:
    manifest = load_manifest()
    companies = manifest["tier2_filings"]["companies"]

    if single_ticker:
        companies = [c for c in companies if c["ticker"].upper() == single_ticker.upper()]
        if not companies:
            print(f"Ticker {single_ticker} not found in manifest.")
            return

    print(f"=== Tier 2 scrape: {len(companies)} companies ===")
    for company in companies:
        ticker, cik = company["ticker"], company["cik"]
        wanted_forms = company["filings"]
        print(f"\n{ticker} (CIK {cik})")

        filings = get_target_filings(cik, wanted_forms)
        time.sleep(REQUEST_DELAY_SEC)

        for filing in filings:
            doc_url = build_document_url(filing)
            print(f"  Fetching {filing['form']} ({filing['filingDate']}): {doc_url}")
            html = fetch_html(doc_url)
            if html is None:
                continue
            text = extract_filing_text(html)
            save_doc(ticker, filing, doc_url, text)
            time.sleep(REQUEST_DELAY_SEC)


def reextract_existing() -> None:
    """
    Re-fetch the exact filings already saved (by the url in each .meta.json)
    and re-run text extraction — picks up extractor fixes without changing
    which filings are in the corpus.
    """
    metas = sorted(p for p in OUT_DIR.glob("*.meta.json") if not p.name.startswith("._"))
    print(f"=== Tier 2 re-extract: {len(metas)} filings ===")
    for meta_path in metas:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        html = fetch_html(metadata["url"])
        if html is None:
            continue
        text = extract_filing_text(html)
        txt_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".txt"))
        old_chars = metadata.get("char_count", 0)
        txt_path.write_text(text, encoding="utf-8")
        metadata["char_count"] = len(text)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"  [RE-EXTRACTED] {txt_path.name}: {old_chars} -> {len(text)} chars")
        time.sleep(REQUEST_DELAY_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None, help="Scrape a single company by ticker")
    parser.add_argument("--reextract", action="store_true",
                        help="Re-fetch and re-extract the filings already in data/raw/tier2")
    args = parser.parse_args()
    if args.reextract:
        reextract_existing()
    else:
        run(single_ticker=args.ticker)
