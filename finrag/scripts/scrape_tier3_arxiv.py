"""
Day 3: Tier 3 (market microstructure / HFT) scraper.

For each search query in config/corpus_manifest.yaml:
  1. Query the arXiv Atom API (export.arxiv.org/api/query)
  2. Parse entries for title/abstract/pdf link/arxiv id
  3. Download the PDF, extract text with pypdf
  4. Save as .txt + .meta.json (same schema as Tier 1/2)

Usage:
    python scripts/scrape_tier3_arxiv.py
    python scripts/scrape_tier3_arxiv.py --query "limit order book"   # single query, for testing
"""
import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

import requests
import yaml
# pyrefly: ignore [missing-import]
from pypdf import PdfReader
import xml.etree.ElementTree as ET
from io import BytesIO

ROOT = Path(__file__).parent.parent
MANIFEST_PATH = ROOT / "config" / "corpus_manifest.yaml"
OUT_DIR = ROOT / "data" / "raw" / "tier3"

CONTACT_EMAIL = "dhritismansarma@gmail.com"  # <-- EDIT THIS before running
HEADERS = {"User-Agent": f"FinRAG-Coach-Bot/0.1 ({CONTACT_EMAIL})"}

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
REQUEST_DELAY_SEC = 3  # arXiv asks for a delay of a few seconds between requests


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def search_arxiv(query: str, max_results: int, categories: list[str]) -> list[dict]:
    """
    Query arXiv's Atom API and return parsed entries (title, abstract, pdf_url, arxiv_id, published).
    """
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    search_query = f"({cat_filter}) AND abs:{query}"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query"

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [SEARCH FAILED] {query} -> {e}")
        return []

    return parse_atom_feed(resp.text)


def parse_atom_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id_full = entry.find("atom:id", ATOM_NS).text.strip()
        arxiv_id = arxiv_id_full.rstrip("/").split("/abs/")[-1]

        title = entry.find("atom:title", ATOM_NS).text.strip()
        title = re.sub(r"\s+", " ", title)

        abstract = entry.find("atom:summary", ATOM_NS).text.strip()
        abstract = re.sub(r"\s+", " ", abstract)

        published = entry.find("atom:published", ATOM_NS).text[:10]  # YYYY-MM-DD

        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib["href"]
                break
        if pdf_url is None:
            # fallback: construct from abs URL
            pdf_url = arxiv_id_full.replace("/abs/", "/pdf/")

        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "published": published,
            "pdf_url": pdf_url,
        })
    return entries


def download_and_extract_pdf_text(pdf_url: str) -> str:
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [PDF FETCH FAILED] {pdf_url} -> {e}")
        return ""

    try:
        reader = PdfReader(BytesIO(resp.content))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages_text).strip()
    except Exception as e:
        print(f"  [PDF PARSE FAILED] {pdf_url} -> {e}")
        return ""


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60]


def save_doc(entry: dict, full_text: str) -> None:
    # Prefer full PDF text; fall back to abstract only if PDF extraction failed
    text = full_text if len(full_text) > 500 else entry["abstract"]
    if len(text) < 200:
        print(f"  [SKIP] no usable text for {entry['arxiv_id']}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{entry['arxiv_id'].replace('.', '-').replace('/', '-')}_{slugify(entry['title'])}"
    txt_path = OUT_DIR / f"tier3_arxiv_{slug}.txt"
    meta_path = OUT_DIR / f"tier3_arxiv_{slug}.meta.json"

    # Prepend title + abstract so they're always present even if body extraction is noisy
    content = f"## {entry['title']}\n\nAbstract: {entry['abstract']}\n\n{text}"
    txt_path.write_text(content, encoding="utf-8")

    metadata = {
        "source": "arxiv",
        "tier": 3,
        "doc_type": "research_paper",
        "title": entry["title"],
        "url": f"https://arxiv.org/abs/{entry['arxiv_id']}",
        "date_fetched": date.today().isoformat(),
        "date_published": entry["published"],
        "company_ticker": None,
        "arxiv_id": entry["arxiv_id"],
        "char_count": len(content),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  [SAVED] {txt_path.name} ({len(content)} chars)")


def run(single_query: str | None = None) -> None:
    manifest = load_manifest()
    tier3 = manifest["tier3_microstructure"]
    categories = tier3["source"]["categories"]
    max_results = tier3["max_results_per_query"]
    queries = tier3["search_queries"]

    if single_query:
        queries = [q for q in queries if single_query.lower() in q.lower()]
        if not queries:
            print(f"No query matching '{single_query}' found in manifest.")
            return

    seen_ids = set()
    print(f"=== Tier 3 scrape: {len(queries)} search queries ===")
    for query in queries:
        print(f"\nSearching: '{query}'")
        entries = search_arxiv(query, max_results, categories)
        time.sleep(REQUEST_DELAY_SEC)

        for entry in entries:
            if entry["arxiv_id"] in seen_ids:
                continue  # dedupe across overlapping queries
            seen_ids.add(entry["arxiv_id"])

            print(f"  Downloading: {entry['title'][:70]}...")
            full_text = download_and_extract_pdf_text(entry["pdf_url"])
            save_doc(entry, full_text)
            time.sleep(REQUEST_DELAY_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Run a single search query (substring match)")
    args = parser.parse_args()
    run(single_query=args.query)
