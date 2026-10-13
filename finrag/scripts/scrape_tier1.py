# pyrefly: ignore [missing-import]
"""
Day 2: Tier 1 (fundamentals) scraper.

Fetches finance explainer pages listed/implied by config/corpus_manifest.yaml,
extracts clean main-content text (stripping nav/ads/boilerplate), and saves:
  - data/raw/tier1/{slug}.txt          (clean article text)
  - data/raw/tier1/{slug}.meta.json    (metadata matching the schema from Day 1)

Usage:
    python scripts/scrape_tier1.py
    python scripts/scrape_tier1.py --topic "EMI"   # scrape a single topic for testing
"""
import argparse
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
MANIFEST_PATH = ROOT / "config" / "corpus_manifest.yaml"
OUT_DIR = ROOT / "data" / "raw" / "tier1"

HEADERS = {
    # A real, descriptive UA is good scraping etiquette and avoids blanket blocks
    "User-Agent": "FinRAG-Coach-Bot/0.1 (educational project; contact: you@example.com)"
}

REQUEST_DELAY_SEC = 1.5  # be polite, avoid hammering source sites


# ---------- 1. Source-specific URL resolution ----------
# Each source needs a way to go from a "topic" string -> a real URL.
# For Investopedia/Khan Academy this is a predictable slug pattern or search.
# For RBI/SEBI, PDFs are typically linked from a fixed index page, so those
# need a small allowlist rather than guessed URLs.

INVESTOPEDIA_SLUG_MAP = {
    "EMI (Equated Monthly Installment) / loan amortization": "amortization",
    "compound interest": "compoundinterest",
    "simple interest": "simple-interest",
    "mutual funds basics": "mutualfund",
    "SIP (systematic investment plan)": "systematicinvestmentplan",
    "bonds and yield": "bond",
    "bond duration": "duration",
    "stocks basics": "stock",
    "options basics (calls/puts)": "option",
    "futures basics": "futures",
    "derivatives overview": "derivative",
    "diversification": "diversification",
    "expense ratio": "expenseratio",
    "net asset value (NAV)": "nav",
    "credit score basics": "credit_score",
    "dollar-cost averaging": "dollarcostaveraging",
    "annual percentage rate (APR)": "apr",
    "annual percentage yield (APY)": "apy",
    "inflation": "inflation",
    "time value of money": "timevalueofmoney",
    "present value": "presentvalue",
    "net present value (NPV)": "npv",
    "dividends": "dividend",
    "price-to-earnings (P/E) ratio": "price-earningsratio",
    "market capitalization": "marketcapitalization",
    "index funds": "indexfund",
    "exchange-traded funds (ETF)": "etf",
    "asset allocation": "assetallocation",
    "risk tolerance": "risktolerance",
    "mortgage basics": "mortgage",
    "call options": "calloption",
    "put options": "putoption",
    "yield to maturity": "yieldtomaturity",
    "capital gains": "capitalgain",
    "return on investment (ROI)": "returnoninvestment",
    "volatility": "volatility",
    "compound annual growth rate (CAGR)": "cagr",
}


def resolve_investopedia_url(topic: str) -> str | None:
    slug = INVESTOPEDIA_SLUG_MAP.get(topic)
    if slug is None:
        return None
    if slug.endswith(".asp"):
        return f"https://www.investopedia.com/terms/{slug}"
    return f"https://www.investopedia.com/terms/{slug[0]}/{slug}.asp"


# ---------- 2. Fetch ----------

def fetch_html(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  [FETCH FAILED] {url} -> {e}")
        return None


# ---------- 3. Clean main-content extraction ----------
# Generic heuristic: prefer <article>/<main>, else score <div> blocks by
# (paragraph count, total text length) and take the highest scorer.
# This generalizes across sites without a bespoke parser per source.

BOILERPLATE_TAGS = ["nav", "header", "footer", "aside", "script", "style", "form", "iframe"]
BOILERPLATE_CLASS_HINTS = re.compile(
    r"(nav|menu|footer|header|sidebar|ad-|advert|cookie|newsletter|social|comment|related)",
    re.I,
)


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Pick the content container FIRST, then strip boilerplate only inside it.
    # Stripping the whole document first deletes the article on sites like
    # Investopedia, where <main>/<article> are nested inside a <header> element
    # and the <article> itself carries an "sc-ad-container" class.
    candidate = (
        soup.find(id=re.compile(r"^article-body"))
        or soup.find(class_="article-body")
        or soup.find("article")
        or soup.find("main")
    )

    if candidate is None:
        # fallback: score all divs by paragraph density
        best_div, best_score = None, 0
        for div in soup.find_all("div"):
            paras = div.find_all("p", recursive=False)
            text_len = sum(len(p.get_text(strip=True)) for p in paras)
            score = len(paras) * text_len
            if score > best_score:
                best_div, best_score = div, score
        candidate = best_div or soup.body

    if candidate is None:
        return ""

    for tag in candidate(BOILERPLATE_TAGS):
        tag.decompose()
    for tag in candidate.find_all(class_=BOILERPLATE_CLASS_HINTS):
        tag.decompose()
    for tag in candidate.find_all(id=BOILERPLATE_CLASS_HINTS):
        tag.decompose()

    # Keep paragraph/heading/list structure as plain text with blank-line separation
    # so Day 4's section-aware chunker has something to work with.
    lines = []
    for el in candidate.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 3:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append(f"\n## {text}\n")
        else:
            lines.append(text)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ---------- 4. Save with metadata ----------

def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60]


def save_doc(source: str, topic: str, url: str, text: str) -> None:
    if len(text) < 200:
        print(f"  [SKIP] extracted text too short ({len(text)} chars) — likely bad parse: {url}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic) or hashlib.md5(url.encode()).hexdigest()[:10]
    txt_path = OUT_DIR / f"tier1_{source}_{slug}.txt"
    meta_path = OUT_DIR / f"tier1_{source}_{slug}.meta.json"

    txt_path.write_text(text, encoding="utf-8")

    metadata = {
        "source": source,
        "tier": 1,
        "doc_type": "explainer",
        "title": topic,
        "url": url,
        "date_fetched": date.today().isoformat(),
        "date_published": None,
        "company_ticker": None,
        "arxiv_id": None,
        "char_count": len(text),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  [SAVED] {txt_path.name} ({len(text)} chars)")


# ---------- 5. Orchestration ----------

def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def run(single_topic: str | None = None) -> None:
    manifest = load_manifest()
    topics = manifest["tier1_fundamentals"]["sources"][0]["topics"]  # investopedia entry

    if single_topic:
        topics = [t for t in topics if single_topic.lower() in t.lower()]
        if not topics:
            print(f"No topic matching '{single_topic}' found in manifest.")
            return

    print(f"=== Tier 1 scrape: {len(topics)} Investopedia topics ===")
    for topic in topics:
        url = resolve_investopedia_url(topic)
        if url is None:
            print(f"  [NO URL MAPPING] {topic} — add to INVESTOPEDIA_SLUG_MAP")
            continue

        print(f"Fetching: {topic}\n  URL: {url}")
        html = fetch_html(url)
        if html is None:
            continue

        text = extract_main_text(html)
        save_doc("investopedia", topic, url, text)
        time.sleep(REQUEST_DELAY_SEC)

    print("\nNOTE: RBI/SEBI/Khan Academy sources are PDF/curriculum-index based, "
          "not clean slug-per-topic — handle those as a small manual allowlist "
          "of direct PDF/page URLs added to this script if you want them included; "
          "Investopedia alone can comfortably cover the 35-doc Tier 1 target.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default=None, help="Scrape a single topic (substring match)")
    args = parser.parse_args()
    run(single_topic=args.topic)