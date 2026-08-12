"""
Day 5: Metadata enrichment.

Reads data/processed/chunks.jsonl (from Day 4) and adds:
  - contains_table / contains_formula flags
  - normalized_date (ISO 8601, best-effort)
  - topic_tags (cheap keyword-based tagging per tier vocabulary)
  - content_hash (for near-duplicate detection before embedding)

Writes data/processed/chunks_enriched.jsonl and prints a coverage report.

Usage:
    python scripts/enrich_metadata.py
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
IN_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "chunks_enriched.jsonl"


# ---------- 1. Table / formula detection ----------
# Heuristics operate on the raw chunk text. Tables (from Day 2/3 extraction)
# show up as a run of alternating words/numbers with no sentence punctuation;
# formulas show up as short spans with math operators and no stopwords.

NUMERIC_TOKEN_RE = re.compile(r"^-?\$?\d[\d,]*\.?\d*%?$")
FORMULA_OPERATOR_RE = re.compile(r"[=*/^]|(?<!\w)-(?!\w)")
FORMULA_CONTEXT_RE = re.compile(r"\b[A-Za-z]\s*=|\(\d|\)\d|\d\)")


def contains_table(text: str) -> bool:
    for line in text.split("\n"):
        tokens = line.split()
        if len(tokens) < 6:
            continue
        numeric_ratio = sum(1 for t in tokens if NUMERIC_TOKEN_RE.match(t)) / len(tokens)
        if numeric_ratio > 0.35:  # a real sentence almost never has this many bare numbers
            return True
    return False


def contains_formula(text: str) -> bool:
    if FORMULA_OPERATOR_RE.search(text) and FORMULA_CONTEXT_RE.search(text):
        return True
    # catches patterns like "P * r * (1+r)^n"
    if re.search(r"\(\s*1\s*[+\-]\s*\w+\s*\)\s*\^", text):
        return True
    return False


# ---------- 2. Date normalization ----------

DATE_PATTERNS = [
    "%Y-%m-%d",       # 2024-01-01 (SEC / our own metadata already use this)
    "%Y-%m-%dT%H:%M:%S%z",
    "%B %d, %Y",      # January 1, 2024 (common on explainer sites)
    "%b %d, %Y",      # Jan 1, 2024
    "%m/%d/%Y",
]


def normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in DATE_PATTERNS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    # last resort: pull a bare YYYY-MM-DD out of a messier string
    match = re.search(r"\d{4}-\d{2}-\d{2}", raw)
    return match.group(0) if match else None


# ---------- 3. Topic tagging (keyword-based, per tier) ----------

TOPIC_KEYWORDS = {
    # tier 1
    "emi": ["emi", "equated monthly installment", "amortization", "loan repayment"],
    "interest": ["compound interest", "simple interest", "interest rate"],
    "mutual_funds": ["mutual fund", "nav", "expense ratio", "sip", "systematic investment"],
    "bonds": ["bond", "yield", "duration", "coupon"],
    "derivatives": ["option", "future", "derivative", "call option", "put option"],
    # tier 2
    "risk_factors": ["risk factor", "risks and uncertainties"],
    "financials": ["revenue", "gross profit", "net income", "balance sheet", "cash flow"],
    "md_and_a": ["management discussion", "results of operations"],
    # tier 3
    "market_microstructure": ["market microstructure", "limit order book", "bid-ask spread", "order flow"],
    "hft": ["high frequency trading", "high-frequency trading", "latency arbitrage", "market making"],
    "execution": ["optimal execution", "price impact", "transaction cost"],
}


def get_topic_tags(text: str) -> list[str]:
    text_lower = text.lower()
    tags = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(topic)
    return tags


# ---------- 4. Content hash (dedup) ----------

def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ---------- 5. Orchestration ----------

def enrich_chunk(chunk: dict) -> dict:
    text = chunk["text"]
    chunk["contains_table"] = contains_table(text)
    chunk["contains_formula"] = contains_formula(text)
    chunk["normalized_date"] = normalize_date(chunk.get("date_published"))
    chunk["topic_tags"] = get_topic_tags(text)
    chunk["content_hash"] = content_hash(text)
    return chunk


def run() -> None:
    if not IN_PATH.exists():
        print(f"[ERROR] {IN_PATH} not found — run scripts/chunk_documents.py first (Day 4).")
        return

    chunks = [json.loads(line) for line in open(IN_PATH, encoding="utf-8")]

    enriched = [enrich_chunk(c) for c in chunks]

    # dedup report (don't auto-drop — flag duplicates so you can inspect before embedding)
    hash_counts = Counter(c["content_hash"] for c in enriched)
    duplicate_hashes = {h for h, n in hash_counts.items() if n > 1}
    for c in enriched:
        c["is_duplicate"] = c["content_hash"] in duplicate_hashes

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in enriched:
            f.write(json.dumps(c) + "\n")

    # ---- coverage report ----
    n = len(enriched)
    table_count = sum(c["contains_table"] for c in enriched)
    formula_count = sum(c["contains_formula"] for c in enriched)
    date_count = sum(c["normalized_date"] is not None for c in enriched)
    tagged_count = sum(len(c["topic_tags"]) > 0 for c in enriched)
    dup_count = sum(c["is_duplicate"] for c in enriched)

    tag_freq = Counter(tag for c in enriched for tag in c["topic_tags"])
    tier_counts = Counter(c["tier"] for c in enriched)

    print("=== Enrichment Report ===")
    print(f"Total chunks           : {n}")
    print(f"  by tier              : {dict(sorted(tier_counts.items()))}")
    print(f"Contains table         : {table_count} ({table_count/n:.1%})")
    print(f"Contains formula       : {formula_count} ({formula_count/n:.1%})")
    print(f"Has normalized date    : {date_count} ({date_count/n:.1%})")
    print(f"Has >=1 topic tag      : {tagged_count} ({tagged_count/n:.1%})")
    print(f"Flagged as duplicate   : {dup_count}")
    print(f"Topic tag frequency    : {dict(tag_freq.most_common())}")
    print(f"\nOutput: {OUT_PATH}")

    if tagged_count / n < 0.5:
        print("\nWARNING: <50% of chunks got a topic tag — consider expanding "
              "TOPIC_KEYWORDS once you see real corpus content, especially for Tier 3.")


if __name__ == "__main__":
    run()