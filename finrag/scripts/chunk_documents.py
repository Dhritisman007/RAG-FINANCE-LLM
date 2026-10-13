"""
Day 4: Structure-aware chunking.

Reads every {source}.txt + {source}.meta.json pair from data/raw/tier{1,2,3}/,
splits each doc by its "## Heading" section markers, then packs the paragraphs/
list-items/table-blocks within each section into token-budgeted chunks without
ever slicing a single line (paragraph, list item, or table blob) in half.

Output: one chunk per line, JSONL format, at data/processed/chunks.jsonl
Usage:
    python scripts/chunk_documents.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

# --- Chunk sizing config ---
MAX_CHUNK_TOKENS = 450   # upper bound before we start a new chunk
MIN_CHUNK_TOKENS = 40    # chunks smaller than this get merged into a neighbor
# Token counting: approximate via chars/4 (roughly matches cl100k_base for
# English prose). Good enough for sizing decisions; swap in tiktoken later
# if you want exact counts and have network access to fetch its BPE file.


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------- 1. Split doc into (heading, body_units) sections ----------

HEADING_RE = re.compile(r"^##\s+(.+)$")


def split_into_sections(text: str, doc_title: str) -> list[dict]:
    """
    Returns a list of {"heading": str, "units": [line, line, ...]}.
    A "unit" is one atomic line from the source .txt — a paragraph, list item,
    or a whole flattened table — as produced by the Day 2/3 extractors.
    Content before the first '## ' heading is grouped under the doc title.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    sections = []
    current_heading = doc_title
    current_units: list[str] = []

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            if current_units:
                sections.append({"heading": current_heading, "units": current_units})
            current_heading = match.group(1)
            current_units = []
        else:
            current_units.append(line)

    if current_units:
        sections.append({"heading": current_heading, "units": current_units})

    return sections


# ---------- 2. Pack units into token-budgeted chunks ----------

def split_oversized_unit(unit: str, max_tokens: int) -> list[str]:
    """
    Fallback for a single unit (e.g. a huge paragraph) that alone exceeds
    max_tokens. Splits on sentence boundaries — never mid-sentence, and
    only invoked as a last resort (flagged via `was_sub_split` downstream).
    """
    sentences = re.split(r"(?<=[.!?])\s+", unit)
    pieces, current = [], ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip()
        if count_tokens(candidate) > max_tokens and current:
            pieces.append(current)
            current = sent
        else:
            current = candidate
    if current:
        pieces.append(current)

    # Last resort for text with no sentence boundaries (flattened tables,
    # long lists): hard-split on whitespace so no piece exceeds max_tokens.
    out = []
    for piece in pieces or [unit]:
        if count_tokens(piece) <= max_tokens:
            out.append(piece)
            continue
        words, current = piece.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if count_tokens(candidate) > max_tokens and current:
                out.append(current)
                current = word
            else:
                current = candidate
        if current:
            out.append(current)
    return out


def pack_section_into_chunks(units: list[str], max_tokens: int) -> list[dict]:
    """
    Greedily packs whole units into chunks up to max_tokens.
    Never splits a unit unless it alone exceeds max_tokens.
    Returns list of {"text": str, "was_sub_split": bool}.
    """
    chunks = []
    current_units: list[str] = []
    current_tokens = 0

    def flush():
        nonlocal current_units, current_tokens
        if current_units:
            chunks.append({"text": "\n".join(current_units), "was_sub_split": False})
            current_units = []
            current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)

        if unit_tokens > max_tokens:
            # oversized single unit (rare: huge table blob or giant paragraph)
            flush()
            for piece in split_oversized_unit(unit, max_tokens):
                chunks.append({"text": piece, "was_sub_split": True})
            continue

        if current_tokens + unit_tokens > max_tokens and current_units:
            flush()

        current_units.append(unit)
        current_tokens += unit_tokens

    flush()
    return chunks


def merge_tiny_chunks(chunks: list[dict], min_tokens: int) -> list[dict]:
    """Merge any chunk below min_tokens into the previous chunk (or next, if it's first)."""
    if not chunks:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if count_tokens(chunk["text"]) < min_tokens:
            merged[-1]["text"] += "\n" + chunk["text"]
            merged[-1]["was_sub_split"] = merged[-1]["was_sub_split"] or chunk["was_sub_split"]
        else:
            merged.append(chunk)

    # if the very first chunk ended up tiny, merge it forward into the second
    if len(merged) > 1 and count_tokens(merged[0]["text"]) < min_tokens:
        merged[1]["text"] = merged[0]["text"] + "\n" + merged[1]["text"]
        merged.pop(0)

    return merged


# ---------- 3. Orchestration: doc -> chunks with metadata ----------

def chunk_document(text: str, metadata: dict) -> list[dict]:
    sections = split_into_sections(text, metadata["title"])

    all_chunks = []
    chunk_index = 0
    for section in sections:
        raw_chunks = pack_section_into_chunks(section["units"], MAX_CHUNK_TOKENS)
        raw_chunks = merge_tiny_chunks(raw_chunks, MIN_CHUNK_TOKENS)

        for c in raw_chunks:
            # Key on something unique per *document* (not company_ticker — a
            # ticker has a 10-K and several 10-Qs, which would collide), and
            # truncate before appending the index so it can never be cut off.
            doc_key = f"{metadata['source']}_{metadata.get('arxiv_id') or metadata['title']}"
            doc_key = re.sub(r"[^a-zA-Z0-9_]+", "-", doc_key)[:90]
            chunk_id = f"{doc_key}_{chunk_index}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": c["text"],
                "section_heading": section["heading"],
                "chunk_index_in_doc": chunk_index,
                "token_count": count_tokens(c["text"]),
                "was_sub_split": c["was_sub_split"],
                # inherited doc-level metadata (flat, for easy filtering at retrieval time)
                "source": metadata["source"],
                "tier": metadata["tier"],
                "doc_type": metadata["doc_type"],
                "doc_title": metadata["title"],
                "url": metadata["url"],
                "date_published": metadata.get("date_published"),
                "company_ticker": metadata.get("company_ticker"),
                "arxiv_id": metadata.get("arxiv_id"),
            })
            chunk_index += 1

    return all_chunks


def run() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_docs, total_chunks, sub_split_count = 0, 0, 0

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for tier_dir in sorted(RAW_DIR.glob("tier*")):
            for txt_path in sorted(tier_dir.glob("*.txt")):
                meta_path = txt_path.with_suffix("").with_suffix(".meta.json")
                if not meta_path.exists():
                    print(f"  [SKIP] no metadata for {txt_path.name}")
                    continue

                text = txt_path.read_text(encoding="utf-8", errors="replace")
                meta_text = meta_path.read_text(encoding="utf-8", errors="replace")
                if not meta_text.strip():
                    print(f"  [SKIP] empty metadata for {txt_path.name}")
                    continue
                try:
                    metadata = json.loads(meta_text)
                except json.JSONDecodeError as e:
                    print(f"  [SKIP] invalid JSON in metadata for {txt_path.name}: {e}")
                    continue

                chunks = chunk_document(text, metadata)
                for chunk in chunks:
                    out_f.write(json.dumps(chunk) + "\n")
                    if chunk["was_sub_split"]:
                        sub_split_count += 1

                total_docs += 1
                total_chunks += len(chunks)
                print(f"  {txt_path.name}: {len(chunks)} chunks")

    print(f"\n=== Done ===")
    print(f"Docs processed : {total_docs}")
    print(f"Chunks written : {total_chunks}")
    print(f"Sub-split units: {sub_split_count} (paragraphs too large for one chunk — worth spot-checking)")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    run()