"""
citations.py -- Source citation formatter for finrag.

Converts raw ChromaDB metadata dicts (as returned by collection.query()) into
clean, human-readable citation strings and structured citation objects.

The three tiers each have their own natural citation style:
  Tier 1  -- web article  ->  "Title" -- Source  (URL)
  Tier 2  -- SEC filing   ->  Company TICKER: Filing Type (Date)  (URL)
  Tier 3  -- arXiv paper  ->  "Title" arXiv:XXXX.XXXXX  (URL)

Usage (from rag_loop.py or any other script):
    from citations import format_citation, format_citations_block
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 1. Dataclass -- one citation per retrieved chunk
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """Structured representation of one retrieved source chunk."""
    rank: int                        # 1-based position in the result list
    tier: int                        # corpus tier (1, 2, or 3)
    source: str                      # short source label, e.g. "investopedia"
    doc_title: str                   # human-readable document title
    section_heading: str             # sub-section the chunk came from
    url: str                         # original URL (may be empty)
    distance: float                  # cosine distance (lower = more relevant)
    text_preview: str                # first N chars of the chunk
    text: str = ""                   # full chunk text (what the LLM sees)
    chunk_id: str = ""               # unique per chunk -- the dedup key
    company_ticker: str = ""         # Tier 2 only
    arxiv_id: str = ""               # Tier 3 only
    date_published: str = ""         # ISO date or empty

    # Raw metadata kept for downstream consumers (e.g. eval scripts)
    raw_metadata: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# 2. Build a Citation from a ChromaDB result row
# ---------------------------------------------------------------------------

def citation_from_chroma(
    rank: int,
    document: str,
    metadata: dict[str, Any],
    distance: float,
    preview_chars: int = 200,
    chunk_id: str = "",
) -> Citation:
    """
    Convert one row from a chromadb collection.query() result into a Citation.

    Args:
        rank:          1-based result index.
        document:      The chunk text returned by Chroma.
        metadata:      The metadata dict for this chunk.
        distance:      Cosine distance from the query embedding.
        preview_chars: How many chars to keep for text_preview.
        chunk_id:      The Chroma id for this chunk.
    """
    return Citation(
        rank=rank,
        tier=int(metadata.get("tier", 0)),
        source=metadata.get("source", ""),
        doc_title=metadata.get("doc_title", metadata.get("title", "")),
        section_heading=metadata.get("section_heading", ""),
        url=metadata.get("url", ""),
        distance=distance,
        text_preview=document[:preview_chars].replace("\n", " "),
        text=document,
        chunk_id=chunk_id or metadata.get("chunk_id", ""),
        company_ticker=metadata.get("company_ticker", "") or "",
        arxiv_id=metadata.get("arxiv_id", "") or "",
        date_published=metadata.get("date_published", "") or "",
        raw_metadata=metadata,
    )


def citations_from_chroma_results(
    results: dict[str, Any],
    preview_chars: int = 200,
) -> list[Citation]:
    """
    Parse the full results dict returned by collection.query() into Citations.

    results must have keys: 'ids', 'documents', 'metadatas', 'distances'
    (standard Chroma query output with n_results > 0).
    """
    citations: list[Citation] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = (results.get("ids") or [[]])[0] or [""] * len(docs)

    for rank, (doc, meta, dist, cid) in enumerate(zip(docs, metas, dists, ids), start=1):
        citations.append(citation_from_chroma(rank, doc, meta, dist, preview_chars, chunk_id=cid))

    return citations


# ---------------------------------------------------------------------------
# 3. Formatting helpers
# ---------------------------------------------------------------------------

def format_citation(c: Citation, verbose: bool = False) -> str:
    """
    Return a single-line (or multi-line if verbose) citation string.

    Tier 1 example:
        [1] "What is an EMI?" -- Investopedia  (dist=0.123)
            https://investopedia.com/...

    Tier 2 example:
        [2] AAPL: 10-K filing -- Risk Factors section  (dist=0.210)
            https://sec.gov/...

    Tier 3 example:
        [3] "Optimal Execution of Portfolio Transactions" arXiv:0006.1412  (dist=0.305)
            https://arxiv.org/abs/...
    """
    tier = c.tier
    rank_str = f"[{c.rank}]"
    dist_str = f"(dist={c.distance:.3f})"

    if tier == 1:
        title_part = f'"{c.doc_title}"' if c.doc_title else c.source
        source_part = f" -- {c.source.title()}" if c.source else ""
        line1 = f"{rank_str} {title_part}{source_part}  {dist_str}"

    elif tier == 2:
        title_part = c.doc_title or "SEC Filing"
        # SEC titles already read "AAPL 10-K filed 2025-10-31", and the chunker
        # uses the title as section_heading — don't repeat any of it.
        ticker = f"{c.company_ticker}: " if c.company_ticker and not title_part.startswith(c.company_ticker) else ""
        section_part = f" -- {c.section_heading}" if c.section_heading and c.section_heading != title_part else ""
        date_part = f" ({c.date_published})" if c.date_published and c.date_published not in title_part else ""
        line1 = f"{rank_str} {ticker}{title_part}{section_part}{date_part}  {dist_str}"

    elif tier == 3:
        title_part = f'"{c.doc_title}"' if c.doc_title else "arXiv paper"
        arxiv_part = f" arXiv:{c.arxiv_id}" if c.arxiv_id else ""
        line1 = f"{rank_str} {title_part}{arxiv_part}  {dist_str}"

    else:
        line1 = f"{rank_str} {c.doc_title or c.source}  {dist_str}"

    if not verbose:
        return line1

    lines = [line1]
    if c.section_heading and tier != 2:
        lines.append(f"    section: {c.section_heading}")
    if c.url:
        lines.append(f"    {c.url}")
    if c.text_preview:
        lines.append(f'    "{c.text_preview}..."')
    return "\n".join(lines)


def format_citations_block(
    citations: list[Citation],
    verbose: bool = False,
    header: bool = True,
) -> str:
    """
    Format a list of Citations into a printable block.

    Args:
        citations: List of Citation objects.
        verbose:   If True, include URL and text preview under each citation.
        header:    If True, prepend a 'Sources:' header line.
    """
    if not citations:
        return ""

    lines = []
    if header:
        lines.append("Sources:")
    for c in citations:
        lines.append(format_citation(c, verbose=verbose))

    return "\n".join(lines)


def deduplicate_citations(citations: list[Citation]) -> list[Citation]:
    """
    Drop citations that refer to the same chunk (same chunk_id, or same
    content_hash when no id is available), keeping the closest match (lowest
    distance). Re-ranks the survivors from 1.

    Distinct chunks from the same document/section are kept -- for SEC filings
    section_heading is the filing title, so keying on it would collapse every
    chunk of a filing into one citation.

    The returned order and ranks are final: callers must build both the LLM
    prompt's [N] labels and the printed source list from this same list.
    """
    seen: dict[str, Citation] = {}
    for c in citations:
        key = c.chunk_id or c.raw_metadata.get("content_hash") or c.text
        if key not in seen or c.distance < seen[key].distance:
            seen[key] = c

    deduped = sorted(seen.values(), key=lambda x: x.distance)
    for i, c in enumerate(deduped, start=1):
        c.rank = i
    return deduped
