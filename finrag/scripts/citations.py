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

import re
from dataclasses import dataclass, field
from typing import Any, Callable


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


def _shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def suppress_near_duplicates(
    citations: list[Citation],
    k: int,
    threshold: float = 0.8,
    max_per_doc: int | None = None,
) -> list[Citation]:
    """
    Greedily keep the closest citations, skipping any whose text is a near-copy
    (5-word-shingle Jaccard >= threshold) of one already kept — e.g. a 10-Q
    restating a 10-K risk factor. Returns at most k, re-ranked from 1.

    Pass more than k candidates (over-fetch) so skipped slots get backfilled.
    max_per_doc: if set, prefer at most this many passages per document, so a
    broad question draws on several sources instead of one paper. It's a soft
    cap: if fewer than k passages pass it (a narrow question only one paper
    covers), the skipped passages backfill the remaining slots.
    """
    kept: list[Citation] = []
    kept_shingles: list[set[str]] = []
    per_doc: dict[str, int] = {}
    over_cap: list[Citation] = []
    for c in sorted(citations, key=lambda x: x.distance):
        if len(kept) == k:
            break
        s = _shingles(c.text or c.text_preview)
        if any(len(s & o) / len(s | o) >= threshold for o in kept_shingles):
            continue
        if max_per_doc is not None and per_doc.get(c.doc_title, 0) >= max_per_doc:
            over_cap.append(c)
            continue
        kept.append(c)
        kept_shingles.append(s)
        per_doc[c.doc_title] = per_doc.get(c.doc_title, 0) + 1
    for c in over_cap:  # backfill: never return fewer than k because of the cap
        if len(kept) == k:
            break
        s = _shingles(c.text or c.text_preview)
        if not any(len(s & o) / len(s | o) >= threshold for o in kept_shingles):
            kept.append(c)
            kept_shingles.append(s)
    kept.sort(key=lambda x: x.distance)
    for i, c in enumerate(kept, start=1):
        c.rank = i
    return kept


# ---------------------------------------------------------------------------
# 4. Citation checking -- do the [N] markers in an answer match their sources?
# ---------------------------------------------------------------------------
# Each marker covers the answer text between the previous marker (or the
# paragraph / list-item start) and itself, so both "claim [2]." and a
# paragraph ending "(Source: [4])" are handled. Support for a claim = best
# cosine similarity between the claim and any sentence of a passage.
#
# Thresholds were calibrated on llama3.2 answers over this corpus, where they
# flagged every hand-verified mis-citation (e.g. swapped [4]/[5]) and no
# near-ties (0.81 vs 0.83 is two passages saying the same thing).

MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
MISCITE_MARGIN = 0.05     # another passage must beat the cited one by this much...
MISCITE_MIN_BEST = 0.80   # ...and be a strong match itself
WEAK_SUPPORT = 0.70       # best cited support below this = claim not clearly in its sources


@dataclass
class ClaimCheck:
    text: str                 # the claim the marker(s) cover
    cited: list[int]          # marker numbers, e.g. [2] or [2, 4]
    status: str               # "ok" | "miscited" | "weak" | "invalid"
    support: float = 0.0      # best support among the cited passages
    suggested: int | None = None   # better-matching passage, for "miscited"
    suggested_support: float = 0.0


@dataclass
class CitationReport:
    claims: list[ClaimCheck]
    uncited_answer: bool      # answer has text but no [N] markers at all
    unused_sources: list[int]
    ungrounded_numbers: list[str] = field(default_factory=list)  # numbers found in no source

    @property
    def problems(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.status != "ok"]


LEAD_IN_RE = re.compile(
    r"(?:as\s+(?:stated|mentioned|noted|discussed|described|shown|explained|outlined)\s+in"
    r"|according\s+to|based\s+on|per|see|from)\s*(?:passage|source)?s?\s*$",
    re.I,
)


def _clean_claim(text: str) -> str:
    text = MARKER_RE.sub("", text)
    text = re.sub(r"\(?\s*(?:Sources?|See)\s*:?\s*$", "", text, flags=re.I).strip(" ,;()\n")
    return text.lstrip(".!?,; ")  # punctuation left over from the previous marker


def split_cited_claims(answer: str) -> list[tuple[str, list[int]]]:
    """
    -> [(claim_text, [marker numbers]), ...].

    A marker normally covers the text before it ("claim [2]."). A lead-in
    marker ("As stated in [2], claim..." / "[2] notes that claim...") covers
    the text after it, to the end of that sentence. Short lines that start
    with a marker ("[1] JPM 10-K filed ...") are a model-written reference
    list, not claims. Claims never span lines.
    """
    claims: list[tuple[str, list[int]]] = []
    for unit in answer.split("\n"):
        if re.match(r"\s*\[\d+\]", unit) and len(MARKER_RE.sub("", unit).strip()) < 60:
            continue  # reference-list line, e.g. "[1] AAPL 10-K filed 2025-10-31"
        markers = list(MARKER_RE.finditer(unit))
        last, unit_start = 0, len(claims)
        for i, m in enumerate(markers):
            if m.start() < last:
                continue  # already consumed as part of a lead-in claim
            nums = sorted({int(n) for n in m.group(1).split(",")})
            raw_before = unit[last:m.start()]
            before = _clean_claim(raw_before)
            next_start = markers[i + 1].start() if i + 1 < len(markers) else len(unit)
            sentence_end = re.search(r"[.!?](?=\s|$)", unit[m.end():next_start])
            after_end = m.end() + sentence_end.end() if sentence_end else next_start
            after = _clean_claim(unit[m.end():after_end])
            lead_in = LEAD_IN_RE.search(raw_before) or (len(before) < 15 and len(claims) == unit_start)
            if lead_in and len(after) >= 15:
                label = _clean_claim(LEAD_IN_RE.sub("", raw_before))
                claims.append((f"{label} {after}".strip(), nums))
                last = after_end
                continue
            if len(before) < 15 and len(claims) > unit_start:
                # "[1], [2]" or "[1][3]" runs belong to the claim before them
                prev_text, prev_nums = claims[-1]
                claims[-1] = (prev_text, sorted(set(prev_nums) | set(nums)))
            elif len(before) >= 15:
                claims.append((before, nums))
            last = m.end()
    return claims


# A number in an answer: "$4,944.29", "3.5%", "139,996", "2026", "0.58".
NUMBER_RE = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")


def _norm_number(token: str) -> str:
    return token.replace("$", "").replace(",", "").rstrip("%").rstrip(".")


def find_ungrounded_numbers(answer: str, sources: list[str]) -> list[str]:
    """
    Numbers in `answer` that appear in none of `sources` (passages, the
    question, the user's positions). Semantic citation scores can't see a
    made-up "$52.50" inside an otherwise well-supported sentence; this can.
    Ignored: [N] markers, list numbering ("2."), and bare integers below 10.
    It can't catch a real number attached to the wrong label (e.g. an expense
    line quoted as revenue) — that needs the reader to check the source.
    """
    text = MARKER_RE.sub(" ", answer)
    text = re.sub(r"^\s*\d+[.)]\s", " ", text, flags=re.M)  # list numbering
    known = {_norm_number(m.group(0)) for s in sources for m in NUMBER_RE.finditer(s)}
    missing = []
    for m in NUMBER_RE.finditer(text):
        tok = m.group(0).rstrip(".,")
        n = _norm_number(tok)
        if not n or ("." not in n and "%" not in tok and "$" not in tok and int(n) < 10):
            continue
        if n not in known and tok not in missing:
            missing.append(tok)
    return missing


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _quoted_in(claim: str, c: Citation) -> bool:
    """Claim (>= 3 words) appears verbatim in the passage. Short claims like
    "it's interest on interest" embed poorly, so lexical containment backs
    up the semantic score before calling a claim weak."""
    words = _words(re.sub(r"^(?:in other words|that is|i\.e\.|so|also),?\s*(?:it'?s|it is)?\s*", "", claim, flags=re.I))
    return len(words.split()) >= 3 and words in _words(c.text or c.text_preview)


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text)]
    return [s for s in parts if len(s) > 20] or [text]


def check_citations(
    answer: str,
    citations: list[Citation],
    encode: Callable[[list[str]], Any],
    extra_sources: list[str] | None = None,
) -> CitationReport:
    """
    Check every [N] marker in `answer` against the passages in `citations`.

    encode: texts -> L2-normalized embedding matrix (numpy array), e.g. the
            local bge model's .encode(..., normalize_embeddings=True). Kept as
            a parameter so this module has no model dependency.
    extra_sources: other text numbers may legitimately come from (the
            question, the user's position lines).
    """
    by_rank = {c.rank: c for c in citations}
    ungrounded = find_ungrounded_numbers(answer, [c.text or c.text_preview for c in citations]
                                         + (extra_sources or []))
    claims = split_cited_claims(answer)
    if not claims:
        # no inline markers (a trailing "[1] Title" reference list doesn't count)
        return CitationReport([], uncited_answer=bool(MARKER_RE.sub("", answer).strip()),
                              unused_sources=sorted(by_rank), ungrounded_numbers=ungrounded)

    passage_vecs = {rank: encode(_sentences(c.text or c.text_preview)) for rank, c in by_rank.items()}
    claim_vecs = encode([text for text, _ in claims])

    checks, used = [], set()
    for (text, nums), vec in zip(claims, claim_vecs):
        invalid = [n for n in nums if n not in by_rank]
        used.update(n for n in nums if n in by_rank)
        if invalid:
            checks.append(ClaimCheck(text, nums, "invalid"))
            continue
        support = {rank: float((pv @ vec).max()) for rank, pv in passage_vecs.items()}
        cited_best = max(support[n] for n in nums)
        best = max(support, key=support.get)
        check = ClaimCheck(text, nums, "ok", support=cited_best)
        if best not in nums and support[best] >= MISCITE_MIN_BEST and support[best] - cited_best >= MISCITE_MARGIN:
            check.status, check.suggested, check.suggested_support = "miscited", best, support[best]
        elif cited_best < WEAK_SUPPORT and not any(_quoted_in(text, by_rank[n]) for n in nums):
            check.status = "weak"
        checks.append(check)

    return CitationReport(checks, uncited_answer=False,
                          unused_sources=sorted(set(by_rank) - used), ungrounded_numbers=ungrounded)


def format_citation_report(report: CitationReport) -> str:
    numbers = ""
    if report.ungrounded_numbers:
        numbers = ("\n  # numbers not found in any source (possibly invented): "
                   + ", ".join(report.ungrounded_numbers[:12]))
    if report.uncited_answer:
        return "Citation check: ⚠ the answer has no [N] citations — its claims can't be traced to sources." + numbers
    if not report.claims:
        return numbers.strip()

    def short(text: str) -> str:
        text = " ".join(text.split())
        return text if len(text) <= 70 else text[:67] + "..."

    n_ok = sum(c.status == "ok" for c in report.claims)
    lines = [f"Citation check: {n_ok}/{len(report.claims)} cited claims match their sources"]
    for c in report.problems:
        cited = ", ".join(f"[{n}]" for n in c.cited)
        if c.status == "invalid":
            lines.append(f"  ✗ {cited} is not one of the listed sources: \"{short(c.text)}\"")
        elif c.status == "miscited":
            lines.append(f"  ⚠ {cited} likely should be [{c.suggested}] "
                         f"(support {c.support:.2f} vs {c.suggested_support:.2f}): \"{short(c.text)}\"")
        else:
            lines.append(f"  ? {cited} weakly supports this (support {c.support:.2f}) — "
                         f"may not be in the sources: \"{short(c.text)}\"")
    return "\n".join(lines) + numbers
