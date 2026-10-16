"""
Regression tests for the Day 8 citation bugs. No Chroma / Ollama needed.

    python tests/test_citations.py
"""
import re
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from citations import (  # noqa: E402
    Citation,
    check_citations,
    format_citation_report,
    find_ungrounded_numbers,
    split_cited_claims,
    citations_from_chroma_results,
    deduplicate_citations,
    format_citations_block,
    suppress_near_duplicates,
)
from rag_loop import build_prompt  # noqa: E402

FILING = "AAPL 10-K filed 2025-10-31"


def meta(chunk_id: str, tier: int = 2, title: str = FILING) -> dict:
    return {"chunk_id": chunk_id, "tier": tier, "source": "sec_edgar", "doc_title": title,
            "section_heading": title, "company_ticker": "AAPL", "url": "", "content_hash": chunk_id}


def fake_results(rows: list[tuple[str, str, float]]) -> dict:
    """rows: (chunk_id, text, distance) in the order Chroma 'returned' them."""
    return {
        "ids": [[r[0] for r in rows]],
        "documents": [[r[1] for r in rows]],
        "metadatas": [[meta(r[0]) for r in rows]],
        "distances": [[r[2] for r in rows]],
    }


def test_same_section_chunks_not_collapsed() -> None:
    results = fake_results([
        ("aapl_10k_7", "Supply chain concentration in Asia.", 0.20),
        ("aapl_10k_8", "Foreign exchange rate risk.", 0.22),
    ])
    cites = deduplicate_citations(citations_from_chroma_results(results))
    assert len(cites) == 2, f"expected 2 distinct citations, got {len(cites)}"
    assert {c.chunk_id for c in cites} == {"aapl_10k_7", "aapl_10k_8"}
    print("PASS (a): two chunks from the same filing section stay as two citations")


def parse_prompt_blocks(prompt: str) -> dict[int, str]:
    """[N] -> chunk text, as the LLM sees it."""
    context = prompt.split("=== Context ===\n", 1)[1].split("\n\n=== Question ===", 1)[0]
    blocks = {}
    for block in context.split("\n\n"):
        m = re.match(r"\[(\d+)\] [^\n]*\n(.*)", block, re.S)
        blocks[int(m.group(1))] = m.group(2)
    return blocks


def parse_printed_sources(printed: str) -> dict[int, str]:
    """[N] -> quoted preview, as the user sees it in the printed source list."""
    out, current = {}, None
    for line in printed.splitlines():
        m = re.match(r"\[(\d+)\]", line)
        if m:
            current = int(m.group(1))
        elif current and line.strip().startswith('"'):
            out[current] = line.strip().strip('"').rstrip(".")
    return out


def test_markers_point_to_matching_source_after_reorder() -> None:
    # Out-of-order distances + a repeated chunk forces dedup to drop and re-sort.
    results = fake_results([
        ("chunk_C", "CCC text about margins.", 0.30),
        ("chunk_A", "AAA text about supply chain.", 0.10),
        ("chunk_C", "CCC text about margins.", 0.35),   # duplicate of chunk_C
        ("chunk_B", "BBB text about litigation.", 0.20),
    ])
    cites = deduplicate_citations(citations_from_chroma_results(results))
    assert [c.chunk_id for c in cites] == ["chunk_A", "chunk_B", "chunk_C"], [c.chunk_id for c in cites]

    prompt_blocks = parse_prompt_blocks(build_prompt("q?", cites))
    printed = parse_printed_sources(format_citations_block(cites, verbose=True))

    answer = "Supply chain is a risk [1]. Litigation too [2]. Margins are under pressure [3]."
    for n in map(int, re.findall(r"\[(\d+)\]", answer)):
        seen_by_llm = prompt_blocks[n]
        shown_to_user = printed[n]
        assert seen_by_llm.startswith(shown_to_user), (n, seen_by_llm, shown_to_user)
    print("PASS (b): every [N] in the answer maps to the same chunk in prompt and printed list")

    # Show the old bug would have failed this: old code paired the deduped
    # labels with docs[:len(citations)] in raw Chroma order.
    old_docs = results["documents"][0][:len(cites)]
    mismatches = [c.rank for c, d in zip(cites, old_docs) if c.text != d]
    assert mismatches, "old pairing should be misaligned on this input"
    print(f"  (old docs[:len(citations)] pairing would misalign markers {mismatches})")


def test_near_duplicates_suppressed_and_backfilled() -> None:
    # Realistic chunk length (~100 words) — a page footer is a small fraction of it.
    risk = ("The Company's operations are also subject to the risks of industrial accidents at its "
            "suppliers and contract manufacturers. While the Company's suppliers are required to "
            "maintain safe working environments and operations, an industrial accident could occur "
            "and could result in serious injuries or loss of life, disruption to the Company's "
            "business, and harm to the Company's reputation. Major public health issues, including "
            "pandemics, have adversely affected, and could in the future materially adversely affect, "
            "the Company due to their impact on the global economy and demand for consumer products; "
            "the imposition of protective public safety measures; and disruptions in the Company's "
            "operations, supply chain, and sales and distribution channels.")
    results = fake_results([
        ("aapl_10k_45", risk, 0.10),
        ("aapl_10q_44", risk + " Apple Inc. | Q3 2026 Form 10-Q | 12", 0.11),  # near-copy
        ("aapl_10k_50", "Single-source custom components can face shortages and price increases.", 0.20),
        ("aapl_10k_60", "Adverse economic conditions can reduce demand for the Company's products.", 0.30),
    ])
    cites = deduplicate_citations(citations_from_chroma_results(results))
    cites = suppress_near_duplicates(cites, k=3)
    assert [c.chunk_id for c in cites] == ["aapl_10k_45", "aapl_10k_50", "aapl_10k_60"], [c.chunk_id for c in cites]
    assert [c.rank for c in cites] == [1, 2, 3]
    print("PASS (c): near-copy 10-Q chunk skipped, next distinct chunk backfills the slot")


def bow_encode(texts: list[str]) -> np.ndarray:
    """Deterministic stand-in for the bge model: hashed bag-of-words, L2-normalized."""
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_split_cited_claims_formats() -> None:
    answer = (
        "Intro sentence about loans and payments here [1].\n\n"
        "1. Supply shortages raise component prices sharply [2].\n"
        "2. Design defects can harm the company reputation [3][1].\n\n"
        "The formula is [ ( P ( 1 + i ) n ) - P ] for compounding growth. (Source: [2])\n\n"
        "References:\n[1] AAPL 10-K filed 2025-10-31\n[2] AAPL 10-K filed 2025-10-31"
    )
    claims = split_cited_claims(answer)
    assert [nums for _, nums in claims] == [[1], [2], [1, 3], [2]], claims
    assert claims[3][0].startswith("The formula is [ ( P"), claims[3][0]  # formula brackets aren't markers
    print("PASS (d): parses [N], [a][b], '(Source: [N])', skips formulas and a model-written reference list")

    lead_ins = (
        "5. Design risks: As stated in [5], the Company's products may have manufacturing defects.\n"
        "[3] notes that adverse economic conditions can reduce demand for products.\n"
        "Margins fell sharply this quarter [1]. According to [4], tariffs raised input costs."
    )
    assert split_cited_claims(lead_ins) == [
        ("5. Design risks: the Company's products may have manufacturing defects.", [5]),
        ("notes that adverse economic conditions can reduce demand for products.", [3]),
        ("Margins fell sharply this quarter", [1]),
        ("tariffs raised input costs.", [4]),
    ], split_cited_claims(lead_ins)
    print("PASS (f): lead-in markers ('As stated in [5], claim', '[3] notes that claim') cover the text after them")


def test_check_flags_swapped_invalid_and_uncited() -> None:
    passages = {
        1: "Industry-wide shortages and commodity pricing fluctuations raise component costs.",
        2: "Design and manufacturing defects could harm the Company's reputation.",
    }
    cites = [Citation(rank=n, tier=2, source="sec_edgar", doc_title="AAPL 10-K", section_heading="", url="",
                      distance=0.1 * n, text_preview=t[:50], text=t) for n, t in passages.items()]
    answer = ("1. Industry-wide shortages and commodity pricing fluctuations raise component costs [1].\n"
              "2. Design and manufacturing defects could harm the Company's reputation [1].\n"
              "3. Shortages also affect manufacturing defects and pricing [7].")
    report = check_citations(answer, cites, bow_encode)
    status = [(c.cited, c.status, c.suggested) for c in report.claims]
    assert status == [([1], "ok", None), ([1], "miscited", 2), ([7], "invalid", None)], status
    assert report.unused_sources == [2]

    uncited = check_citations("EMIs are fixed monthly loan payments.", cites, bow_encode)
    assert uncited.uncited_answer and not uncited.claims
    assert "no [N] citations" in format_citation_report(uncited)
    print("PASS (e): flags a swapped citation with the right suggestion, an invalid [7], and an uncited answer")


def test_numbers_must_come_from_sources() -> None:
    sources = ["If the rate is 3.5% for 10 years the EMI is $4,944.29.", "(In millions) Revenue $ 139,996 $ 120,810"]
    answer = ("1. Deposit $1,000 at 5% and earn $50 [1].\n"
              "2. The EMI is $4,944.29 at 3.5% [1]. Revenue was $139,996 million in 2026 [2].")
    assert find_ungrounded_numbers(answer, sources + ["What was revenue in 2026?"]) == ["$1,000", "5%", "$50"]
    cites = [Citation(rank=1, tier=1, source="", doc_title="", section_heading="", url="", distance=0.1,
                      text_preview="", text=sources[0])]
    report = check_citations("Deposit $1,000 to earn $50.", cites, bow_encode)
    assert report.uncited_answer and report.ungrounded_numbers == ["$1,000", "$50"]
    assert "numbers not found in any source" in format_citation_report(report)
    quoted = [Citation(rank=1, tier=1, source="", doc_title="", section_heading="", url="", distance=0.1,
                       text_preview="", text='Compounding means earning interest on accumulated interest, i.e., "interest on interest."')]
    calls = iter(range(0, 256, 16))
    orthogonal = lambda xs: np.eye(256)[next(calls):][:len(xs)]  # every call gets fresh, mutually orthogonal vectors
    short = check_citations("In other words, it's interest on interest [1].", quoted, orthogonal)
    assert short.claims[0].support == 0.0 and short.claims[0].status == "ok", short.claims  # rescued by verbatim match
    print("PASS (g): numbers absent from every source are flagged; source/question numbers, [N] and list numbering aren't")


if __name__ == "__main__":
    test_same_section_chunks_not_collapsed()
    test_markers_point_to_matching_source_after_reorder()
    test_near_duplicates_suppressed_and_backfilled()
    test_split_cited_claims_formats()
    test_check_flags_swapped_invalid_and_uncited()
    test_numbers_must_come_from_sources()
    print("\nAll citation tests passed.")
