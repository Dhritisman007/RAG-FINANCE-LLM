"""
Per-route prompt template tests. No Chroma / Ollama needed.

    python tests/test_prompts.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import prompts  # noqa: E402
from citations import (  # noqa: E402
    citations_from_chroma_results,
    deduplicate_citations,
    format_citations_block,
    suppress_near_duplicates,
)
from rag_loop import CLOUD_MODEL_RE, CloudModelError, build_prompt, ensure_local_model  # noqa: E402

ROUTES = ("concept", "technical", "portfolio")

GROUNDING_MUST_SAY = [
    "Answer ONLY from the numbered context",
    "End every sentence or list item that uses the context with the number",
    "Only use numbers of passages listed in the context",
    "Every number you write must appear in the context",
    "never invent example numbers",
    "If the context does not cover the question",
    "say so explicitly",
    'Do not add a separate "Sources" or "References" list',
]
STYLE_MUST_SAY = {
    "concept": ["Plain language", "Define any jargon", "ONLY if a worked example appears in the context",
                "Do NOT write illustrations", "Every number in your answer must be copied from the context"],
    "technical": ["Copy formulas", "verbatim", "Never round", "State the assumptions", "do not estimate"],
    "portfolio": ["Do NOT restate, recompute or estimate any position numbers", "[N] citation",
                  "not investment advice", "Do not tell the user to buy, sell or hold"],
}
POSITION = {"ticker": "AAPL", "quantity": 25.0, "avg_cost": 182.5, "purchase_date": "2024-03-15"}


def fake_results(rows):
    """rows: (chunk_id, text, distance) in raw Chroma order."""
    meta = lambda cid: {"chunk_id": cid, "tier": 2, "source": "sec_edgar", "doc_title": "AAPL 10-K filed 2025-10-31",
                        "section_heading": "AAPL 10-K filed 2025-10-31", "company_ticker": "AAPL", "url": "",
                        "content_hash": cid}
    return {"ids": [[r[0] for r in rows]], "documents": [[r[1] for r in rows]],
            "metadatas": [[meta(r[0]) for r in rows]], "distances": [[r[2] for r in rows]]}


def final_citations():
    # out-of-order distances + a repeated chunk, so dedup re-orders the list
    results = fake_results([
        ("c_C", "CCC margins are under pressure from component costs.", 0.30),
        ("c_A", "AAA supplier industrial accidents could disrupt production.", 0.10),
        ("c_C", "CCC margins are under pressure from component costs.", 0.35),
        ("c_B", "BBB litigation outcomes are inherently uncertain.", 0.20),
    ])
    return suppress_near_duplicates(deduplicate_citations(citations_from_chroma_results(results)), k=5)


def test_every_template_has_grounding_and_style_rules() -> None:
    cites = final_citations()
    for route in ROUTES:
        prompt = build_prompt("q?", cites, positions=[POSITION] if route == "portfolio" else None, route=route)
        for rule in GROUNDING_MUST_SAY + STYLE_MUST_SAY[route]:
            assert rule in prompt, f"{route} prompt is missing: {rule!r}"
        others = [s for r, s in prompts.STYLES.items() if r != route]
        assert not any(s in prompt for s in others), f"{route} prompt contains another route's style"
    print("PASS: each route's template has all grounding rules + its own style rules (and no other route's)")


def test_labels_match_printed_source_order() -> None:
    cites = final_citations()
    printed = format_citations_block(cites, verbose=True)
    printed_previews = {int(m.group(1)): m.group(2) for m in
                        re.finditer(r'^\[(\d+)\].*\n(?:    .*\n)*?    "(.{20})', printed + "\n", re.M)}
    for route in ROUTES:
        prompt = build_prompt("q?", cites, positions=[POSITION] if route == "portfolio" else None, route=route)
        context = prompt.split("=== Context ===\n", 1)[1].split("\n\n=== Question ===", 1)[0]
        blocks = re.findall(r"^\[(\d+)\] [^\n]*\n(.*)$", context, re.M)
        labels = [int(n) for n, _ in blocks]
        assert labels == [1, 2, 3], (route, labels)
        for n, text in blocks:
            assert text.startswith(printed_previews[int(n)]), (route, n, text, printed_previews[int(n)])
    print("PASS: in every template, [N] labels run 1..n and each matches printed source [N]")


def test_portfolio_positions_only_in_portfolio_prompt() -> None:
    cites = final_citations()
    portfolio = build_prompt("q?", cites, positions=[POSITION], route="portfolio")
    assert "=== My Portfolio" in portfolio and "AAPL: 25 shares @ avg cost $182.50" in portfolio
    concept = build_prompt("q?", cites, route="concept")
    assert "My Portfolio" not in concept
    print("PASS: position facts appear only in the portfolio prompt, verbatim from holdings")


def test_empty_context_and_default_route() -> None:
    prompt = build_prompt("How is my portfolio doing?", [], positions=[POSITION], route="portfolio")
    assert "(no passages matched this question)" in prompt and "say so explicitly" in prompt
    assert prompts.CONCEPT_STYLE in build_prompt("q?", final_citations())  # default route stays concept
    assert "don't cover this question" in prompts.NO_CONTEXT_ANSWER
    print("PASS: empty context is stated explicitly; build_prompt defaults to the concept template")


def test_rules_contain_no_copyable_domain_example() -> None:
    # llama3.2 copied the old example sentence ("An EMI is a fixed monthly loan
    # payment [1].") into an unrelated answer about bid-ask spreads.
    assert "<sentence> [2]" in prompts.GROUNDING_RULES
    for word in ("EMI", "loan", "interest", "bond", "revenue"):
        assert not re.search(rf"\b{word}\b", prompts.GROUNDING_RULES, re.I), word
    print("PASS: grounding rules use a content-free citation example the model can't copy into answers")


def test_cloud_models_are_refused() -> None:
    # Ollama cloud models answer on localhost but run remotely — portfolio data would leave the machine.
    for model in ("kimi-k2.5:cloud", "gemma4:31b-cloud", "gpt-oss:120b-cloud", "deepseek-v3.1:671b-cloud"):
        try:
            ensure_local_model(model)
        except CloudModelError:
            continue
        raise AssertionError(f"{model} was not refused")
    assert not any(CLOUD_MODEL_RE.search(m) for m in ("llama3.2", "qwen2.5:7b", "mycloudmodel", "phi3"))
    print("PASS: Ollama cloud models are refused; local model names are allowed")


def test_concept_reminder_sits_right_before_the_answer() -> None:
    concept = build_prompt("q?", final_citations(), route="concept")
    assert concept.rstrip().endswith(prompts.FINAL_REMINDERS["concept"] + "\n\n=== Answer ===")
    technical = build_prompt("q?", final_citations(), route="technical")
    assert prompts.FINAL_REMINDERS["concept"] not in technical
    print("PASS: the concept prompt ends with the no-made-up-examples reminder; other routes don't")


if __name__ == "__main__":
    test_every_template_has_grounding_and_style_rules()
    test_labels_match_printed_source_order()
    test_portfolio_positions_only_in_portfolio_prompt()
    test_empty_context_and_default_route()
    test_rules_contain_no_copyable_domain_example()
    test_cloud_models_are_refused()
    test_concept_reminder_sits_right_before_the_answer()
    print("\nAll prompt tests passed.")
