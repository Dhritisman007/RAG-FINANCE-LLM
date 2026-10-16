"""
Tests for the evaluator's scoring helpers (a bug here would skew every score).

    python tests/test_evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import prompts  # noqa: E402
from evaluate import ABSTAIN_RE, GOLDEN_PATH, is_refusal, is_relevant, normalize_number_text  # noqa: E402


def test_abstain_detection() -> None:
    abstains = [
        prompts.NO_CONTEXT_ANSWER,
        "The provided sources don't cover Nvidia's revenue.",
        "The context does not contain information about quantitative easing.",
        "Third-quarter 2026 revenue is not provided in the context.",
        "I cannot determine Apple's future stock price from these filings.",
    ]
    answers = [
        "ExxonMobil's sales and other operating revenue in 2025 was 323,905 million [1].",
        "Compound interest is interest on interest [1].",
        "Duration measures how sensitive a bond's price is to rate changes [2].",
    ]
    assert all(ABSTAIN_RE.search(a) for a in abstains), [a for a in abstains if not ABSTAIN_RE.search(a)]
    assert not any(ABSTAIN_RE.search(a) for a in answers), [a for a in answers if ABSTAIN_RE.search(a)]
    print("PASS: 'not covered' answers are detected; ordinary answers are not")


def test_refusal_means_leading_with_not_covered() -> None:
    refusals = [
        prompts.NO_CONTEXT_ANSWER,
        "The provided sources do not cover Apple's future stock price predictions. [1][3][4][5]",
        "The provided sources do not cover Q3 2026 revenue. The closest data is Q1 2026: 83,161 million [1].",
    ]
    partial = "Compound interest is interest on interest [1]. The sources don't cover tax treatment."
    assert all(is_refusal(a) for a in refusals), [a for a in refusals if not is_refusal(a)]
    assert not is_refusal(partial) and not is_refusal("Duration measures rate sensitivity [2].")
    print("PASS: a refusal leads with 'not covered' (citations allowed); a later gap note is a partial answer")


def test_relevance_and_numbers() -> None:
    assert is_relevant("AAPL 10-K filed 2025-10-31", ["AAPL 10-K"])
    assert not is_relevant("AAPL 10-Q filed 2026-05-01", ["AAPL 10-K"])
    assert normalize_number_text("$323,905") in normalize_number_text("revenue was 323,905 million")
    print("PASS: relevant-doc prefix matching and number normalization")


def test_golden_set_is_well_formed() -> None:
    rows = [json.loads(line) for line in open(GOLDEN_PATH, encoding="utf-8")]
    ids = [q["id"] for q in rows]
    assert len(ids) == len(set(ids)) and len(rows) >= 40
    for q in rows:
        assert q["route"] in ("concept", "technical", "portfolio"), q
        assert q["answerable"] == bool(q["relevant_docs"]), f"{q['id']}: answerable iff it has relevant docs"
    print(f"PASS: golden set well-formed ({len(rows)} questions, unique ids, labels consistent)")


if __name__ == "__main__":
    test_abstain_detection()
    test_refusal_means_leading_with_not_covered()
    test_relevance_and_numbers()
    test_golden_set_is_well_formed()
    print("\nAll evaluate tests passed.")
