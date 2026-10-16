"""
Day 11: Per-route prompt templates (concept / technical / portfolio).

Every template shares GROUNDING_RULES — answer only from the numbered context,
cite [N], say so when the context doesn't cover the question — and the same
context block: [N] labels come from c.rank of the final deduped citation list,
the same list format_citations_block() prints, so [N] always matches source [N].

Portfolio facts are never left to the model: rag_loop prints the position
lines verbatim before the answer and appends PORTFOLIO_DISCLAIMER after it.
The portfolio template tells the model both are handled and forbids it from
restating or computing any position numbers.
"""
from __future__ import annotations

from typing import Callable

from citations import Citation

GROUNDING_RULES = """\
Grounding rules (required):
- Answer ONLY from the numbered context passages below. Do not use outside knowledge, even if you know the answer.
- End every sentence or list item that uses the context with the number of the passage it came from in square brackets, like this: "<sentence> [2]." Use [1][3] if it combines two passages.
- Only use numbers of passages listed in the context.
- Every number you write must appear in the context. Copy it exactly, with its unit and period (e.g. "(In millions)", "fiscal 2026"); never invent example numbers.
- If the context does not cover the question, or only part of it, say so explicitly (e.g. "The provided sources don't cover X.") instead of guessing.
- Do not add a separate "Sources" or "References" list — one is printed automatically."""

CONCEPT_STYLE = """\
You are a friendly finance tutor explaining a concept to a beginner.
Style:
- Plain language and short: aim for about 150 words.
- Define any jargon in simple words the first time you use it.
- If the context contains a worked example (numbers, a scenario), include ONE of them, using its numbers exactly. Never make up an example."""

TECHNICAL_STYLE = """\
You are a precise quantitative finance analyst answering a technical question.
Style:
- Be exact. Copy formulas, variable definitions and table values verbatim from the context.
- Never round, convert or re-derive numbers that appear in the context; quote them as written, with their units and periods (e.g. "fiscal 2025", "in millions").
- State the assumptions, conditions or scope the source attaches to a result (model, sample period, company, filing).
- If a value or step is not in the context, say it is not provided — do not estimate it."""

PORTFOLIO_STYLE = """\
You are a careful assistant explaining how source material relates to the user's own holdings.
The user's position details are shown to them separately, exactly as recorded, and an "educational, not investment advice" note is added automatically.
Style:
- Do NOT restate, recompute or estimate any position numbers (shares, average cost, cost basis, current value, gains or returns). Refer to the holding only as "your <TICKER> position".
- Explain the relevant filing-based risks or context, each with its [N] citation.
- When quoting figures from a filing table, copy them verbatim with the table's unit and period, and name the line item they belong to.
- Do not tell the user to buy, sell or hold anything."""

PORTFOLIO_DISCLAIMER = "Note: this is educational information based on company filings, not investment advice."

# B: with no passages there is nothing to ground an answer in — rag_loop prints
# this instead of calling the model (llama3.2 invented a "[1]" source when asked).
NO_CONTEXT_ANSWER = ("The sources in this corpus don't cover this question, so there is nothing to answer it from. "
                     "(The corpus has no market prices or performance data.)")

STYLES = {"concept": CONCEPT_STYLE, "technical": TECHNICAL_STYLE, "portfolio": PORTFOLIO_STYLE}


def build_prompt(
    query: str,
    citations: list[Citation],
    source_label: Callable[[Citation], str],
    route: str = "concept",
    positions: list[dict] | None = None,
    format_position: Callable[[dict], str] | None = None,
) -> str:
    """
    Build the full prompt for `route`. [N] labels, source labels and chunk text
    all come from the same Citation, numbered by c.rank — the same list
    format_citations_block() prints.
    """
    style = STYLES.get(route, CONCEPT_STYLE)
    ctx_parts = [f"[{c.rank}] {source_label(c)}\n{c.text}" for c in citations]
    context_str = "\n\n".join(ctx_parts) if ctx_parts else "(no passages matched this question)"

    portfolio_str = ""
    if positions and format_position:
        lines = "\n".join(f"- {format_position(h)}" for h in positions)
        portfolio_str = (f"=== My Portfolio (for your reference only — never restate or compute these numbers) ===\n"
                         f"{lines}\n\n")

    return (
        f"{style}\n\n{GROUNDING_RULES}\n\n"
        f"{portfolio_str}"
        f"=== Context ===\n{context_str}\n\n"
        f"=== Question ===\n{query}\n\n"
        f"=== Answer ===\n"
    )
