"""
Day 12: Evaluation against the golden QA set (data/eval/golden_qa.jsonl).

Runs every question through the same code path as rag_loop.py (router ->
retrieval plan -> route prompt -> local Ollama -> citation/number check) and
reports, per route and overall:

  Retrieval (no LLM, fast)
    route_acc   router picked the labelled route
    hit@k       at least one passage from a relevant document was retrieved
    mrr         1 / rank of the first relevant passage (0 if none)
    precision   share of retrieved passages that come from relevant documents

  Answers (--answers, one model per run)
    cite_ok     share of cited claims the citation check accepts
    invented#   numbers in the answer found in no source (per answer)
    facts       required figures (must_include) present in the answer
    abstain     unanswerable questions answered "not covered" (higher is better)
    false_abst  answerable questions wrongly answered "not covered"

Relevant documents were labelled from the raw text (which documents actually
contain the answer), not from retrieval output.

Usage:
    python scripts/evaluate.py                       # retrieval only
    python scripts/evaluate.py --answers             # + answers with llama3.2
    python scripts/evaluate.py --answers --model qwen2.5:7b
    python scripts/evaluate.py --answers --only q26,q27
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prompts  # noqa: E402
import rag_loop as r  # noqa: E402
from citations import check_citations  # noqa: E402
from portfolio_store import format_position  # noqa: E402

ROOT = Path(__file__).parent.parent
GOLDEN_PATH = ROOT / "data" / "eval" / "golden_qa.jsonl"
RESULTS_DIR = ROOT / "data" / "eval"
ROUTES = ("concept", "technical", "portfolio")

# Phrases that mean "the sources don't answer this"
ABSTAIN_RE = re.compile(
    r"(?:don'?t|do not|does not|doesn'?t) (?:cover|contain|provide|include|mention|have|specify)"
    r"|not (?:covered|provided|mentioned|available|included|specified|found)"
    r"|no (?:information|data|mention|details?) (?:on|about|regarding|is|are|was)"
    r"|(?:cannot|can'?t|unable to) (?:answer|determine|find|provide)"
    r"|nothing to answer",
    re.I,
)


def load_golden(only: set[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(GOLDEN_PATH, encoding="utf-8")]
    return [q for q in rows if not only or q["id"] in only]


def is_relevant(doc_title: str, relevant: list[str]) -> bool:
    return any(doc_title.startswith(label) for label in relevant)


def validate_labels(golden: list[dict], collection) -> list[str]:
    """Every relevant_docs label must match a document in the store."""
    titles = {m["doc_title"] for m in collection.get(include=["metadatas"])["metadatas"]}
    return [f"{q['id']}: {label!r}" for q in golden for label in q["relevant_docs"]
            if not any(t.startswith(label) for t in titles)]


def normalize_number_text(text: str) -> str:
    return text.replace(",", "").replace("$", "")


def evaluate_question(q: dict, collection, model: str | None, options: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):  # retrieval progress lines
        decision, plan, positions = r.plan_query(q["query"])
        citations = r.retrieve_for_plan(q["query"], collection, plan)

    ranks = [c.rank for c in citations if is_relevant(c.doc_title, q["relevant_docs"])]
    res = {
        "id": q["id"], "query": q["query"], "route": q["route"], "got_route": decision.route,
        "answerable": q["answerable"],
        "retrieved": [f"[{c.rank}] t{c.tier} {c.doc_title[:60]} ({c.distance:.3f})" for c in citations],
        "hit": bool(ranks) if q["relevant_docs"] else None,
        "rr": (1 / min(ranks) if ranks else 0.0) if q["relevant_docs"] else None,
        "precision": (len(ranks) / len(citations) if citations else 0.0) if q["relevant_docs"] else None,
    }
    if model is None:
        return res

    if not citations:  # same as rag_query: never ask the model to answer from nothing
        answer, seconds = prompts.NO_CONTEXT_ANSWER, 0.0
    else:
        prompt = r.build_prompt(q["query"], citations, positions=positions, route=decision.route)
        start = time.time()
        answer = r.generate(prompt, model=model, options=options)
        seconds = time.time() - start

    report = check_citations(answer, citations, r.get_embedder().encode_texts,
                             extra_sources=[q["query"]] + [format_position(h) for h in positions])
    statuses = [c.status for c in report.claims]
    norm = normalize_number_text(answer)
    res.update({
        "answer": answer,
        "seconds": round(seconds, 1),
        "claims": len(statuses),
        "claims_ok": statuses.count("ok"),
        "miscited": statuses.count("miscited"),
        "weak": statuses.count("weak"),
        "invalid": statuses.count("invalid"),
        "uncited": report.uncited_answer,
        "invented_numbers": report.ungrounded_numbers,
        "facts_ok": (all(normalize_number_text(f) in norm for f in q["must_include"])
                     if q["must_include"] else None),
        "abstained": bool(ABSTAIN_RE.search(answer)),
    })
    return res


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, pct: bool = True) -> str:
    if x is None:
        return "  —  "
    return f"{x:5.0%}" if pct else f"{x:5.2f}"


def summarize(results: list[dict], with_answers: bool) -> dict:
    groups = defaultdict(list)
    for res in results:
        groups[res["route"]].append(res)
        groups["ALL"].append(res)

    summary = {}
    header = f"{'route':<10} {'n':>3} {'route_acc':>9} {'hit@k':>6} {'mrr':>6} {'precision':>9}"
    if with_answers:
        header += f" {'cite_ok':>7} {'invented#':>9} {'facts':>6} {'abstain':>7} {'false_abst':>10} {'sec/ans':>7}"
    print(header)
    for name in (*ROUTES, "ALL"):
        rs = groups.get(name, [])
        if not rs:
            continue
        s = {
            "n": len(rs),
            "route_acc": mean([float(x["got_route"] == x["route"]) for x in rs]),
            "hit": mean([None if x["hit"] is None else float(x["hit"]) for x in rs]),
            "mrr": mean([x["rr"] for x in rs]),
            "precision": mean([x["precision"] for x in rs]),
        }
        line = (f"{name:<10} {s['n']:>3} {fmt(s['route_acc']):>9} {fmt(s['hit']):>6} "
                f"{fmt(s['mrr'], pct=False):>6} {fmt(s['precision']):>9}")
        if with_answers:
            claims = sum(x["claims"] for x in rs)
            answered = [x for x in rs if x["answerable"]]
            s.update({
                "cite_ok": sum(x["claims_ok"] for x in rs) / claims if claims else None,
                "invented_per_answer": mean([len(x["invented_numbers"]) for x in answered]),
                "facts": mean([None if x["facts_ok"] is None else float(x["facts_ok"]) for x in rs]),
                "abstain": mean([float(x["abstained"]) for x in rs if not x["answerable"]]),
                "false_abstain": mean([float(x["abstained"]) for x in answered]),
                "sec_per_answer": mean([x["seconds"] for x in rs if x["seconds"]]),
            })
            line += (f" {fmt(s['cite_ok']):>7} {fmt(s['invented_per_answer'], pct=False):>9} {fmt(s['facts']):>6}"
                     f" {fmt(s['abstain']):>7} {fmt(s['false_abstain']):>10} {fmt(s['sec_per_answer'], pct=False):>7}")
        print(line)
        summary[name] = s
    return summary


def print_failures(results: list[dict], with_answers: bool) -> None:
    print("\nNeeds attention:")
    any_shown = False
    for x in results:
        issues = []
        if x["got_route"] != x["route"]:
            issues.append(f"route {x['got_route']} (expected {x['route']})")
        if x["hit"] is False:
            issues.append("no relevant document retrieved")
        if with_answers:
            if x["facts_ok"] is False:
                issues.append("required figure missing")
            if x["invented_numbers"]:
                issues.append(f"numbers not in sources: {', '.join(x['invented_numbers'][:5])}")
            if x["miscited"] or x["weak"] or x["invalid"]:
                issues.append(f"citations: {x['miscited']} miscited, {x['weak']} weak, {x['invalid']} invalid")
            if x["uncited"]:
                issues.append("no citations")
            if not x["answerable"] and not x["abstained"]:
                issues.append("answered an unanswerable question")
            if x["answerable"] and x["abstained"]:
                issues.append("said 'not covered' (may be partial)")
        if issues:
            any_shown = True
            print(f"  {x['id']} {x['query'][:55]!r}: " + "; ".join(issues))
    if not any_shown:
        print("  nothing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FinRAG against the golden QA set.")
    parser.add_argument("--answers", action="store_true", help="Also generate and score answers (slow).")
    parser.add_argument("--model", default=r.DEFAULT_MODEL, help="Ollama model for --answers.")
    parser.add_argument("--only", default=None, help="Comma-separated question ids, e.g. q26,q27.")
    parser.add_argument("--no-diversity", action="store_true",
                        help="Disable the per-document cap for broad technical questions (A/B comparison).")
    args = parser.parse_args()
    if args.no_diversity:
        import router
        router.TECHNICAL_MAX_PER_DOC, router.TECHNICAL_OVERFETCH = None, 3

    golden = load_golden(set(args.only.split(",")) if args.only else None)
    with contextlib.redirect_stdout(io.StringIO()):
        collection = r.open_collection()
    if collection is None:
        sys.exit("Vector store not found — run scripts/build_vectorstore.py")
    bad = validate_labels(golden, collection)
    if bad:
        sys.exit("Golden-set labels that match no document:\n  " + "\n  ".join(bad))

    model = args.model if args.answers else None
    if model:
        r.ensure_local_model(model)  # never evaluate (send golden prompts) to a cloud model
    options = {"temperature": 0, "seed": 0}  # deterministic, comparable runs
    print(f"=== FinRAG eval: {len(golden)} questions | "
          f"{'answers with ' + model if model else 'retrieval only'} ===\n")
    results = []
    for i, q in enumerate(golden, 1):
        if model:
            print(f"  [{i}/{len(golden)}] {q['id']} {q['query'][:60]}", file=sys.stderr)
        results.append(evaluate_question(q, collection, model, options))

    summary = summarize(results, with_answers=bool(model))
    print_failures(results, with_answers=bool(model))

    name = "retrieval" if not model else "answers_" + re.sub(r"[^A-Za-z0-9.]+", "-", model)
    if args.no_diversity:
        name += "_nodiversity"
    if args.only:
        name += "_subset"
    out = RESULTS_DIR / f"eval_{name}.json"
    out.write_text(json.dumps({"model": model, "options": options, "summary": summary, "results": results},
                              indent=1), encoding="utf-8")
    print(f"\nSaved per-question results to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
