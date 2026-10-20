"""
FinRAG chat UI (Streamlit) — the same pipeline as scripts/rag_loop.py:
route -> retrieve -> route prompt -> local Ollama -> sources + citation check.

Local only, by design: the server binds 127.0.0.1 and sends no usage
statistics (.streamlit/config.toml); the app refuses to start otherwise,
because it shows your holdings. Generation only uses local Ollama models.

Run from finrag/:
    streamlit run app.py
"""
from __future__ import annotations

import contextlib
import io
import itertools
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import prices  # noqa: E402
import prompts  # noqa: E402
import rag_loop as r  # noqa: E402
from citations import Citation, CitationReport, check_citations, is_refusal  # noqa: E402
from portfolio_store import HOLDINGS_PATH, format_position, load_holdings  # noqa: E402

st.set_page_config(page_title="FinRAG", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")  # model, route and privacy note visible on load
# Streamlit collapses the sidebar to 0 px but lets its contents spill out ("F/i/n/R/A/G"
# stacked down the left edge); clip it while collapsed.
st.markdown('<style>[data-testid="stSidebar"][aria-expanded="false"] { overflow: hidden; }</style>',
            unsafe_allow_html=True)

# ── Privacy guard ────────────────────────────────────────────────────────────
if st.get_option("server.address") not in ("127.0.0.1", "localhost") or st.get_option("browser.gatherUsageStats"):
    st.error("FinRAG shows your holdings, so it only runs bound to 127.0.0.1 with usage statistics off. "
             "Run it from `finrag/` (which has `.streamlit/config.toml`):  `streamlit run app.py`")
    st.stop()


# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the vector store…")
def get_collection():
    with contextlib.redirect_stdout(io.StringIO()):
        return r.open_collection()


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_encoder():
    return r.get_embedder().encode_texts


def md(text: str) -> str:
    """Escape '$' so Streamlit doesn't render '$209,586 … $416,161' as LaTeX."""
    return text.replace("$", "\\$")


# ── Rendering ────────────────────────────────────────────────────────────────
def render_route(res: dict, show_details: bool) -> None:
    also = f" · also matched: {', '.join(res['secondary'])}" if res["secondary"] else ""
    st.caption(f"🧭 Route: **{res['route']}** (confidence {res['confidence']:.2f}){also}")
    if show_details:
        with st.expander("Routing details"):
            st.code(res["describe"], language=None)


def render_retrieval_notes(res: dict) -> None:
    for note in res["retrieval_notes"]:
        st.caption(note)


def short_claim(text: str, limit: int = 140) -> str:
    """A flagged claim for display: no list marker or bold markup, cut at a word boundary."""
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", " ".join(text.split())).replace("**", "")
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def render_positions(positions: list[dict]) -> None:
    if positions:
        st.markdown("**Your position(s)** — from your local holdings file, shown exactly as recorded:")
        st.markdown("\n".join(f"- {md(format_position(h))}" for h in positions))


def render_valuation(valued: list[dict] | None) -> None:
    """Market value and gain per position — computed by prices.py, never by the model."""
    if not valued:
        return
    st.markdown("**Valuation** — local prices, computed by code:")
    st.markdown("\n".join(f"- {md(line)}" for line in prices.format_valuation(valued)))


def source_title(c: Citation) -> str:
    return f"[{c.rank}] {r.source_label(c)} · tier {c.tier} · distance {c.distance:.3f}"


def render_sources(citations: list[Citation], refusal: bool = False) -> None:
    if not citations:
        return
    if refusal:
        st.markdown(f"**Closest passages found** ({len(citations)}) — not enough to answer the question")
    else:
        st.markdown(f"**Sources** ({len(citations)})")
    for c in citations:
        with st.expander(source_title(c)):
            if c.url:
                st.markdown(f"[{c.url}]({c.url})")
            if c.section_heading and c.section_heading != c.doc_title:
                st.caption(c.section_heading)
            st.text(c.text[:3000] + ("…" if len(c.text) > 3000 else ""))


def render_report(report: CitationReport | None) -> None:
    if report is None:
        return
    if report.uncited_answer:
        st.warning("Citation check: the answer has no [N] citations — its claims can't be traced to sources.")
    elif report.claims:
        n_ok = sum(c.status == "ok" for c in report.claims)
        msg = f"Citation check: {n_ok}/{len(report.claims)} cited claims match their sources"
        (st.success if not report.problems else st.warning)(msg)
        for c in report.problems:
            cited = ", ".join(f"[{n}]" for n in c.cited)
            claim = short_claim(c.text)
            if c.status == "miscited":
                st.markdown(md(f"- ⚠ {cited} likely should be **[{c.suggested}]** "
                               f"(support {c.support:.2f} vs {c.suggested_support:.2f}): “{claim}”"))
            elif c.status == "weak":
                st.markdown(md(f"- ? {cited} weakly supports this ({c.support:.2f}) — may not be in the sources: “{claim}”"))
            else:
                st.markdown(md(f"- ✗ {cited} is not one of the listed sources: “{claim}”"))
    if report.ungrounded_numbers:
        st.warning(md("Numbers not found in any source (possibly invented): "
                      + ", ".join(report.ungrounded_numbers[:12])))


def render_assistant(res: dict, show_details: bool) -> None:
    """Re-render a finished answer from chat history."""
    render_route(res, show_details)
    render_retrieval_notes(res)
    render_positions(res["positions"])
    render_valuation(res.get("valuation"))
    st.markdown(md(res["answer"]))
    if res["disclaimer"]:
        st.caption(res["disclaimer"])
    render_sources(res["citations"], refusal=res.get("refusal", False))
    render_report(res["report"])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("FinRAG")
    models = r.local_models()
    if not models:
        st.error("No local Ollama models found. Is Ollama running on 127.0.0.1:11434? If the models live on "
                 "the external drive and the Mac just restarted, Ollama may have started before the drive "
                 "mounted and fallen back to its default (empty) folder — quit and reopen Ollama.")
        st.stop()
    default = models.index(r.DEFAULT_MODEL) if r.DEFAULT_MODEL in models else \
        next((i for i, m in enumerate(models) if m.startswith(r.DEFAULT_MODEL)), 0)
    model = st.selectbox("Model (local only)", models, index=default)
    route = st.radio("Route", ["auto", "concept", "technical", "portfolio"], horizontal=True,
                     help="auto = deterministic rules in router.py")
    use_portfolio = st.toggle("Use my holdings", value=True)
    check = st.toggle("Check citations", value=True)
    show_details = st.toggle("Show routing details", value=False)

    if use_portfolio:
        holdings = load_holdings()
        st.subheader("Holdings")
        if holdings:
            st.markdown("\n".join(f"- **{h['ticker']}** · {h['quantity']:g} sh @ {md(f'${h['avg_cost']:,.2f}')}"
                                   f" · since {h['purchase_date']}" for h in holdings))
            summary = prices.summarize(prices.value_positions(holdings, prices.load_prices()))
            if summary["priced"]:
                st.caption(md(f"Worth ${summary['market_value']:,.2f} · {summary['gain_pct']:+.1f}% "
                              f"on cost · prices as of {summary['as_of']}"
                              + (" ⚠ stale" if summary["stale"] else "")))
            if st.button("Update prices", help="One request downloads ~7,000 US stock prices from Nasdaq's "
                                               "screener. It contains no tickers, so it reveals nothing "
                                               "about your holdings."):
                with st.spinner("Downloading whole-market prices…"):
                    try:
                        n = prices.fetch_market_prices()
                        st.session_state.price_note = f"Updated {n:,} prices"
                        st.rerun()  # redraw the sidebar summary with the new prices
                    except Exception as e:
                        st.error(f"Price update failed: {e}")
            if st.session_state.get("price_note"):
                st.success(st.session_state.pop("price_note"))
        else:
            st.caption(f"No holdings file at {HOLDINGS_PATH.relative_to(Path(__file__).parent)}")
    st.caption("🔒 Local only: Ollama on 127.0.0.1, cloud models refused, no telemetry. "
               "Holdings never leave this machine.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main ─────────────────────────────────────────────────────────────────────
collection = get_collection()
if collection is None:
    st.error("Vector store not found — run `python scripts/build_vectorstore.py` first.")
    st.stop()

st.title("📈 FinRAG")
st.caption(f"Answers from {collection.count():,} passages of finance explainers, SEC filings "
           "and market-microstructure papers — every answer cites its sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(md(msg["content"]))
        else:
            render_assistant(msg["result"], show_details)

# A new question while an answer is still generating would restart the script and
# silently drop that answer, so the input is locked until the pending one is done.
pending = st.session_state.get("pending")
query = st.chat_input("Ask about a finance concept, a filing, a paper, or your holdings…",
                      disabled=pending is not None)
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.pending = query
    st.rerun()

if pending:
    query = pending
    try:
        with st.chat_message("assistant"):
            decision, plan, positions = r.plan_query(query, use_portfolio=use_portfolio, route=route)
            res = {
                "route": decision.route, "confidence": decision.confidence, "secondary": decision.secondary_routes,
                "describe": decision.describe(), "positions": positions, "citations": [],
                "retrieval_notes": [], "answer": "", "disclaimer": "", "report": None,
                "valuation": r.valuation_for(decision, positions),
            }
            render_route(res, show_details)  # the route is known before searching — show it right away
            log = io.StringIO()
            with st.spinner("Searching the corpus…"), contextlib.redirect_stdout(log):
                citations = r.retrieve_for_plan(query, collection, plan)
            res["citations"] = citations
            res["retrieval_notes"] = [line.strip() for line in log.getvalue().splitlines() if line.startswith("↪")]
            render_retrieval_notes(res)
            render_positions(positions)
            render_valuation(res["valuation"])

            if not citations:  # never ask the model to answer from nothing
                res["answer"] = (prices.summary_sentence(res["valuation"]) if res["valuation"]
                                 else prompts.NO_CONTEXT_ANSWER)
                st.markdown(md(res["answer"]))
            else:
                prompt = r.build_prompt(query, citations, positions=positions, route=decision.route)
                try:
                    tokens = r.stream_tokens(prompt, model=model)
                    # On an 8 GB machine the first token can take a minute (model load + reading
                    # the passages): show progress until it arrives, then stream.
                    with st.spinner(f"{model} is reading {len(citations)} passages… "
                                    "(the first answer also loads the model)"):
                        first = next(tokens, "")
                    res["answer"] = st.write_stream(md(tok) for tok in itertools.chain([first], tokens))
                    res["answer"] = res["answer"].replace("\\$", "$")  # store unescaped
                except Exception as e:  # unreachable server, cloud-model refusal, ...
                    st.error(f"Generation failed: {e}")

            res["refusal"] = bool(res["answer"]) and is_refusal(res["answer"])
            if decision.route == "portfolio" and res["answer"] and citations and not res["refusal"]:
                res["disclaimer"] = prompts.PORTFOLIO_DISCLAIMER
                st.caption(res["disclaimer"])
            render_sources(citations, refusal=res["refusal"])
            # A refusal has nothing to check — the "no citations" warning would make a
            # correct "not covered" answer look like a failure.
            if check and res["answer"] and citations and not res["refusal"]:
                with st.spinner("Checking citations…"):
                    res["report"] = check_citations(res["answer"], citations, get_encoder(),
                                                    extra_sources=[query] + [format_position(h) for h in positions])
                render_report(res["report"])

        st.session_state.messages.append({"role": "assistant", "result": res})
    except Exception:
        st.session_state.pending = None  # a failing question must not lock the input forever
        raise
    st.session_state.pending = None
    st.rerun()  # re-enable the input
