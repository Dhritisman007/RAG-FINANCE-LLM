"""
FinRAG chat UI (Streamlit) — the same pipeline as scripts/rag_loop.py:
route -> retrieve -> route prompt -> local LLM (Ollama or MLX) -> sources + citation check.

Local only, by design: the server binds 127.0.0.1 and sends no usage
statistics (.streamlit/config.toml); the app refuses to start otherwise,
because it shows your holdings. Generation only uses local models (Ollama, or MLX
on 127.0.0.1:8080 — see scripts/start_mlx_server.sh).

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
from portfolio_store import HOLDINGS_PATH, format_position, load_holdings, add_holding, save_holdings  # noqa: E402

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


@st.dialog("Edit holdings", width="large")
def edit_holdings_dialog() -> None:
    """Add, change or delete holdings; saved to data/portfolio/holdings.csv after validation."""
    from datetime import date
    st.caption("Add a row with the + at the bottom; select rows and press Delete to remove them. "
               "Saved only on this machine.")
    rows = [{"ticker": h["ticker"], "quantity": h["quantity"], "avg_cost": h["avg_cost"],
             "purchase_date": date.fromisoformat(h["purchase_date"]) if h["purchase_date"] else None}
            for h in load_holdings()]
    edited = st.data_editor(
        rows or [{"ticker": "", "quantity": None, "avg_cost": None, "purchase_date": None}],
        num_rows="dynamic", hide_index=True, width="stretch", key="holdings_editor",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", help="NSE: RELIANCE.NS · BSE: RELIANCE.BO · US: AAPL", max_chars=20),
            "quantity": st.column_config.NumberColumn("Shares", min_value=0.0, format="%g"),
            "avg_cost": st.column_config.NumberColumn("Avg price", min_value=0.0, format="%.2f", help="₹ for NSE/BSE, $ for US"),
            "purchase_date": st.column_config.DateColumn("Purchased", format="YYYY-MM-DD", max_value=date.today()),
        },
    )
    save, cancel = st.columns(2)
    if save.button("Save", type="primary", width="stretch"):
        errors = save_holdings(edited)
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.session_state.price_note = "Holdings saved"
            st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("FinRAG")
    models = r.local_models()
    if not models:
        st.error("No local models found. Either start Ollama (127.0.0.1:11434) or the MLX server "
                 "(`scripts/start_mlx_server.sh`, 127.0.0.1:8080). If Ollama's models live on the external "
                 "drive and the Mac just restarted, Ollama may have started before the drive mounted and "
                 "fallen back to its default (empty) folder — quit and reopen Ollama.")
        st.stop()
    default = models.index(r.DEFAULT_MODEL) if r.DEFAULT_MODEL in models else \
        next((i for i, m in enumerate(models) if m.startswith(r.DEFAULT_MODEL)), 0)
    model = st.selectbox("Model (local only)", models, index=default,
                         help="mlx:… models run on Apple's MLX server; the others on Ollama.")
    route = st.radio("Route", ["auto", "concept", "technical", "portfolio"], horizontal=True,
                     help="auto = deterministic rules in router.py")
    use_portfolio = st.toggle("Use my holdings", value=True)
    check = st.toggle("Check citations", value=True)
    show_details = st.toggle("Show routing details", value=False)

    st.caption("🔒 Answers are generated locally (cloud models refused, no telemetry). Your holdings file "
               "stays on this machine; live prices send only stock symbols to Yahoo Finance.")
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

LIVE_EVERY_SEC = 60


def portfolio_table(holdings: list[dict], quotes: dict[str, dict]) -> tuple[list[dict], dict]:
    """
    Rows for the live table plus ₹ totals — all arithmetic here, never by the model.
    Indian listings are in ₹; US holdings are converted at the USD→INR rate.
    """
    fx = (quotes.get(prices.FX_TICKER) or {}).get("price")
    rows = []
    tot = {"value": 0.0, "cost": 0.0, "day": 0.0, "prev": 0.0, "missing": [], "no_fx": []}
    for h in holdings:
        symbol, exchange = prices.display_symbol(h["ticker"])
        cur = prices.currency(h["ticker"])
        rate = 1.0 if cur == "INR" else fx
        q = quotes.get(prices.normalize_ticker(h["ticker"]))
        row = {"Stock": symbol, "Exchange": exchange, "Qty": h["quantity"],
               "Avg price": prices.money(h["avg_cost"], cur), "LTP": None, "Day %": None,
               "Current (₹)": None, "P&L (₹)": None, "P&L %": None}
        if q is None or rate is None:
            (tot["missing"] if q is None else tot["no_fx"]).append(symbol)
            rows.append(row)
            continue
        cost, value = h["quantity"] * h["avg_cost"] * rate, h["quantity"] * q["price"] * rate
        prev = q.get("prev_close")
        tot["value"] += value
        tot["cost"] += cost
        if prev:
            tot["day"] += h["quantity"] * (q["price"] - prev) * rate
            tot["prev"] += h["quantity"] * prev * rate
        row.update({"LTP": prices.money(q["price"], cur), "Day %": (q["price"] / prev - 1) * 100 if prev else None,
                    "Current (₹)": prices.inr(value), "P&L (₹)": prices.signed_money(value - cost),
                    "P&L %": (value - cost) / cost * 100 if cost else None})
        rows.append(row)
    tot["fx"] = fx if any(prices.currency(h["ticker"]) == "USD" for h in holdings) else None
    return rows, tot


def portfolio_panel(pending) -> None:
    with st.expander("📊 My portfolio", expanded=True):
        with st.form("add_stock", clear_on_submit=True, border=False):
            c1, c2, c3, c4, c5, c6 = st.columns([1.4, 0.9, 0.9, 1.1, 1.3, 0.8], vertical_alignment="bottom")
            symbol = c1.text_input("Stock symbol", placeholder="e.g. RELIANCE, TCS", max_chars=16)
            exchange = c2.selectbox("Exchange", ["NSE", "BSE", "US"])
            qty = c3.number_input("Qty", min_value=0.0, value=None, step=1.0, format="%g")
            cost = c4.number_input("Avg buy price", min_value=0.0, value=None, step=1.0, format="%.2f",
                                   help="Per share — in ₹ for NSE/BSE, in $ for US stocks")
            bought = c5.date_input("Bought on (optional)", value=None, format="DD/MM/YYYY")
            if c6.form_submit_button("➕ Add", type="primary", width="stretch"):
                ticker = prices.with_exchange(symbol, exchange) if symbol.strip() else ""
                errors = add_holding({"ticker": ticker, "quantity": qty, "avg_cost": cost,
                                      "purchase_date": bought.isoformat() if bought else ""})
                for e in errors:
                    st.error(e)
                if not errors:
                    st.session_state.live_quotes = None  # fetch the new stock's price now
                    st.session_state.price_note = f"Saved {prices.display_symbol(ticker)[0]} ({exchange})"

        holdings = load_holdings()
        if not holdings:
            st.caption("No holdings yet — add your first stock above (NSE symbols like RELIANCE, TCS, HDFCBANK). "
                       "Saved only on this machine.")
            return
        top = st.columns([1, 1, 3], vertical_alignment="center")
        live = top[0].toggle("Live prices", value=True, key="live",
                             help=f"Refresh prices from Yahoo Finance every {LIVE_EVERY_SEC} s. Yahoo sees "
                                  "your stock symbols — never your quantities or prices paid.")
        if top[1].button("Edit / remove", help="Change or delete holdings"):
            edit_holdings_dialog()
        if st.session_state.get("price_note"):
            top[2].success(st.session_state.pop("price_note"))

        # Pause live refresh while an answer is generating, so a refresh can't interrupt it.
        @st.fragment(run_every=LIVE_EVERY_SEC if live and pending is None else None)
        def live_view() -> None:
            import time
            from datetime import datetime
            tickers = [h["ticker"] for h in holdings]
            cached = st.session_state.get("live_quotes")
            fresh = cached and cached["tickers"] == tickers and time.time() - cached["at"] < LIVE_EVERY_SEC - 5
            error = None
            if live and not fresh:
                try:
                    quotes = prices.yahoo_quotes(prices.with_fx(tickers))
                    prices.save_yahoo_prices(quotes)  # answers use the same prices
                    st.session_state.live_quotes = {"tickers": tickers, "at": time.time(), "quotes": quotes}
                except Exception as e:
                    error = e
            cached = st.session_state.get("live_quotes")
            if cached and cached["tickers"] == tickers:
                quotes, stamp = cached["quotes"], datetime.fromtimestamp(cached["at"]).strftime("%H:%M:%S")
            else:  # live off (or failed before any fetch): last saved prices, no day change
                quotes = {t: {**p, "prev_close": None} for t, p in prices.load_prices().items()}
                stamp = None
            rows, tot = portfolio_table(holdings, quotes)
            pnl = tot["value"] - tot["cost"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Invested", prices.inr(tot["cost"]))
            m2.metric("Current value", prices.inr(tot["value"]))
            m3.metric("Day's P&L", prices.signed_money(tot["day"]) if tot["prev"] else "—",
                      f"{tot['day'] / tot['prev'] * 100:+.2f}%" if tot["prev"] else None)
            m4.metric("Total P&L", prices.signed_money(pnl),
                      f"{pnl / tot['cost'] * 100:+.2f}%" if tot["cost"] else None)
            st.dataframe(rows, hide_index=True, width="stretch", column_config={
                "Qty": st.column_config.NumberColumn(format="%g"),
                "Day %": st.column_config.NumberColumn(format="%+.2f%%"),
                "P&L %": st.column_config.NumberColumn(format="%+.2f%%"),
            })
            note = (f"Live · updated {stamp} · Yahoo Finance (NSE/BSE quotes may be delayed)" if stamp
                    else "Last saved prices — turn on Live prices to refresh")
            if tot["fx"]:
                note += f" · US stocks at ₹{tot['fx']:.2f}/$"
            if tot["missing"]:
                note += f" · no price for {', '.join(tot['missing'])} — check the symbol and exchange"
            if tot["no_fx"]:
                note += f" · no USD→INR rate yet for {', '.join(tot['no_fx'])}"
            st.caption(note)
            if error:
                st.warning(f"Price refresh failed: {error}")

        live_view()


if "messages" not in st.session_state:
    st.session_state.messages = []

pending = st.session_state.get("pending")
if use_portfolio:
    portfolio_panel(pending)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(md(msg["content"]))
        else:
            render_assistant(msg["result"], show_details)

# A new question while an answer is still generating would restart the script and
# silently drop that answer, so the input is locked until the pending one is done.
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
