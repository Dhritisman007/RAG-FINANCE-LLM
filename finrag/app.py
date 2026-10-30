"""
PortIQ — the FinRAG chat UI (Streamlit), the same pipeline as scripts/rag_loop.py:
route -> retrieve -> route prompt -> local LLM (Ollama or MLX) -> sources + citation check.

Local only, by design: the server binds 127.0.0.1 and sends no usage
statistics (.streamlit/config.toml); the app refuses to start otherwise,
because it shows your holdings. Generation only uses local models (Ollama, or MLX
on 127.0.0.1:8080 — see scripts/start_mlx_server.sh).

Hosted mode (PORTIQ_MODE=hosted — the website): Google sign-in, a privacy notice
each user accepts before adding anything, and each user's holdings in their own
folder (scripts/workspace.py); broker connections are off. Local mode is unchanged.

Run from finrag/:
    streamlit run app.py
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import html
import io
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import llm_backends as backends  # noqa: E402
import prices  # noqa: E402
import prompts  # noqa: E402
import insights  # noqa: E402
import mutual_funds  # noqa: E402
import remote_store  # noqa: E402
import usage  # noqa: E402
import portfolio_db  # noqa: E402
import workspace  # noqa: E402
from brokers.credentials import get_secret  # noqa: E402
from brokers.csv_adapter import guess_mapping, parse_rows, read_headers  # noqa: E402
from brokers.base import BrokerError, MalformedResponse  # noqa: E402
import stock_list  # noqa: E402
import rag_loop as r  # noqa: E402
from citations import Citation, CitationReport, check_citations, is_refusal  # noqa: E402
from portfolio_store import (HOLDINGS_PATH, add_holding, format_position, load_holdings,  # noqa: E402
                             load_manual_holdings, save_holdings, source_label)

# Brand: PortIQ — "Upload. Ask. Get Insights." The mark in assets/ is an SVG recreation;
# a saved original (assets/portiq_logo.png or .svg) is used instead when present.
ASSETS = Path(__file__).parent / "assets"
LOGO = next((p for p in (ASSETS / "portiq_logo.svg", ASSETS / "portiq_logo.png") if p.exists()),
            ASSETS / "portiq_mark.svg")
st.set_page_config(page_title="PortIQ", page_icon=str(LOGO), layout="wide", initial_sidebar_state="collapsed")

def as_session_user(fn):
    """
    Dialogs and fragments rerun on their own, without the page's sign-in gate, so they re-apply this browser
    session's user (stored server-side by the gate) before touching any data. Put it under @st.dialog/@st.fragment.
    """
    @functools.wraps(fn)
    def run(*args, **kwargs):
        workspace.use(st.session_state.get("_portiq_workspace"))
        return fn(*args, **kwargs)
    return run


# ── Privacy guard ────────────────────────────────────────────────────────────
HOSTED = workspace.hosted()
workspace.use(None)  # every run starts as nobody; the sign-in gate (Page, below) sets the user in hosted mode
if st.get_option("browser.gatherUsageStats") or (
        not HOSTED and st.get_option("server.address") not in ("127.0.0.1", "localhost")):
    st.error("PortIQ shows your holdings, so it only runs bound to 127.0.0.1 with usage statistics off. "
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
ROUTE_LABELS = {"concept": "📖 Finance explainer", "technical": "🏢 Company documents & research",
                "portfolio": "💼 Your portfolio"}


def render_route(res: dict, show_details: bool) -> None:
    st.markdown(f'<span class="route-chip">{ROUTE_LABELS.get(res["route"], res["route"])}</span>',
                unsafe_allow_html=True)
    if show_details:
        also = f" · also matched: {', '.join(res['secondary'])}" if res["secondary"] else ""
        with st.expander("Routing details"):
            st.caption(f"Route: {res['route']} (confidence {res['confidence']:.2f}){also}")
            st.code(res["describe"], language=None)


def render_retrieval_notes(res: dict, show_details: bool) -> None:
    if show_details:  # "best tier-1 match 0.31 > 0.28 …" is for debugging, not customers
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


def source_info(c: Citation) -> tuple[str, str, str]:
    """(icon, document, where in it) for a source card."""
    section = c.section_heading if c.section_heading and c.section_heading != c.doc_title else ""
    if c.tier == 1:
        return "📖", f"{c.doc_title} — {c.source.title()}", section or "Explainer"
    if c.tier == 2 and "Annual Report FY" in c.doc_title:
        return "📘", c.doc_title, section or "Annual report"
    if c.tier == 2:
        return "🧾", c.doc_title, section or "SEC filing"
    return "📄", c.doc_title, f"Research paper · arXiv:{c.arxiv_id}" if c.arxiv_id else "Research paper"


def render_sources(citations: list[Citation], refusal: bool = False, show_details: bool = False) -> None:
    if not citations:
        return
    head = "Closest passages found — not enough to answer" if refusal else f"Sources ({len(citations)})"
    cards = []
    for c in citations:
        icon, doc, where = source_info(c)
        title = html.escape(doc)
        if (c.url or "").startswith(("https://", "http://")):
            title = f'<a href="{html.escape(c.url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
        if show_details:
            where += f" · distance {c.distance:.3f}"
        cards.append(f'<div class="src"><div class="n">{c.rank}</div><div><div class="t">{icon} {title}</div>'
                     f'<div class="s">{html.escape(where)}</div></div></div>')
    st.markdown(f'<p class="pane-title" style="margin-top:8px">{head}</p><div class="srcs">{"".join(cards)}</div>',
                unsafe_allow_html=True)
    with st.expander("Read the cited passages"):
        for c in citations:
            _, doc, where = source_info(c)
            st.markdown(md(f"**[{c.rank}] {doc}** — {where}"))
            st.text(c.text[:3000] + ("…" if len(c.text) > 3000 else ""))


def compact_title(doc: str) -> str:
    """'Tata Consultancy Services (TCS) Annual Report FY2025-26' -> 'TCS Annual Report FY2025-26'."""
    m = re.match(r"^.*\((\S+)\) (Annual Report FY\S+)$", doc)
    return f"{m.group(1)} {m.group(2)}" if m else doc


def trust_bar(res: dict) -> None:
    """What was verified about this answer, where it came from, and how long it took."""
    chips, report = [], res.get("report")
    if res.get("refusal"):
        chips.append(("info", "ⓘ Not covered by PortIQ's documents — it didn't guess"))
    elif res.get("citations") and report is None:
        chips.append(("", "Citation check off"))
    elif report is not None and report.uncited_answer:
        chips.append(("warn", "⚠ No source references in this answer"))
    elif report is not None and report.claims:
        ok, n = sum(c.status == "ok" for c in report.claims), len(report.claims)
        chips.append(("good" if ok == n else "warn",
                      f"{'✓' if ok == n else '⚠'} {ok} of {n} statement{'s match' if n != 1 else ' matches'} "
                      f"{'their sources' if n != 1 else 'its source'}"))
    if report is not None and not res.get("refusal"):
        if report.ungrounded_numbers:
            chips.append(("warn", f"⚠ {len(report.ungrounded_numbers)} figure(s) not found in the sources"))
        elif re.search(r"\d{2,}", res["answer"]):
            chips.append(("good", "✓ All figures found in the sources"))
    docs = list(dict.fromkeys(compact_title(source_info(c)[1]) for c in res.get("citations", [])))
    if docs and not res.get("refusal"):
        first = docs[0] if len(docs[0]) <= 52 else docs[0][:50] + "…"
        chips.append(("", f"📚 {first}" + (f" +{len(docs) - 1} more" if len(docs) > 1 else "")))
    if res.get("cached"):
        chips.append(("info", "⚡ Instant — asked before"))
    elif res.get("seconds"):
        chips.append(("", f"⏱ {res['seconds']:.0f} s"))
    if chips:
        st.markdown('<div class="chips">' + "".join(f'<span class="chip {cls}">{html.escape(text)}</span>'
                                                     for cls, text in chips) + "</div>", unsafe_allow_html=True)
    if report is not None and (report.problems or report.ungrounded_numbers or report.uncited_answer):
        with st.expander("What the check found"):
            render_report(report)


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
    render_retrieval_notes(res, show_details)
    render_positions(res["positions"])
    render_valuation(res.get("valuation"))
    if res.get("notice"):  # a daily limit stopped this one before the AI was asked
        st.info(res["notice"])
    st.markdown(md(res["answer"]))
    if res["disclaimer"]:
        st.caption(res["disclaimer"])
    trust_bar(res)
    render_sources(res["citations"], refusal=res.get("refusal", False), show_details=show_details)


@st.dialog("Edit holdings", width="large")
@as_session_user
def edit_holdings_dialog() -> None:
    """Add, change or delete holdings; saved to data/portfolio/holdings.csv after validation."""
    from datetime import date
    st.caption("Add a row with the + at the bottom; select rows and press Delete to remove them. "
               + ("Saved to your private PortIQ account — only you can see it." if HOSTED else
                  "Saved only on this machine."))
    rows = [{"ticker": h["ticker"], "quantity": h["quantity"], "avg_cost": h["avg_cost"],
             "purchase_date": date.fromisoformat(h["purchase_date"]) if h["purchase_date"] else None}
            for h in load_manual_holdings()]  # synced broker holdings are refreshed by portfolio_sync.py
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


# ── Theme ────────────────────────────────────────────────────────────────────
# System fonts only: a web font would be fetched from the internet on every load.
st.markdown("""<style>
:root { --card: #121826; --line: #1f2937; --muted: #8b95a7; --up: #22c55e; --down: #ef4444; --accent: #6366f1;
        --brand: linear-gradient(90deg, #3b6cff 0%, #7c3aed 100%); }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1600px; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Inter, sans-serif; }
.topbar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.brand { display: flex; align-items: center; gap: 10px; }
.brand img { width: 34px; height: 34px; display: block; }
.brand .word { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; }
.brand .iq { background: var(--brand); -webkit-background-clip: text; background-clip: text; color: transparent; }
.brand small { display: block; color: var(--muted); font-weight: 500; font-size: 0.72rem; letter-spacing: .02em; margin-top: 3px; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
        border: 1px solid var(--line); background: var(--card); color: var(--muted); font-size: 0.78rem; white-space: nowrap; }
.pill.open { color: var(--up); border-color: rgba(34,197,94,.35); }
.pill.closed { color: #f59e0b; border-color: rgba(245,158,11,.35); }
.pane-title { font-size: 0.78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin: 0; }
/* Cards wrap to 2 per row when the pane is narrow — an amount is never cut off */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin: 4px 0 10px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; min-width: 0; }
.kpi .label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: .06em; }
.kpi .value { font-size: 1.2rem; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.kpi .sub { font-size: 0.8rem; margin-top: 2px; color: var(--muted); font-variant-numeric: tabular-nums; }
.up { color: var(--up) !important; } .down { color: var(--down) !important; } .warn { color: #f59e0b !important; }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 6px; }
@media (max-width: 900px) { .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.stat { border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; }
.stat .label { color: var(--muted); font-size: 0.72rem; } .stat .value { font-weight: 600; font-variant-numeric: tabular-nums; }
.footnote { color: var(--muted); font-size: 0.75rem; }
.range { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; margin: 8px 0; }
.range-title { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 10px; }
.range-bar { position: relative; height: 10px; border-radius: 5px; background: var(--line); }
.range-bar .band { position: absolute; top: 0; bottom: 0; border-radius: 5px; background: var(--accent); opacity: .55; }
.range-bar .now { position: absolute; top: -4px; bottom: -4px; width: 2px; margin-left: -1px; background: #e5e7eb; }
.range-ends { display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 6px;
              font-variant-numeric: tabular-nums; color: var(--muted); }
.range-ends .mid { color: #e5e7eb; font-weight: 600; }
div[data-testid="stTabs"] button p { font-size: 0.9rem; }
/* ── Answer trust bar and source cards ── */
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; }
.chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; border-radius: 999px; font-size: 0.76rem;
        border: 1px solid var(--line); background: var(--card); color: var(--muted); }
.chip.good { color: var(--up); border-color: rgba(34,197,94,.35); background: rgba(34,197,94,.07); }
.chip.warn { color: #f59e0b; border-color: rgba(245,158,11,.4); background: rgba(245,158,11,.07); }
.chip.info { color: #93c5fd; border-color: rgba(147,197,253,.35); background: rgba(59,130,246,.07); }
.route-chip { display: inline-block; font-size: 0.72rem; letter-spacing: .04em; text-transform: uppercase;
              color: var(--muted); margin-bottom: 4px; }
.srcs { display: grid; gap: 6px; margin: 6px 0; }
.src { display: flex; gap: 10px; align-items: flex-start; padding: 8px 10px; border: 1px solid var(--line);
       border-radius: 10px; background: rgba(255,255,255,.015); }
.src .n { flex: none; width: 20px; height: 20px; border-radius: 6px; background: rgba(99,102,241,.18); color: #c7d2fe;
          font-size: 0.72rem; display: flex; align-items: center; justify-content: center; font-weight: 600; }
.src .t { font-size: 0.84rem; font-weight: 600; line-height: 1.3; }
.src .t a { color: inherit; text-decoration: none; } .src .t a:hover { text-decoration: underline; }
.src .s { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
.disclaimer { color: var(--muted); font-size: 0.72rem; text-align: center; margin-top: 4px; }
/* ── Insights ── */
.insight { display: flex; gap: 10px; padding: 9px 12px; border-radius: 10px; border: 1px solid var(--line);
           background: var(--card); margin-bottom: 6px; font-size: 0.86rem; line-height: 1.4; }
.insight .dot { flex: none; width: 8px; height: 8px; border-radius: 50%; margin-top: 6px; background: #93c5fd; }
.insight.good .dot { background: var(--up); } .insight.warn .dot { background: #f59e0b; }
div[data-testid="stCheckbox"] p, div[data-testid="stPopover"] button p { white-space: nowrap; }  /* "Live", "＋ Add" */
</style>""", unsafe_allow_html=True)

LIVE_EVERY_SEC = 60
UP, DOWN = "#22c55e", "#ef4444"
# Allocation colours: no red or green, which would read as loss or gain.
PALETTE = ["#3b6cff", "#7c3aed", "#06b6d4", "#a78bfa", "#14b8a6", "#f59e0b", "#6366f1", "#ec4899", "#0ea5e9", "#eab308"]
SUGGESTIONS = ["How is my portfolio doing?", "What risks does Reliance disclose in its annual report?",
               "What is a SIP and how does compounding help it?", "Compare TCS and Infosys margins"]


def logo_src() -> str:
    """The logo as a data URI, so the page loads it without any network request."""
    import base64
    mime = "image/svg+xml" if LOGO.suffix == ".svg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(LOGO.read_bytes()).decode()}"


@st.cache_data(show_spinner=False)
def corpus_counts() -> dict[str, int]:
    """How many documents of each kind PortIQ answers from (read from data/raw, not hard-coded)."""
    raw = Path(__file__).parent / "data" / "raw"
    names = [m.name for m in raw.glob("tier*/*.meta.json") if not m.name.startswith("._")]
    sec = [n for n in names if n.startswith("tier2_secedgar_")]
    return {"annual": sum(n.startswith("tier2_nse_") for n in names), "sec": len(sec),
            "sec_companies": len({n.split("_")[2].split("-")[0] for n in sec}),
            "explainers": sum(n.startswith("tier1_") for n in names),
            "papers": sum(n.startswith("tier3_") for n in names)}


def privacy_summary() -> str:
    if HOSTED:
        return ("Your holdings are kept in your private PortIQ account (only you can see them; delete them any time "
                "in ⚙ Settings). Questions, the passages found and the holdings you ask about go to this site's AI "
                "model server to write the answer; live prices send only stock symbols to Yahoo Finance.")
    return ("Runs on this machine (cloud models refused, no telemetry). Your holdings file stays here; live prices "
            "send only stock symbols to Yahoo Finance.")


def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:  # no secrets.toml at all
        return False


def brand_header() -> None:
    st.markdown(f'<div class="topbar"><div class="brand"><img src="{logo_src()}" alt=""><div><div class="word">'
                'Port<span class="iq">IQ</span></div><small>Upload. Ask. Get Insights.</small></div></div></div>',
                unsafe_allow_html=True)


def sign_in_page() -> None:
    """Hosted: what PortIQ is, and Google sign-in. Nothing personal is shown or read before this."""
    brand_header()
    left, _ = st.columns([1.4, 1])
    with left:
        st.markdown("### Your portfolio, explained from the companies' own reports\n"
                    "- Upload your monthly **CAS statement** — every demat account, every broker, in one PDF\n"
                    "- Live value and P&L in ₹, with insights computed by code, not guessed\n"
                    "- Ask about a company's annual report or filings — every answer cites its sources")
        st.button("Sign in with Google", type="primary", on_click=st.login)
        st.caption("Sign-in keeps your portfolio private to you. Educational information — not investment advice.")


def privacy_notice(ws: workspace.Workspace) -> None:
    """Shown once (and again when the notice changes) before anything can be added."""
    api = backends.api_config()
    model_host = f" ({api['host']})" if api and api.get("host") else ""
    host = f" ({os.environ['PORTIQ_HOST_NAME']})" if os.environ.get("PORTIQ_HOST_NAME") else ""
    store = os.environ.get("PORTIQ_STORE_NAME", "")
    store_line = (" and, encrypted with a key only this site holds, in its database" + (f" ({html.escape(store)})" if store
                  else "") + " — so the database provider can't read it." if remote_store.enabled() else ".")
    brand_header()
    st.markdown(f"""
### Before you add your portfolio

**What PortIQ keeps** — the holdings you add or import (symbol, quantity, average price, when imported) and copies
of broker CSV files you upload. **Not kept:** your CAS PDF (deleted right after it is read), its password (your PAN),
or the name, address and account numbers printed on it.

**Where** — in a private folder for your account on the server that runs this site{html.escape(host)}{store_line}
Only your signed-in account can see it. From Google sign-in PortIQ stores only an account ID, as a one-way code — not your
e-mail or name (your e-mail is only shown to you while you're signed in).

**When you ask a question** — your question, the passages PortIQ found and the details of holdings you ask about
are sent to the AI model server this site uses{html.escape(model_host)} to write the answer. Live prices send only
stock symbols to Yahoo Finance.

**Your control** — ⚙ Settings → *Download my data* or *Delete my data*, any time.

**Not investment advice** — PortIQ gives educational information from public documents. It is not a
SEBI-registered investment adviser and doesn't recommend buying or selling any security.
""")
    agree = st.checkbox("I've read this and agree", key="notice_agree")
    if st.button("Continue", type="primary", disabled=not agree):
        ws.record_consent()
        st.rerun()


def export_zip(ws: workspace.Workspace) -> bytes:
    """Download my data: everything PortIQ keeps for this account, readable without PortIQ."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if ws.holdings.exists():
            z.write(ws.holdings, "holdings.csv")
        z.writestr("imported_holdings.json", json.dumps(portfolio_db.latest_snapshots(ws.db), indent=2, default=str))
        for f in sorted(ws.imports.glob("*.csv")) if ws.imports.exists() else []:
            z.write(f, f"uploaded_csv/{f.name}")
        if ws.consent_file.exists():
            z.write(ws.consent_file, "privacy_notice_accepted.json")
    return buf.getvalue()


def account_controls(ws: workspace.Workspace, signed_in_as: str, dev_login: bool) -> None:
    st.markdown(f"**Signed in** as {html.escape(signed_in_as)}")
    st.download_button("⬇ Download my data", data=export_zip(ws), file_name="portiq-my-data.zip",
                       mime="application/zip", width="stretch")
    with st.expander("Delete my data"):
        st.caption("Permanently deletes your holdings, imports and settings from PortIQ. Your broker account and "
                   "the files on your computer are untouched.")
        confirm = st.text_input("Type DELETE to confirm", key="delete_confirm")
        if st.button("Delete everything", type="primary", disabled=confirm.strip() != "DELETE", width="stretch"):
            ws.delete_everything()
            remote_store.forget(ws.user_key)
            usage.forget(ws.user_key)
            cache = answer_cache()
            for key in [k for k in cache if k[0] == ws.user_key]:
                cache.pop(key, None)
            for key in list(st.session_state):
                del st.session_state[key]
            if dev_login:
                st.rerun()
            else:
                st.logout()
    if not dev_login:
        st.button("Sign out", on_click=st.logout, width="stretch")


def how_it_works() -> None:
    n = corpus_counts()
    st.markdown(f"""
**Where answers come from**
- {n['annual']} Indian annual reports (FY2025-26, from NSE) · {n['sec']} SEC filings of {n['sec_companies']} US companies
  · {n['explainers']} finance explainers · {n['papers']} research papers
- The model may only use the passages PortIQ finds in these documents, and must cite each one as **[n]**.
  If they don't cover a question, it says so instead of guessing.

**How each answer is checked**
- Every cited statement is compared with its passage, and every figure is looked up in the sources.
  The result is shown under the answer (✓ or ⚠), with the passages one click away.
- The check can't tell if a real figure is attached to the wrong line item — that's why the sources are shown.

**Computed by code, not AI**
- Portfolio values, P&L, allocation and all Insights — from your holdings and Yahoo Finance prices.

**Privacy**
- {privacy_summary()}

**Not investment advice**
- PortIQ gives educational information from public documents. It is not a SEBI-registered investment
  adviser and doesn't recommend buying or selling any security.
""")


def market_status() -> tuple[str, bool]:
    """NSE regular session 09:15–15:30 IST, Monday–Friday (exchange holidays not included)."""
    from datetime import datetime, time as dtime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    is_open = now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(15, 30)
    return f"NSE {'open' if is_open else 'closed'}", is_open  # no clock: the bar only redraws on a full rerun


def ask(question: str) -> None:
    """Queue a question for the assistant (used by suggestion and stock-view buttons)."""
    if st.session_state.get("pending") is None:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.pending = question


def tone(x) -> str:
    return "" if x is None else ("up" if x >= 0 else "down")


def kpi(label: str, value: str, sub: str = "", cls: str = "", sub_cls: str | None = None) -> str:
    sub_cls = cls if sub_cls is None else sub_cls
    return (f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
            f'<div class="value {cls}">{html.escape(value)}</div><div class="sub {sub_cls}">{html.escape(sub)}</div></div>')


def portfolio_frame(holdings: list[dict], quotes: dict[str, dict]):
    """Per-holding table (numbers in ₹ for totals) plus ₹ totals — all arithmetic here, never by the model."""
    import pandas as pd
    fx = (quotes.get(prices.FX_TICKER) or {}).get("price")
    rows, tot = [], {"value": 0.0, "cost": 0.0, "value_costed": 0.0, "costed": 0, "day": 0.0, "prev": 0.0,
                     "missing": [], "no_fx": [], "no_cost": []}
    for h in holdings:
        symbol, exchange = prices.display_symbol(h["ticker"])
        cur = prices.currency(h["ticker"])
        rate = 1.0 if cur == "INR" else fx
        q = quotes.get(prices.normalize_ticker(h["ticker"]))
        src = h.get("source", "manual")
        row = {"Stock": symbol, "Exch": exchange,
               # short so the column stays narrow; full source + as-of are in Import → Synced sources and in answers
               "Source": short_source(src),
               "Qty": h["quantity"], "Avg": h["avg_cost"], "LTP": None,
               "Day %": None, "Invested ₹": None, "Current ₹": None, "P&L ₹": None, "P&L %": None, "cur": cur,
               "ticker": h["ticker"]}
        if q is None or rate is None:
            (tot["missing"] if q is None else tot["no_fx"]).append(symbol)
        else:
            value = h["quantity"] * q["price"] * rate
            cost = None if h["avg_cost"] is None else h["quantity"] * h["avg_cost"] * rate  # None: e.g. from a CAS
            prev = q.get("prev_close")
            tot["value"] += value
            if cost is None:
                tot["no_cost"].append(symbol)
            else:
                tot["cost"] += cost
                tot["value_costed"] += value
                tot["costed"] += 1
            if prev:
                tot["day"] += h["quantity"] * (q["price"] - prev) * rate
                tot["prev"] += h["quantity"] * prev * rate
            row.update({"LTP": q["price"], "Day %": (q["price"] / prev - 1) * 100 if prev else None,
                        "Invested ₹": cost, "Current ₹": value, "P&L ₹": None if cost is None else value - cost,
                        "P&L %": (value - cost) / cost * 100 if cost else None})
        rows.append(row)
    tot["fx"] = fx if any(prices.currency(h["ticker"]) == "USD" for h in holdings) else None
    return pd.DataFrame(rows), tot


def holdings_table(df) -> None:
    def money_native(row, col):
        v = row[col]
        return "—" if v is None or v != v else prices.money(v, row["cur"])

    def colour(v):
        return "" if v is None or v != v else f"color: {UP if v >= 0 else DOWN}"

    def dash_or(fmt):
        return lambda v: "—" if v is None or v != v else fmt(v)

    # Display text, not numbers: Streamlit draws an empty number as "None" whatever a Styler formatter returns,
    # and holdings without a price or an average cost (e.g. from a CAS) have empty cells.
    view = df.copy()
    view["Avg"] = [money_native(r, "Avg") for _, r in df.iterrows()]
    view["LTP"] = [money_native(r, "LTP") for _, r in df.iterrows()]
    for col, fmt in (("Day %", lambda v: f"{v:+.2f}%"), ("P&L %", lambda v: f"{v:+.2f}%"), ("Invested ₹", prices.inr),
                     ("Current ₹", prices.inr), ("P&L ₹", prices.signed_money)):
        view[col] = df[col].map(dash_or(fmt))
    view = view.drop(columns=["cur", "ticker", "Exch"])  # US stocks are the ones priced in $
    toned = ("Day %", "P&L ₹", "P&L %")  # coloured from the numbers in df

    styled = (view.style.format({"Qty": "{:g}"})
              .apply(lambda _: type(view)({c: [colour(v) for v in df[c]] if c in toned else [""] * len(df)
                                           for c in view.columns}, index=view.index), axis=None))
    st.dataframe(styled, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(view)))


def allocation_charts(df) -> None:
    import plotly.graph_objects as go
    priced = df[df["Current ₹"].notna()]
    if priced.empty:
        st.caption("No prices yet.")
        return
    left, right = st.columns(2)
    donut = go.Figure(go.Pie(labels=priced["Stock"], values=priced["Current ₹"], hole=0.62, sort=True,
                             marker=dict(colors=PALETTE, line=dict(color="#0b0f17", width=2)),
                             textinfo="percent", hovertemplate="%{label}: ₹%{value:,.0f}<extra></extra>"))
    donut.update_layout(title=dict(text="Allocation by value", font=dict(size=13)), height=300,
                        margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="v", font=dict(size=11)),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    left.plotly_chart(donut, width="stretch", config={"displayModeBar": False})
    ordered = priced.sort_values("P&L %")
    bars = go.Figure(go.Bar(x=ordered["P&L %"], y=ordered["Stock"], orientation="h",
                            marker_color=[UP if v >= 0 else DOWN for v in ordered["P&L %"]],
                            hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
    bars.update_layout(title=dict(text="Return on cost", font=dict(size=13)), height=300,
                       margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(ticksuffix="%", gridcolor="#1f2937"),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    right.plotly_chart(bars, width="stretch", config={"displayModeBar": False})


@st.cache_data(show_spinner=False)
def stock_options() -> tuple[list[str], dict[str, str]]:
    """Dropdown labels for the add-stock search (NSE first, then US by size) and label -> ticker."""
    rows = stock_list.load_stocks()
    labels = [stock_list.label(r) for r in rows]
    return labels, dict(zip(labels, (r["ticker"] for r in rows)))


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(ticker: str, period: str = "1y") -> list[tuple[str, float]]:
    return prices.yahoo_history(ticker, period)


def range_card(ticker: str, cur: str) -> None:
    """Next 3 months, IF they move like this stock's past 3-month periods — a range, never a forecast."""
    try:
        r = prices.past_moves_range(cached_history(ticker, "5y"))
    except Exception:
        r = {}
    title = '<div class="range-title">Next 3 months — if they move like its past 3-month periods</div>'
    if not r:
        st.markdown(f'<div class="range">{title}<p class="footnote">Not enough price history yet (needs about 15 '
                    "months) to show a range from past moves.</p></div>", unsafe_allow_html=True)
        return
    span = r["high95"] - r["low95"] or 1.0
    at = lambda x: (x - r["low95"]) / span * 100
    m = lambda x: html.escape(prices.money(x, cur))
    (a, b), (c, d) = r["move68"], r["move95"]
    st.markdown(
        f'<div class="range">{title}'
        f'<div class="range-bar"><div class="band" style="left:{at(r["low68"]):.1f}%;width:{at(r["high68"]) - at(r["low68"]):.1f}%"></div>'
        f'<div class="now" style="left:{at(r["last"]):.1f}%" title="Today {m(r["last"])}"></div></div>'
        f'<div class="range-ends"><span>{m(r["low95"])}</span><span class="mid">{m(r["low68"])} – {m(r["high68"])}</span>'
        f'<span>{m(r["high95"])}</span></div>'
        f'<p class="footnote">In {r["periods"]:,} overlapping 3-month periods since {r["since"][:4]}, 2 in 3 moved between '
        f'{a:+.0f}% and {b:+.0f}% (shaded) and 19 in 20 between {c:+.0f}% and {d:+.0f}% (whole bar); {r["up_share"]:.0f}% '
        f'ended higher. Applied to today\'s {m(r["last"])} (the line). <b>Not a forecast</b> — the past doesn\'t repeat, '
        "and results, news or a market fall can move a price beyond any past range.</p></div>", unsafe_allow_html=True)


def stock_view(holdings: list[dict], pending) -> None:
    import plotly.graph_objects as go
    names = {prices.display_symbol(h["ticker"])[0]: h for h in holdings}
    pick = st.selectbox("Stock", list(names), label_visibility="collapsed", key="stock_pick")
    h = names[pick]
    cur = prices.currency(h["ticker"])
    if mutual_funds.is_mf(h["ticker"]):  # funds: one NAV a day from AMFI, no price history here yet
        q = mutual_funds.quotes([h["ticker"]]).get(h["ticker"].upper())
        avg = h["avg_cost"]
        ret = (q["price"] / avg - 1) * 100 if q and avg else None
        cells = [("Latest NAV", prices.money(q["price"]) if q else "—", ""), ("NAV date", q["as_of"] if q else "—", ""),
                 ("Your avg cost", prices.money(avg) if avg is not None else "—", ""),
                 ("Your return", f"{ret:+.1f}%" if ret is not None else "—", tone(ret))]
        st.markdown('<div class="stats">' + "".join(
            f'<div class="stat"><div class="label">{label}</div><div class="value {cls}">{html.escape(v)}</div></div>'
            for label, v, cls in cells) + "</div>", unsafe_allow_html=True)
        st.markdown('<p class="footnote">Mutual fund · NAV from AMFI, published once a day · price history and '
                    "charts for funds aren't available yet</p>", unsafe_allow_html=True)
        return
    try:
        history = cached_history(h["ticker"])
    except Exception as e:
        st.warning(f"Couldn't load price history: {e}")
        return
    if len(history) < 2:
        st.caption("Not enough price history.")
        return
    s = prices.price_stats(history)
    dates, closes = [d for d, _ in history], [c for _, c in history]
    colour = UP if closes[-1] >= closes[0] else DOWN
    fig = go.Figure(go.Scatter(x=dates, y=closes, mode="lines", line=dict(color=colour, width=2),
                               fill="tozeroy", fillcolor="rgba(34,197,94,0.08)" if colour == UP else "rgba(239,68,68,0.08)",
                               hovertemplate="%{x}: " + ("₹" if cur == "INR" else "$") + "%{y:,.2f}<extra></extra>"))
    avg = h["avg_cost"]  # None when the source has no purchase price (CAS)
    if avg is not None:
        fig.add_hline(y=avg, line_dash="dot", line_color="#8b95a7",
                      annotation_text=f"your avg {prices.money(avg, cur)}", annotation_position="top left",
                      annotation_font_color="#8b95a7")
    lo, hi = min(closes + ([avg] if avg is not None else [])), max(closes + ([avg] if avg is not None else []))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      yaxis=dict(range=[lo * 0.95, hi * 1.03], gridcolor="#1f2937"), xaxis=dict(showgrid=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    def pct(v):
        return "—" if v is None else f"{v:+.1f}%"
    cells = [("52-week high", prices.money(s["high_52w"], cur), ""), ("52-week low", prices.money(s["low_52w"], cur), ""),
             ("1 month", pct(s["ret_1m"]), tone(s["ret_1m"])), ("3 months", pct(s["ret_3m"]), tone(s["ret_3m"])),
             ("1 year", pct(s["ret_1y"]), tone(s["ret_1y"])), ("Volatility (ann.)", f"{s['volatility']:.1f}%", ""),
             ("Max drawdown", f"{s['max_drawdown']:.1f}%", "down" if s["max_drawdown"] < 0 else ""),
             ("Your return", pct((s["last"] / avg - 1) * 100 if avg else None),
              tone(s["last"] - avg) if avg else "")]
    st.markdown('<div class="stats">' + "".join(
        f'<div class="stat"><div class="label">{label}</div><div class="value {cls}">{html.escape(v)}</div></div>'
        for label, v, cls in cells) + "</div>", unsafe_allow_html=True)
    st.markdown('<p class="footnote">Past prices, not a forecast · Yahoo Finance daily closes</p>',
                unsafe_allow_html=True)
    range_card(h["ticker"], cur)
    a, b, c = st.columns(3)
    for col, label, q in [(a, "Key risks", f"What are the key risks for {pick}?"),
                          (b, "Revenue & margins", f"What were {pick}'s revenue and profit in the latest year?"),
                          (c, "Business outlook", f"What does {pick}'s management say about the outlook?")]:
        if col.button(label, key=f"q_{label}_{pick}", width="stretch",
                      disabled=pending is not None or assistant_problem is not None,
                      help=assistant_problem and "The research assistant is off — see the right-hand pane"):
            ask(q)
            st.rerun(scope="app")


@st.cache_data(ttl=1800, show_spinner=False)
def cached_histories(tickers: tuple[str, ...]) -> dict[str, list[tuple[str, float]]]:
    return prices.yahoo_histories(list(tickers), "1y")


def insights_view(holdings: list[dict], df, tot: dict) -> None:
    """Observations computed by code from your holdings and Yahoo prices — never by the model."""
    import plotly.graph_objects as go

    def num(v):
        return None if v is None or v != v else float(v)

    rows = [{"symbol": r["Stock"], "currency": r["cur"], "value_inr": num(r["Current ₹"]), "pnl_pct": num(r["P&L %"])}
            for r in df.to_dict("records")]
    alloc = insights.allocation(rows)
    periods = insights.holding_periods(holdings)
    held = [{**h, "ticker": prices.normalize_ticker(h["ticker"])} for h in holdings]
    try:
        hist = cached_histories(tuple(sorted({h["ticker"] for h in held} | set(insights.BENCHMARKS))))
    except Exception as e:
        hist = {}
        st.caption(f"Price history unavailable right now ({e}); showing today's figures only.")
    fx = {h["ticker"]: 1.0 if prices.currency(h["ticker"]) == "INR" else tot["fx"] for h in held
          if prices.currency(h["ticker"]) == "INR" or tot["fx"]}
    values = insights.portfolio_history(held, hist, fx)
    perf = insights.performance(values)
    bench = {name: insights.performance(hist[t])["return_pct"] for t, name in insights.BENCHMARKS.items()
             if insights.performance(hist.get(t, []))}
    lines = insights.headlines(alloc, perf, bench, periods)
    st.markdown("".join(f'<div class="insight {tone_}"><div class="dot"></div><div>{html.escape(text)}</div></div>'
                        for tone_, text in lines), unsafe_allow_html=True)
    funds = [h for h in holdings if mutual_funds.is_mf(h["ticker"])]
    if funds and perf:  # history comes from Yahoo, which has no Indian fund NAVs
        st.caption(f"The 1-year return, volatility and worst drop cover your stocks only — {len(funds)} mutual "
                   "fund(s) have no price history here yet. They are included in today's value and allocation.")
    if perf:
        sym, pct = alloc["largest"] if alloc else ("—", 0)
        st.markdown('<div class="kpis">'
                    + kpi("1-year return", f"{perf['return_pct']:+.1f}%",
                          " · ".join(f"{n} {v:+.1f}%" for n, v in bench.items()), tone(perf["return_pct"]), sub_cls="")
                    + kpi("Volatility", f"{perf['volatility']:.1f}%", "yearly, from daily moves")
                    + kpi("Worst drop", f"{perf['max_drawdown']:.1f}%", "from a high, past year",
                          "down" if perf["max_drawdown"] < -10 else "")
                    + kpi("Largest holding", f"{pct:.0f}%", sym, "warn" if pct >= insights.CONCENTRATION_FLAG else "")
                    + "</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[d for d, _ in values], y=[v / values[0][1] * 100 for _, v in values],
                                 name="Your holdings", line=dict(color="#6366f1", width=2.5)))
        for t, name in insights.BENCHMARKS.items():
            series = hist.get(t)
            if series:
                fig.add_trace(go.Scatter(x=[d for d, _ in series], y=[c / series[0][1] * 100 for _, c in series],
                                         name=name, line=dict(width=1.5, dash="dot")))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                          title=dict(text="Growth of ₹100 over the past year", font=dict(size=13)),
                          legend=dict(orientation="h", y=-0.15), yaxis=dict(gridcolor="#1f2937"),
                          xaxis=dict(showgrid=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<p class="footnote">Computed by PortIQ from your holdings and Yahoo Finance daily closes '
                "(today's holdings over the past year; US stocks at today's ₹/$). Describes the past — not a "
                "forecast or a recommendation.</p>", unsafe_allow_html=True)


# ── Import & sync (Day 21) ───────────────────────────────────────────────────
# Absolute paths: Streamlit may be started from another directory (`streamlit run finrag/app.py`),
# and the sync runs in a subprocess with its own working directory.
APP_DIR = Path(__file__).resolve().parent
SYNC_SCRIPT = APP_DIR / "scripts" / "portfolio_sync.py"
def imports_dir() -> Path:
    """Uploaded CSV copies: the signed-in user's folder (hosted), else data/portfolio/imports (git-ignored)."""
    ws = workspace.current()
    return ws.imports if ws else APP_DIR / "data" / "portfolio" / "imports"


def mappings_dir() -> Path:
    """Saved column choices: per user (hosted), else config/broker_mappings (*.local.json git-ignored)."""
    ws = workspace.current()
    return ws.mappings if ws else APP_DIR / "config" / "broker_mappings"
NOT_IN_FILE = "(not in this file)"
BROKERS = {
    "upstox": {"label": "Upstox", "env": ["UPSTOX_ACCESS_TOKEN"], "optional": [],
               "setup": "1. Create an app on the Upstox developer portal and generate an **access token** "
                        "(it expires at 3:30 AM each day).\n2. Put it in `finrag/.env` as `UPSTOX_ACCESS_TOKEN=…` "
                        "(copy `.env.example`).\n3. Come back here and press **Sync now**."},
    "dhan": {"label": "Dhan", "env": ["DHAN_ACCESS_TOKEN"], "optional": ["DHAN_CLIENT_ID"],
             "setup": "1. On web.dhan.co open **Access DhanHQ APIs** and generate an access token (valid 24 hours).\n"
                      "2. Put it in `finrag/.env` as `DHAN_ACCESS_TOKEN=…`, and your client ID as `DHAN_CLIENT_ID=…` "
                      "(needed for prices).\n3. Come back here and press **Sync now**."},
    "groww": {"label": "Groww", "env": ["GROWW_ACCESS_TOKEN"], "optional": [],
              "setup": "1. On groww.in open **Profile → Trading APIs** and generate an access token (it expires at "
                       "6:00 AM each day).\n2. Put it in `finrag/.env` as `GROWW_ACCESS_TOKEN=…`.\n"
                       "3. Come back here and press **Sync now**."},
}


def run_sync_cli(*args: str, stdin: str | None = None):
    """
    Run portfolio_sync.py in its own process and read its --json output. A sync limits the network
    to one broker host for the whole process — in a separate process that can't touch the app's own
    live prices. Credentials stay in .env, and a CAS password goes through stdin: nothing secret is
    ever on the command line (where other programs on the machine could see it).
    """
    ws = workspace.current()
    if ws:  # hosted: the subprocess writes only to this user's database
        args = (*args, "--db", str(ws.db))
    try:
        out = subprocess.run([sys.executable, str(SYNC_SCRIPT), *args, "--json"], capture_output=True, text=True,
                             timeout=180, cwd=APP_DIR, input=stdin)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_type": "Timeout", "message": "the sync took longer than 3 minutes"}
    lines = [line for line in out.stdout.strip().splitlines() if line.strip()]
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError):
        return {"ok": False, "error_type": "SyncError", "message": (out.stderr or out.stdout).strip()[-400:]}


def short_source(src: str) -> str:
    """For the narrow Source column: Manual · zerodha (a named CSV import) · CAS · CAS (mom) · Upstox."""
    kind, _, name = src.partition(":")
    if src == "manual":
        return "Manual"
    if kind == "csv" and name:
        return name
    if kind == "cas":
        return f"CAS ({name})" if name else "CAS"
    return source_label(src)


def import_copies(label: str) -> list[Path]:
    """This import's stored uploads, oldest first — exact label only, so 'zerodha' never matches 'zerodha-holdings'."""
    pattern = re.compile(rf"{re.escape(label)}-\d{{8}}-\d{{6}}\.csv")
    return sorted(p for p in imports_dir().glob(f"{label}-*.csv") if pattern.fullmatch(p.name))


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def csv_import_ui() -> None:
    up = st.file_uploader("Broker export (CSV)", type=["csv"],
                          help="Most brokers let you download your holdings as CSV or Excel — save it as CSV. "
                               + ("PortIQ keeps a copy in your private folder (removed with Delete my data)."
                                  if HOSTED else "The file stays on this machine (data/portfolio/imports/, "
                                                 "git-ignored)."))
    if not up:
        st.caption("Upload a holdings export to preview it. Nothing is imported until you press Import.")
        return
    raw = up.getvalue()
    text = _decode(raw)
    default = re.sub(r"[^a-z0-9]+", "-", Path(up.name).stem.lower()).strip("-")[:24] or "import"
    c1, c2 = st.columns([2, 1])
    label = re.sub(r"[^a-z0-9-]+", "-", c1.text_input("Name this import", value=default, max_chars=24,
                                                       help="Keeps exports from different brokers apart, e.g. "
                                                            "zerodha. Re-importing the same name replaces it.")
                   .lower()).strip("-") or "import"
    saved_path = mappings_dir() / f"{label}.local.json"
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    skip = int(c2.number_input("Lines before the header", 0, 50, value=int(saved.get("skip_lines", 0)),
                               help="Some exports start with a title or account details above the column names."))
    headers = read_headers(text, skip)
    if not headers:
        st.error("No column names found — check 'Lines before the header'.")
        return
    guess = guess_mapping(headers)
    if saved:  # the columns chosen last time for this import name, where they still exist
        guess.update({f: col for f, col in saved.get("columns", {}).items() if col in headers})
    st.markdown("**Which column is which?** — pre-filled from the file's headers; change anything that's wrong.")
    options = [NOT_IN_FILE] + headers
    pick = {}
    cols = st.columns(3)
    for i, (field_name, title) in enumerate([("symbol", "Symbol *"), ("quantity", "Quantity *"),
                                             ("avg_cost", "Average buy price *"), ("exchange", "Exchange"),
                                             ("isin", "ISIN")]):
        current = guess.get(field_name)
        pick[field_name] = cols[i % 3].selectbox(title, options, index=options.index(current) if current in options else 0,
                                                 key=f"map_{label}_{field_name}")
    default_exchange = cols[2].selectbox("Exchange if not in the file", ["NSE", "BSE", "US"],
                                         index=["NSE", "BSE", "US"].index(saved.get("defaults", {}).get("exchange", "NSE")),
                                         key=f"map_{label}_exch")
    missing = [f for f in ("symbol", "quantity", "avg_cost") if pick[f] == NOT_IN_FILE]
    if missing:
        st.warning("Choose the column for: " + ", ".join(missing))
        return
    mapping = {"name": label, "columns": {f: c for f, c in pick.items() if c != NOT_IN_FILE},
               "defaults": {"exchange": default_exchange, "currency": "USD" if default_exchange == "US" else "INR"},
               "skip_lines": skip, "delimiter": ","}
    try:
        report = parse_rows(text, mapping)
    except MalformedResponse as e:
        st.error(str(e))
        return
    st.markdown(f"**{len(report.holdings)} holding(s) will be imported**")
    if report.holdings:
        st.dataframe([{"Symbol": h.symbol, "Exchange": h.exchange, "Qty": h.quantity, "Avg cost": round(h.avg_cost, 2),
                       "Priced as": h.ticker} for h in report.holdings], hide_index=True, width="stretch",
                     height=min(260, 38 + 35 * len(report.holdings)))
    for n, why in report.merged:
        st.caption(f"Line {n}: merged — {why}")
    if report.skipped:
        with st.expander(f"⚠ {len(report.skipped)} row(s) will be skipped — see why", expanded=True):
            st.markdown("\n".join(f"- line {n}: {html.escape(why)}" for n, why in report.skipped))
    previous = next((s for s in portfolio_db.latest_snapshots() if s["source"] == f"csv:{label}"), None)
    if previous:
        st.caption(f"Replaces your previous '{label}' import ({previous['holdings_count']} holdings, "
                   f"{previous['fetched_at']}). Other sources are untouched.")
    if st.button(f"Import {len(report.holdings)} holding(s)", type="primary", disabled=not report.holdings,
                 key=f"import_{label}"):
        imports_dir().mkdir(parents=True, exist_ok=True)
        mappings_dir().mkdir(parents=True, exist_ok=True)
        file_path = imports_dir() / f"{label}-{datetime.now():%Y%m%d-%H%M%S}.csv"
        file_path.write_bytes(raw)
        saved_path.write_text(json.dumps(mapping, indent=2))
        res = run_sync_cli("--source", "csv", "--file", str(file_path), "--mapping", str(saved_path), "--label", label)
        if res.get("ok"):
            for old in import_copies(label)[:-3]:  # keep this import's last 3 uploads
                old.unlink()
            st.session_state.live_quotes = None
            st.session_state.price_note = f"Imported {res['holdings_count']} holding(s) as '{label}'"
            st.rerun()
        else:
            st.error(f"Import failed ({res.get('error_type')}): {res.get('message')}")


def cas_import_ui() -> None:
    from brokers.cas_adapter import PASSWORD_HINT, holdings_from_cas, read_statement  # PDF library loads only here
    st.caption("Your **Consolidated Account Statement (CAS)** from NSDL or CDSL lists the shares in every demat "
               "account under your PAN — whichever brokers you use (Zerodha, Groww, Upstox, …) — in one PDF. It is "
               "e-mailed to you every month, and NSDL's and CDSL's websites let you download it again. PortIQ reads "
               + ("it on its server, deletes the PDF right after, " if HOSTED else "it on this computer ")
               + "and keeps only the holdings — not your name, PAN, address or account numbers. "
               "**Mutual funds** held outside demat: upload the CAS from CAMS or KFintech too — it includes what "
               "you invested.")
    up = st.file_uploader("CAS statement (PDF)", type=["pdf"], key="cas_upload")
    if up is None:
        return
    c1, c2 = st.columns([1.2, 1])
    password = c1.text_input("PDF password", type="password", key="cas_password",
                             help=f"NSDL and CDSL lock the PDF with {PASSWORD_HINT}; a CAMS or KFintech statement "
                                  "usually with the password you chose when requesting it. It is used only to open "
                                  "the file — never saved or logged.")
    label = re.sub(r"[^a-z0-9-]+", "-", c2.text_input(
        "Whose statement? (optional)", max_chars=24, key="cas_label",
        help="Only needed if you import several people's statements, e.g. 'mom'.").lower()).strip("-")
    if not password:
        st.caption(f"Enter the password to read the statement — {PASSWORD_HINT}.")
        return
    raw = up.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    cached = st.session_state.get("cas_read")
    if not cached or cached[0] != digest:
        try:
            with st.spinner("Reading the statement…"):
                report = holdings_from_cas(read_statement(io.BytesIO(raw), password))
        except BrokerError as e:
            msg = str(e).removeprefix("cas: ")
            st.error(msg[:1].upper() + msg[1:])
            return
        st.session_state.cas_read = cached = (digest, report)  # holdings only — the statement itself isn't kept
    report = cached[1]
    funds = report.account_word == "folio(s)"
    if funds:  # a fund statement is its own source, so it never replaces your demat (shares) statement
        label = f"{label}-funds" if label else "funds"
    source = f"cas:{label}" if label else "cas"
    previous = next((s for s in portfolio_db.latest_snapshots() if s["source"] == source), None)
    known = {h["isin"]: h["avg_cost"] for h in (previous or {}).get("holdings", [])
             if h["isin"] and h["avg_cost"] is not None}
    st.markdown(f"**{len(report.holdings)} holding(s) found** · {report.issuer} statement for {report.period[0]} "
                f"to {report.period[1]} · {report.accounts} {report.account_word}")
    edited = None
    if report.holdings:
        import pandas as pd
        table = pd.DataFrame([{"Holding": mutual_funds.display_name(h.symbol) if h.exchange == "MF" else h.symbol,
                               "Quantity": h.quantity,
                               "Statement price": report.prices[h.symbol].last_price if h.symbol in report.prices else None,
                               "Avg buy price": h.avg_cost if h.avg_cost is not None else known.get(h.isin),
                               "ISIN": h.isin} for h in report.holdings])
        table["Avg buy price"] = table["Avg buy price"].astype(float)  # blank cells, not "None"
        if all(h.avg_cost is not None for h in report.holdings):
            st.caption("Average buy prices come from the statement — change any that look wrong.")
        else:
            st.caption("The statement doesn't say what you paid for some holdings. Enter your **average buy price** "
                       "to see P&L for them, or leave it blank — the holding is still valued, but left out of "
                       "Invested and P&L."
                       + (" Prices from your previous import of this statement are filled in — check them if you "
                          "bought or sold since." if known else ""))
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key=f"cas_costs_{digest[:12]}",
            disabled=["Holding", "Quantity", "Statement price", "ISIN"], height=min(320, 38 + 35 * len(table)),
            column_config={"Quantity": st.column_config.NumberColumn(help="Shares, or units of a fund", format="%g"),
                           "Statement price": st.column_config.NumberColumn(
                               "Statement price (₹)", format="%.2f",
                               help="Market price or fund NAV on the statement's valuation date"),
                           "Avg buy price": st.column_config.NumberColumn(
                               "Avg buy price (₹)", min_value=0.0, format="%.2f",
                               help="What you paid per share or unit, on average — your broker's or fund "
                                    "house's holdings page shows it")})
    for note in report.merged:
        st.caption(f"Merged — {note}")
    if report.skipped:
        with st.expander(f"⚠ {len(report.skipped)} item(s) not imported — see why", expanded=not report.holdings):
            st.markdown("\n".join(f"- {html.escape(why)}" for why in report.skipped))
    for why in report.warnings:
        st.caption(f"⚠ {why}")
    if previous:
        st.caption(f"Replaces your previous {source_label(source)} import ({previous['holdings_count']} holdings, "
                   f"{previous['fetched_at']}). Other sources are untouched.")
    if edited is not None and st.button(f"Import {len(report.holdings)} holding(s)", type="primary",
                                        key=f"cas_import_{digest[:12]}"):
        costs = {row["ISIN"]: float(row["Avg buy price"]) for row in edited.to_dict("records")
                 if row["Avg buy price"] is not None and row["Avg buy price"] == row["Avg buy price"]}  # skip blank/NaN
        with tempfile.TemporaryDirectory(prefix="portiq-cas-") as d:  # the PDF exists only while it is read
            pdf, costs_path = Path(d) / "statement.pdf", Path(d) / "costs.json"
            pdf.write_bytes(raw)
            costs_path.write_text(json.dumps(costs))
            res = run_sync_cli("--source", "cas", "--file", str(pdf), "--costs", str(costs_path), "--password-stdin",
                               *(["--label", label] if label else []), stdin=password + "\n")
        if res.get("ok"):
            st.session_state.pop("cas_read", None)
            st.session_state.live_quotes = None
            st.session_state.price_note = f"Imported {res['holdings_count']} holding(s) from your CAS statement"
            st.rerun()
        else:
            st.error(f"Import failed ({res.get('error_type')}): {res.get('message')}")


def brokers_ui() -> None:
    st.caption("Read-only: PortIQ only reads your holdings and prices, using your own tokens from finrag/.env. "
               "A sync contacts only that broker; tokens are never shown, logged or stored in the database.")
    snaps = {s["source"]: s for s in portfolio_db.latest_snapshots()}
    for name, b in BROKERS.items():
        configured = all(get_secret(v) for v in b["env"])  # reads .env locally — no network call
        last = portfolio_db.last_sync(portfolio_db.DB_PATH, name)
        snap = snaps.get(name)
        with st.container(border=True):
            c1, c2, c3 = st.columns([2.4, 1, 1], vertical_alignment="center")
            state = "🔑 token in .env" if configured else "⚪ not set up"
            c1.markdown(f"**{b['label']}** · {state}")
            lines = [f"Snapshot: {snap['holdings_count']} holdings, as of {snap['fetched_at']}" if snap else "No snapshot yet"]
            if last:
                lines.append(f"Last sync: {last['status']}{' — ' + last['error_type'] if last['error_type'] else ''} "
                             f"({last['started_at']})")
            c1.caption(" · ".join(lines))
            if not configured:
                with c1.expander("How to connect"):
                    st.markdown(b["setup"])
                continue
            missing_optional = [v for v in b["optional"] if not get_secret(v)]
            if missing_optional:
                c1.caption(f"Prices need {', '.join(missing_optional)} in .env — holdings sync without it.")
            if c2.button("Check", key=f"check_{name}", width="stretch", help="Is the token still accepted?"):
                rows = run_sync_cli("--status", "--source", name)
                st.session_state[f"auth_{name}"] = rows[0]["auth"] if isinstance(rows, list) and rows else \
                    f"unknown ({rows.get('message', '') if isinstance(rows, dict) else ''})"
            if c3.button("Sync now", key=f"sync_{name}", type="primary", width="stretch"):
                with st.spinner(f"Reading your {b['label']} holdings…"):
                    res = run_sync_cli("--source", name)
                if res.get("ok"):
                    st.session_state.live_quotes = None
                    st.session_state.price_note = (f"{b['label']}: {res['holdings_count']} holding(s) synced · "
                                                   f"prices {res['prices_status']}")
                    st.rerun()
                elif res.get("error_type") in ("AuthExpired", "CredentialsMissing"):
                    st.warning(f"🔑 {res.get('message')}")
                else:
                    st.error(f"Sync failed ({res.get('error_type')}): {res.get('message')} — your last snapshot is unchanged.")
            auth = st.session_state.get(f"auth_{name}")
            if auth:
                (st.success if auth == "ok" else st.warning)(f"Connection check: {auth}"
                                                             + (" — generate a new token and update .env"
                                                                if auth == "expired" else ""))


def sources_ui() -> None:
    manual = load_manual_holdings()
    st.markdown(f"**Your holdings file** · {len(manual)} holding(s) — change them with **Edit** or **＋ Add**.")
    snaps = portfolio_db.latest_snapshots()
    if not snaps:
        st.caption("No imported or synced sources yet.")
        return
    for s in snaps:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f"**{source_label(s['source'])}** · {s['holdings_count']} holding(s) · as of {s['fetched_at']}")
            if s["source"].startswith("csv"):
                c1.caption("Prices: live from Yahoo Finance, like your other holdings (exports carry no live prices)")
            elif s["source"].startswith("cas"):
                dated = next((h["price_as_of"] for h in s["holdings"] if h["price_as_of"]), None)
                no_cost = sum(h["avg_cost"] is None for h in s["holdings"])
                c1.caption("Prices: live from Yahoo Finance"
                           + (f"; the statement's own prices (as of {dated}) are the fallback" if dated else "")
                           + (f" · average price unknown for {no_cost} holding(s)" if no_cost else ""))
            else:
                c1.caption(f"Prices from {source_label(s['source'])}: {s['prices_status']}")
            if c2.button("Remove", key=f"remove_{s['source']}", width="stretch",
                         help="Remove this source's holdings from PortIQ, with PortIQ's stored copy of an uploaded "
                              "file (your original file and broker account are untouched)"):
                res = run_sync_cli("--remove", s["source"])
                if res.get("ok") is False:
                    st.error(f"Couldn't remove it ({res.get('error_type')}): {res.get('message')}")
                    return
                if s["source"].startswith("csv:"):
                    for copy in import_copies(s["source"][4:]):
                        copy.unlink()
                st.session_state.live_quotes = None
                st.session_state.price_note = f"Removed {source_label(s['source'])}"
                st.rerun()


@st.dialog("Import & sync holdings", width="large")
@as_session_user
def import_sync_dialog() -> None:
    names = ["CAS statement", "Upload CSV"] + ([] if HOSTED else ["Brokers"]) + ["Synced sources"]
    tabs = dict(zip(names, st.tabs(names)))  # website: no Brokers tab — the server's tokens are never a visitor's
    with tabs["CAS statement"]:
        cas_import_ui()
    with tabs["Upload CSV"]:
        csv_import_ui()
    if "Brokers" in tabs:
        with tabs["Brokers"]:
            brokers_ui()
    with tabs["Synced sources"]:
        sources_ui()


def portfolio_pane(pending) -> None:
    head = st.columns([1.7, 0.9, 0.9, 0.8, 1.05], vertical_alignment="center")
    head[0].markdown('<p class="pane-title">Portfolio</p>', unsafe_allow_html=True)
    live = head[1].toggle("Live", value=True, key="live",
                          help=f"Refresh prices from Yahoo Finance every {LIVE_EVERY_SEC} s. Yahoo sees your "
                               "stock symbols — never your quantities or prices paid.")
    with head[2].popover("＋ Add", width="stretch"):
        with st.form("add_stock", clear_on_submit=True, border=False):
            labels, by_label = stock_options()
            # Typing filters the list as you go (symbol or company name); something not in the
            # list can still be entered — NSE by default, or with a suffix like TCS.BO for BSE.
            choice = st.selectbox("Stock", labels, index=None, accept_new_options=True,
                                  placeholder="Type a symbol or company — e.g. RELIANCE, Infosys",
                                  help="NSE stocks and ETFs, and US companies. Not listed? Type the symbol "
                                       "(NSE by default; add .BO for BSE, e.g. TCS.BO).")
            c1, c2 = st.columns(2)
            qty = c1.number_input("Quantity", min_value=0.0, value=None, step=1.0, format="%g")
            cost = c2.number_input("Avg buy price", min_value=0.0, value=None, step=1.0, format="%.2f",
                                   help="Per share — ₹ for NSE/BSE, $ for US")
            bought = st.date_input("Bought on (optional)", value=None, format="DD/MM/YYYY")
            if st.form_submit_button("Add to portfolio", type="primary", width="stretch"):
                ticker = stock_list.ticker_for(choice, by_label)
                errors = add_holding({"ticker": ticker, "quantity": qty, "avg_cost": cost,
                                      "purchase_date": bought.isoformat() if bought else ""})
                for e in errors:
                    st.error(e)
                if not errors:
                    st.session_state.live_quotes = None  # fetch the new stock's price now
                    symbol, exchange = prices.display_symbol(ticker)
                    st.toast(f"Added {symbol} ({exchange})")
                    st.rerun()
    if head[3].button("Edit", width="stretch", help="Change or delete holdings"):
        edit_holdings_dialog()
    if head[4].button("⇅ Import", width="stretch", help="Upload your CAS statement (every broker) or a broker CSV"
                       + ("" if HOSTED else ", or sync Upstox / Dhan / Groww")):
        import_sync_dialog()
    if st.session_state.get("price_note"):
        st.toast(st.session_state.pop("price_note"))

    holdings = load_holdings()
    if not holdings:
        st.info("No holdings yet — use **＋ Add** to add a stock, or **⇅ Import** to upload your monthly CAS "
                "statement (all your brokers in one PDF) or a broker CSV"
                + (". Only you can see your holdings." if HOSTED else
                   ", or sync Upstox / Dhan / Groww. Your holdings stay on this machine."))
        return

    # Pause live refresh while an answer is generating, so a refresh can't interrupt it.
    @st.fragment(run_every=LIVE_EVERY_SEC if live and pending is None else None)
    @as_session_user
    def dashboard() -> None:
        import time
        from datetime import datetime
        tickers = [h["ticker"] for h in holdings]
        cached = st.session_state.get("live_quotes")
        fresh = cached and cached["tickers"] == tickers and time.time() - cached["at"] < LIVE_EVERY_SEC - 5
        error = None
        if live and not fresh:
            try:
                quotes = prices.yahoo_quotes(prices.with_fx(tickers))  # funds (MF:…) are never sent to Yahoo
                prices.save_yahoo_prices(quotes)  # answers use the same prices
                fund_tickers = [t for t in tickers if mutual_funds.is_mf(t)]
                if fund_tickers:
                    nav_error = mutual_funds.refresh_if_stale()  # AMFI's whole NAV file, at most every 6 hours
                    quotes.update(mutual_funds.quotes(fund_tickers))
                    if nav_error:
                        st.session_state.price_note = nav_error
                st.session_state.live_quotes = {"tickers": tickers, "at": time.time(), "quotes": quotes}
            except Exception as e:
                error = e
        cached = st.session_state.get("live_quotes")
        if cached and cached["tickers"] == tickers:
            quotes, stamp = cached["quotes"], datetime.fromtimestamp(cached["at"]).strftime("%H:%M:%S")
        else:  # live off (or failed before any fetch): last saved prices, no day change
            quotes = {t: {**p, "prev_close": None} for t, p in prices.load_prices().items()}
            stamp = None
        df, tot = portfolio_frame(holdings, quotes)
        pnl = tot["value_costed"] - tot["cost"]  # only holdings whose average price is known
        day_pct = tot["day"] / tot["prev"] * 100 if tot["prev"] else None
        pnl_pct = pnl / tot["cost"] * 100 if tot["cost"] else None
        no_cost = f" · {len(tot['no_cost'])} without avg price" if tot["no_cost"] else ""
        st.markdown('<div class="kpis">'
                    # whole rupees on the cards (paise in the table), as Indian broker apps show it
                    + kpi("Invested", prices.inr(tot["cost"], 0) if tot["costed"] else "—",
                          f"{len(holdings)} holdings{no_cost}")
                    + kpi("Current value", prices.inr(tot["value"], 0), "")
                    + kpi("Day's P&L", prices.signed_money(tot["day"], decimals=0) if tot["prev"] else "—",
                          f"{day_pct:+.2f}%" if day_pct is not None else "", tone(day_pct))
                    + kpi("Total P&L", prices.signed_money(pnl, decimals=0) if tot["costed"] else "—",
                          f"{pnl_pct:+.2f}%" if pnl_pct is not None else "",
                          tone(pnl_pct))
                    + "</div>", unsafe_allow_html=True)
        insights_tab, holdings_tab, alloc_tab, stock_tab = st.tabs(["Insights", "Holdings", "Allocation", "Stock view"])
        with insights_tab:
            insights_view(holdings, df, tot)
        with holdings_tab:
            holdings_table(df)
        with alloc_tab:
            allocation_charts(df)
        with stock_tab:
            stock_view(holdings, pending)
        note = (f"Live · updated {stamp} · Yahoo Finance (NSE/BSE quotes may be delayed)" if stamp
                else "Last saved prices — turn on Live to refresh")
        if tot["fx"]:
            note += f" · US stocks at ₹{tot['fx']:.2f}/$"
        nav_dates = sorted({q["as_of"][:10] for t, q in quotes.items() if mutual_funds.is_mf(t)})
        if nav_dates:
            note += f" · mutual funds at AMFI NAV of {nav_dates[-1]}"
        if tot["missing"]:
            note += f" · no price for {', '.join(tot['missing'])} — check the symbol and exchange"
        if tot["no_fx"]:
            note += f" · no USD→INR rate yet for {', '.join(tot['no_fx'])}"
        if tot["no_cost"]:
            note += (f" · average price unknown for {', '.join(tot['no_cost'])} — Invested and P&L leave "
                     f"{'it' if len(tot['no_cost']) == 1 else 'them'} out "
                     "(enter it by importing the statement again: ⇅ Import → CAS statement)")
        st.markdown(f'<p class="footnote">{html.escape(note)}</p>', unsafe_allow_html=True)
        if error:
            st.warning(f"Price refresh failed: {error}")

    dashboard()


@st.cache_resource(show_spinner=False)
def answer_cache() -> dict:
    """Finished answers by question, shared by every session while the app runs (not portfolio ones:
    those depend on your holdings and today's prices)."""
    return {}


@st.cache_resource(show_spinner=False)
def warm_up(model_name: str) -> bool:
    """Load the model in the background when the app opens, so the first question doesn't wait for it
    (an idle model on this Mac took ~20 s extra to start). The prompt is a fixed word — nothing personal."""
    import threading

    def run():
        try:
            r.generate("Reply with OK.", model=model_name, options={"num_predict": 1})
        except Exception:
            pass  # the real question will report any problem
    threading.Thread(target=run, daemon=True).start()
    return True


def short_doc(c: Citation) -> str:
    """'TCS Annual Report FY2025-26 (p. 88)' style label for the progress line."""
    _, doc, where = source_info(c)
    page = re.search(r"\(?(p\. [\d–]+)\)?", where)
    return compact_title(doc) + (f" ({page.group(1)})" if page else "")


STILL_WORKS = "Your portfolio, prices and insights keep working."
LIMIT_NOTICES = {
    "user_limit": f"You've used today's {usage.user_limit()} free question{'s' if usage.user_limit() != 1 else ''} — "
                  "they reset at midnight (India time). "
                  + STILL_WORKS,
    "site_limit": "PortIQ's free daily question limit is used up for everyone today — questions reopen at midnight "
                  "(India time). " + STILL_WORKS,
}


def answer_pending(query: str) -> None:
    """Route, retrieve, stream the answer and check citations for the queued question."""
    import time
    started = time.time()
    with st.chat_message("assistant"):
        decision, plan, positions = r.plan_query(query, use_portfolio=use_portfolio, route=route)
        key = (workspace.cache_scope(), " ".join(query.lower().split()), decision.route, model, check,
               collection.count())  # per user: an answer can carry the asker's positions
        cached = answer_cache().get(key) if decision.route != "portfolio" else None
        if cached:
            res = {**cached, "cached": True}
            render_assistant(res, show_details)
            st.session_state.messages.append({"role": "assistant", "result": res})
            return
        res = {
            "route": decision.route, "confidence": decision.confidence, "secondary": decision.secondary_routes,
            "describe": decision.describe(), "positions": positions, "citations": [],
            "retrieval_notes": [], "answer": "", "disclaimer": "", "report": None,
            "valuation": r.valuation_for(decision, positions),
        }
        render_route(res, show_details)  # the route is known before searching — show it right away
        status = st.status(f"Searching {collection.count():,} passages…", expanded=False)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            citations = r.retrieve_for_plan(query, collection, plan)
        res["citations"] = citations
        res["retrieval_notes"] = [line.strip() for line in log.getvalue().splitlines() if line.startswith("↪")]
        docs = list(dict.fromkeys(short_doc(c) for c in citations))
        render_retrieval_notes(res, show_details)
        render_positions(positions)
        render_valuation(res["valuation"])

        if not citations:  # never ask the model to answer from nothing
            status.update(label="No matching passages in PortIQ's documents", state="complete")
            res["answer"] = (prices.summary_sentence(res["valuation"]) if res["valuation"]
                             else prompts.NO_CONTEXT_ANSWER)
            st.markdown(md(res["answer"]))
        else:
            where = docs[0] + (f" and {len(docs) - 1} more" if len(docs) > 1 else "")
            status.update(label=f"Reading {len(citations)} passages — {where}…")
            prompt = r.build_prompt(query, citations, positions=positions, route=decision.route)
            ws = workspace.current()
            allowance = usage.reserve(ws.user_key) if ws else usage.Allowance(True)  # only AI calls count
            if not allowance.ok:
                status.update(label="Today's question limit is reached", state="complete")
                res["notice"] = LIMIT_NOTICES[allowance.reason]
                st.info(res["notice"])
            else:
                try:
                    tokens = r.stream_tokens(prompt, model=model)
                    first = next(tokens, "")
                    status.update(label=f"Read {len(citations)} passages in {time.time() - started:.0f} s — {where}",
                                  state="complete")
                    res["answer"] = st.write_stream(md(tok) for tok in itertools.chain([first], tokens))
                    res["answer"] = res["answer"].replace("\\$", "$")  # store unescaped
                except backends.ProviderLimit as e:  # the free AI quota (daily) or rate (per minute)
                    if ws:
                        usage.refund(ws.user_key)
                    usage.note("provider_limit" if e.daily else "provider_busy")
                    status.update(label="The AI service is at its free limit", state="error")
                    res["notice"] = str(e) + " " + STILL_WORKS
                    st.info(res["notice"])
                except Exception as e:  # unreachable server, cloud-model refusal, ...
                    if ws:
                        usage.refund(ws.user_key)
                    status.update(label="The model couldn't answer", state="error")
                    st.error(f"Generation failed: {e}")

        res["refusal"] = bool(res["answer"]) and is_refusal(res["answer"])
        if decision.route == "portfolio" and res["answer"] and citations and not res["refusal"]:
            res["disclaimer"] = prompts.PORTFOLIO_DISCLAIMER
            st.caption(res["disclaimer"])
        # A refusal has nothing to check — the "no citations" warning would make a
        # correct "not covered" answer look like a failure.
        if check and res["answer"] and citations and not res["refusal"]:
            with st.spinner("Checking every statement against its source…"):
                res["report"] = check_citations(res["answer"], citations, get_encoder(),
                                                extra_sources=[query] + [format_position(h) for h in positions])
        res["seconds"] = time.time() - started
        trust_bar(res)
        render_sources(citations, refusal=res["refusal"], show_details=show_details)
    if res["answer"] and decision.route != "portfolio":
        answer_cache()[key] = res
    st.session_state.messages.append({"role": "assistant", "result": res})


def assistant_pane(pending) -> None:
    head = st.columns([3, 1], vertical_alignment="center")
    head[0].markdown('<p class="pane-title">Research assistant</p>', unsafe_allow_html=True)
    if head[1].button("Clear", width="stretch", disabled=pending is not None or not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()
    box = st.container(height=640, border=True)
    if assistant_problem:  # no model or no document store: the portfolio still works, questions don't
        with box:
            st.warning(assistant_problem)
        st.chat_input("The research assistant is off until the problem above is fixed", disabled=True)
        return
    with box:
        if not st.session_state.messages:
            st.markdown("**Ask about your holdings, a company's filings or annual report, or a finance concept.**")
            st.caption(f"Answers come from {collection.count():,} passages — finance explainers, US SEC filings, "
                       "Indian annual reports and market-microstructure papers — and every answer cites its sources.")
            for i, s in enumerate(SUGGESTIONS):
                if st.button(s, key=f"suggest_{i}", width="stretch"):
                    ask(s)
                    st.rerun()
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(md(msg["content"]))
                else:
                    render_assistant(msg["result"], show_details)
    # A new question while an answer is still generating would restart the script and
    # silently drop that answer, so the input is locked until the pending one is done.
    ws = workspace.current()
    left, site_out = usage.left_today(ws.user_key) if ws else (None, False)
    out_of_questions = ws is not None and (site_out or left == 0)
    query = st.chat_input("Questions reopen at midnight (India time)" if out_of_questions else
                          "Ask about a stock, a filing, a concept, or your holdings…",
                          disabled=pending is not None or out_of_questions)
    if ws:
        st.caption(LIMIT_NOTICES["site_limit"] if site_out else LIMIT_NOTICES["user_limit"] if left == 0 else
                   f"{left} of {usage.user_limit()} free questions left today · repeats and questions PortIQ "
                   "has no documents for don't count")
    st.markdown('<p class="disclaimer">Educational information from public company documents — not investment '
                "advice. Figures are checked against the cited sources; open them to verify.</p>",
                unsafe_allow_html=True)
    if query:
        ask(query)
        st.rerun()
    if pending:
        try:
            with box:
                answer_pending(pending)
        finally:
            st.session_state.pending = None  # a failing question must not lock the input forever
        st.rerun()  # re-enable the input


# ── Page ─────────────────────────────────────────────────────────────────────
# Who is this? Hosted: nothing personal is read until a signed-in user has accepted the privacy notice.
ws, signed_in_as, dev_login = None, "", False
st.session_state["_portiq_workspace"] = None  # set again below once this run's user is known
if HOSTED:
    dev = os.environ.get("PORTIQ_DEV_LOGIN", "").strip()
    if dev and st.get_option("server.address") in ("127.0.0.1", "localhost"):  # testing hosted mode locally only
        ws, signed_in_as, dev_login = workspace.for_user("dev", dev), f"{dev} (dev login)", True
    elif not auth_configured():
        st.error("Sign-in isn't set up: add an [auth] section (Google client ID and secret) to "
                 ".streamlit/secrets.toml — see the README's hosting section.")
        st.stop()
    elif not st.user.is_logged_in:
        sign_in_page()
        st.stop()
    else:
        ws = workspace.for_user(str(st.user.get("iss") or ""), str(st.user.get("sub") or ""))
        signed_in_as = str(st.user.get("email") or st.user.get("name") or "your Google account")
    if remote_store.enabled() and not ws.root.exists():  # free hosting wiped its disk: restore this user's files
        try:
            remote_store.pull(ws.user_key, ws.root)
        except Exception as e:  # never go on with an empty folder — a save would overwrite the real holdings
            st.error(f"Couldn't load your saved portfolio right now ({type(e).__name__}). Please reload in a minute.")
            st.stop()
    workspace.use(ws)
    st.session_state["_portiq_workspace"] = ws  # for dialogs and fragments, which rerun without this gate
    try:
        remote_store.push(ws.user_key, ws.root)  # save what the last run changed (every change is followed by a rerun)
    except Exception as e:
        st.warning(f"Your latest changes aren't backed up yet ({type(e).__name__}) — PortIQ will retry.")
    if not ws.has_consented():
        privacy_notice(ws)
        st.stop()

collection = get_collection()
if "messages" not in st.session_state:
    st.session_state.messages = []
pending = st.session_state.get("pending")

models = r.local_models()
model = None
api_cfg = backends.api_config()  # an OpenAI-compatible server from PORTIQ_LLM_URL, if configured
status, is_open = market_status()
top = st.columns([4.4, 1.25, 1.1], vertical_alignment="center")
with top[1].popover("ⓘ How it works", width="stretch"):
    how_it_works()
with top[2].popover("⚙ Settings", width="stretch"):
    if ws:
        account_controls(ws, signed_in_as, dev_login)
        st.divider()
    if models:
        default = models.index(r.DEFAULT_MODEL) if r.DEFAULT_MODEL in models else \
            next((i for i, m in enumerate(models) if m.startswith(r.DEFAULT_MODEL)), 0)
        model = st.selectbox("Model", models, index=default,
                             help="mlx:… runs on Apple's MLX server, api:… on the server in PORTIQ_LLM_URL, "
                                  "the others on Ollama.")
    remote_host = (api_cfg["host"] if models and api_cfg and model == api_cfg["name"] and not api_cfg["local"]
                   else None)
    route = st.radio("Route", ["auto", "concept", "technical", "portfolio"], horizontal=True,
                     help="auto = deterministic rules in router.py")
    use_portfolio = st.toggle("Use my holdings in answers", value=True)
    check = st.toggle("Check citations", value=True)
    show_details = st.toggle("Show routing details", value=False)
    if api_cfg and not api_cfg["allowed"]:
        st.warning(f"PORTIQ_LLM_URL points to {api_cfg['host']}, which isn't this machine, so it isn't offered. "
                   "Set PORTIQ_ALLOW_REMOTE_LLM=1 to allow sending prompts there.")
    if remote_host:
        st.caption(f"☁ Answers are generated by the model server at {remote_host} (allowed with "
                   "PORTIQ_ALLOW_REMOTE_LLM): your questions, the retrieved passages and the details of holdings "
                   "you ask about are sent there. Live prices send only stock symbols to Yahoo Finance.")
    else:
        st.caption("🔒 " + privacy_summary())
top[0].markdown(
    f'<div class="topbar"><div class="brand"><img src="{logo_src()}" alt="">'
    '<div><div class="word">Port<span class="iq">IQ</span></div><small>Upload. Ask. Get Insights.</small></div></div>'
    f'<span class="pill {"open" if is_open else "closed"}">● {html.escape(status)}</span>'
    + (f'<span class="pill">☁ {html.escape(remote_host)} · {html.escape(model)}</span>' if remote_host else
       f'<span class="pill">🔒 Local · {html.escape(model if models else "no model")}</span>')
    + '</div>', unsafe_allow_html=True)
# Without a model or the document store only the research assistant is off — the portfolio
# (holdings, prices, insights, import) needs neither.
assistant_problem = (
    "The document store wasn't found, so questions can't be answered yet. Build it once with "
    "`python scripts/build_vectorstore.py`." if collection is None else
    "No language model is running, so questions can't be answered right now — your portfolio still works. "
    "Start the MLX server (`scripts/start_mlx_server.sh`, 127.0.0.1:8080) or Ollama (127.0.0.1:11434), or point "
    "PORTIQ_LLM_URL / PORTIQ_LLM_MODEL at an OpenAI-compatible server (vLLM, a Hugging Face endpoint), then "
    "reload this page. If Ollama's models live on the external drive and the Mac just restarted, Ollama may have "
    "started before the drive mounted — quit and reopen it." if not models else None)
if models and collection is not None:
    warm_up(model)

left, right = st.columns([1.35, 1], gap="large")
with left:
    portfolio_pane(pending)
with right:
    assistant_pane(pending)
