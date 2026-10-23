"""
PortIQ — the FinRAG chat UI (Streamlit), the same pipeline as scripts/rag_loop.py:
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
import html
import io
import itertools
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import llm_backends as backends  # noqa: E402
import prices  # noqa: E402
import prompts  # noqa: E402
import stock_list  # noqa: E402
import rag_loop as r  # noqa: E402
from citations import Citation, CitationReport, check_citations, is_refusal  # noqa: E402
from portfolio_store import HOLDINGS_PATH, format_position, load_holdings, add_holding, save_holdings  # noqa: E402

# Brand: PortIQ — "Upload. Ask. Get Insights." The mark in assets/ is an SVG recreation;
# a saved original (assets/portiq_logo.png or .svg) is used instead when present.
ASSETS = Path(__file__).parent / "assets"
LOGO = next((p for p in (ASSETS / "portiq_logo.svg", ASSETS / "portiq_logo.png") if p.exists()),
            ASSETS / "portiq_mark.svg")
st.set_page_config(page_title="PortIQ", page_icon=str(LOGO), layout="wide", initial_sidebar_state="collapsed")

# ── Privacy guard ────────────────────────────────────────────────────────────
if st.get_option("server.address") not in ("127.0.0.1", "localhost") or st.get_option("browser.gatherUsageStats"):
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
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 4px 0 10px; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; min-width: 0; }
.kpi .label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: .06em; }
.kpi .value { font-size: clamp(1rem, 1.5vw, 1.3rem); font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi .sub { font-size: 0.8rem; margin-top: 2px; color: var(--muted); font-variant-numeric: tabular-nums; }
.up { color: var(--up) !important; } .down { color: var(--down) !important; }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 6px; }
@media (max-width: 900px) { .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.stat { border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; }
.stat .label { color: var(--muted); font-size: 0.72rem; } .stat .value { font-weight: 600; font-variant-numeric: tabular-nums; }
.footnote { color: var(--muted); font-size: 0.75rem; }
div[data-testid="stTabs"] button p { font-size: 0.9rem; }
div[data-testid="stCheckbox"] p, div[data-testid="stPopover"] button p,
div[data-testid="stButton"] button p { white-space: nowrap; }  /* "Live", "＋ Add", "Edit" stay on one line */
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


def kpi(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
            f'<div class="value {cls}">{html.escape(value)}</div><div class="sub {cls}">{html.escape(sub)}</div></div>')


def portfolio_frame(holdings: list[dict], quotes: dict[str, dict]):
    """Per-holding table (numbers in ₹ for totals) plus ₹ totals — all arithmetic here, never by the model."""
    import pandas as pd
    fx = (quotes.get(prices.FX_TICKER) or {}).get("price")
    rows, tot = [], {"value": 0.0, "cost": 0.0, "day": 0.0, "prev": 0.0, "missing": [], "no_fx": []}
    for h in holdings:
        symbol, exchange = prices.display_symbol(h["ticker"])
        cur = prices.currency(h["ticker"])
        rate = 1.0 if cur == "INR" else fx
        q = quotes.get(prices.normalize_ticker(h["ticker"]))
        row = {"Stock": symbol, "Exch": exchange, "Qty": h["quantity"], "Avg": h["avg_cost"], "LTP": None,
               "Day %": None, "Invested ₹": None, "Current ₹": None, "P&L ₹": None, "P&L %": None, "cur": cur,
               "ticker": h["ticker"]}
        if q is None or rate is None:
            (tot["missing"] if q is None else tot["no_fx"]).append(symbol)
        else:
            cost, value = h["quantity"] * h["avg_cost"] * rate, h["quantity"] * q["price"] * rate
            prev = q.get("prev_close")
            tot["value"] += value
            tot["cost"] += cost
            if prev:
                tot["day"] += h["quantity"] * (q["price"] - prev) * rate
                tot["prev"] += h["quantity"] * prev * rate
            row.update({"LTP": q["price"], "Day %": (q["price"] / prev - 1) * 100 if prev else None,
                        "Invested ₹": cost, "Current ₹": value, "P&L ₹": value - cost,
                        "P&L %": (value - cost) / cost * 100 if cost else None})
        rows.append(row)
    tot["fx"] = fx if any(prices.currency(h["ticker"]) == "USD" for h in holdings) else None
    return pd.DataFrame(rows), tot


def holdings_table(df) -> None:
    def money_native(row, col):
        v = row[col]
        return "—" if v is None or v != v else prices.money(v, row["cur"])

    view = df.copy()
    view["Avg"] = [money_native(r, "Avg") for _, r in df.iterrows()]
    view["LTP"] = [money_native(r, "LTP") for _, r in df.iterrows()]
    view = view.drop(columns=["cur", "ticker", "Exch"])  # US stocks are the ones priced in $

    def colour(v):
        return "" if v is None or v != v else f"color: {UP if v >= 0 else DOWN}"

    def inr_or_dash(v):
        return "—" if v is None or v != v else prices.inr(v)

    styled = (view.style
              .format({"Qty": "{:g}", "Day %": lambda v: "—" if v != v or v is None else f"{v:+.2f}%",
                       "P&L %": lambda v: "—" if v != v or v is None else f"{v:+.2f}%",
                       "Invested ₹": inr_or_dash, "Current ₹": inr_or_dash,
                       "P&L ₹": lambda v: "—" if v != v or v is None else prices.signed_money(v)})
              .map(colour, subset=["Day %", "P&L ₹", "P&L %"]))
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
def cached_history(ticker: str) -> list[tuple[str, float]]:
    return prices.yahoo_history(ticker, "1y")


def stock_view(holdings: list[dict], pending) -> None:
    import plotly.graph_objects as go
    names = {prices.display_symbol(h["ticker"])[0]: h for h in holdings}
    pick = st.selectbox("Stock", list(names), label_visibility="collapsed", key="stock_pick")
    h = names[pick]
    cur = prices.currency(h["ticker"])
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
    fig.add_hline(y=h["avg_cost"], line_dash="dot", line_color="#8b95a7",
                  annotation_text=f"your avg {prices.money(h['avg_cost'], cur)}", annotation_position="top left",
                  annotation_font_color="#8b95a7")
    lo, hi = min(closes + [h["avg_cost"]]), max(closes + [h["avg_cost"]])
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
             ("Your return", pct((s["last"] / h["avg_cost"] - 1) * 100 if h["avg_cost"] else None),
              tone(s["last"] - h["avg_cost"]))]
    st.markdown('<div class="stats">' + "".join(
        f'<div class="stat"><div class="label">{label}</div><div class="value {cls}">{html.escape(v)}</div></div>'
        for label, v, cls in cells) + "</div>", unsafe_allow_html=True)
    st.markdown('<p class="footnote">Past prices, not a forecast · Yahoo Finance daily closes</p>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, label, q in [(a, "Key risks", f"What are the key risks for {pick}?"),
                          (b, "Revenue & margins", f"What were {pick}'s revenue and profit in the latest year?"),
                          (c, "Business outlook", f"What does {pick}'s management say about the outlook?")]:
        if col.button(label, key=f"q_{label}_{pick}", width="stretch", disabled=pending is not None):
            ask(q)
            st.rerun(scope="app")


def portfolio_pane(pending) -> None:
    head = st.columns([2.2, 1, 1, 1], vertical_alignment="center")
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
    if st.session_state.get("price_note"):
        st.toast(st.session_state.pop("price_note"))

    holdings = load_holdings()
    if not holdings:
        st.info("No holdings yet — use **＋ Add** to add your first stock (NSE symbols like RELIANCE, TCS, HDFCBANK). "
                "Your holdings stay on this machine.")
        return

    # Pause live refresh while an answer is generating, so a refresh can't interrupt it.
    @st.fragment(run_every=LIVE_EVERY_SEC if live and pending is None else None)
    def dashboard() -> None:
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
        df, tot = portfolio_frame(holdings, quotes)
        pnl = tot["value"] - tot["cost"]
        day_pct = tot["day"] / tot["prev"] * 100 if tot["prev"] else None
        pnl_pct = pnl / tot["cost"] * 100 if tot["cost"] else None
        st.markdown('<div class="kpis">'
                    # whole rupees on the cards (paise in the table), as Indian broker apps show it
                    + kpi("Invested", prices.inr(tot["cost"], 0), f"{len(holdings)} holdings")
                    + kpi("Current value", prices.inr(tot["value"], 0), "")
                    + kpi("Day's P&L", prices.signed_money(tot["day"], decimals=0) if tot["prev"] else "—",
                          f"{day_pct:+.2f}%" if day_pct is not None else "", tone(day_pct))
                    + kpi("Total P&L", prices.signed_money(pnl, decimals=0), f"{pnl_pct:+.2f}%" if pnl_pct is not None else "",
                          tone(pnl_pct))
                    + "</div>", unsafe_allow_html=True)
        holdings_tab, alloc_tab, stock_tab = st.tabs(["Holdings", "Allocation", "Stock view"])
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
        if tot["missing"]:
            note += f" · no price for {', '.join(tot['missing'])} — check the symbol and exchange"
        if tot["no_fx"]:
            note += f" · no USD→INR rate yet for {', '.join(tot['no_fx'])}"
        st.markdown(f'<p class="footnote">{html.escape(note)}</p>', unsafe_allow_html=True)
        if error:
            st.warning(f"Price refresh failed: {error}")

    dashboard()


def answer_pending(query: str) -> None:
    """Route, retrieve, stream the answer and check citations for the queued question."""
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


def assistant_pane(pending) -> None:
    head = st.columns([3, 1], vertical_alignment="center")
    head[0].markdown('<p class="pane-title">Research assistant</p>', unsafe_allow_html=True)
    if head[1].button("Clear", width="stretch", disabled=pending is not None or not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()
    box = st.container(height=640, border=True)
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
    query = st.chat_input("Ask about a stock, a filing, a concept, or your holdings…",
                          disabled=pending is not None)
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
collection = get_collection()
if collection is None:
    st.error("Vector store not found — run `python scripts/build_vectorstore.py` first.")
    st.stop()
if "messages" not in st.session_state:
    st.session_state.messages = []
pending = st.session_state.get("pending")

models = r.local_models()
api_cfg = backends.api_config()  # an OpenAI-compatible server from PORTIQ_LLM_URL, if configured
status, is_open = market_status()
top = st.columns([5, 1.1], vertical_alignment="center")
with top[1].popover("⚙ Settings", width="stretch"):
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
        st.caption("🔒 Answers are generated on this machine (cloud models refused, no telemetry). Your holdings "
                   "file stays here; live prices send only stock symbols to Yahoo Finance.")
top[0].markdown(
    f'<div class="topbar"><div class="brand"><img src="{logo_src()}" alt="">'
    '<div><div class="word">Port<span class="iq">IQ</span></div><small>Upload. Ask. Get Insights.</small></div></div>'
    f'<span class="pill {"open" if is_open else "closed"}">● {html.escape(status)}</span>'
    + (f'<span class="pill">☁ {html.escape(remote_host)} · {html.escape(model)}</span>' if remote_host else
       f'<span class="pill">🔒 Local · {html.escape(model if models else "no model")}</span>')
    + f'<span class="pill">{collection.count():,} passages</span></div>', unsafe_allow_html=True)
if not models:
    st.error("No models found. Start Ollama (127.0.0.1:11434) or the MLX server "
             "(`scripts/start_mlx_server.sh`, 127.0.0.1:8080), or point PORTIQ_LLM_URL / PORTIQ_LLM_MODEL at an "
             "OpenAI-compatible server (vLLM, a Hugging Face endpoint). If Ollama's models live on the external "
             "drive and the Mac just restarted, Ollama may have started before the drive mounted and "
             "fallen back to its default (empty) folder — quit and reopen Ollama.")
    st.stop()

left, right = st.columns([1.35, 1], gap="large")
with left:
    portfolio_pane(pending)
with right:
    assistant_pane(pending)
