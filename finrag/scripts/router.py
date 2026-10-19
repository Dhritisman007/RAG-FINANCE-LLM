"""
Day 10: Deterministic query router.

Classifies a query into one of three routes with plain rules — no LLM call
(a 3B model is unreliable at this) — and returns the retrieval plan for it:

    portfolio  query names a held ticker, or asks about "my portfolio/position/..."
    technical  market microstructure, quantitative modelling, or SEC-filing questions
    concept    everything else: "what is / how does X work" fundamentals

Priority when several routes match: portfolio > technical > concept. The
lower-priority routes that also matched are reported as `secondary_routes`.

Ambiguous words ("duration", "spread", "options", "formula", ...) are not
signals by themselves; they only count as technical next to a modelling /
research word ("bid-ask spread modeling", "option pricing model").

Usage:
    python scripts/router.py "explain duration for my bond fund"
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from portfolio_store import find_held_tickers, load_holdings  # noqa: E402

ROUTES = ("portfolio", "technical", "concept")

# Companies with SEC filings in the corpus (Tier 2). Names map to tickers so
# "Tesla's revenue" routes like "TSLA revenue".
CORPUS_COMPANIES = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase"],
    "TSLA": ["tesla"],
    "XOM": ["exxon", "exxonmobil", "exxon mobil"],
}

PORTFOLIO_PHRASES = [
    r"\bmy (?:portfolio|holdings?|positions?|stocks?|shares|funds?|investments?|bonds?|bond fund|etfs?)\b",
    r"\bi (?:own|hold|bought|invested in)\b",
    r"\b(?:should i|do i) (?:sell|buy|hold)\b",
]

# Unambiguous technical terms — each one routes technical on its own.
MICROSTRUCTURE_TERMS = [
    r"limit order books?", r"\border books?\b", r"\blatency\b", r"bid[- ]ask spread", r"price impact",
    r"optimal execution", r"market[- ]mak(?:ing|ers?)", r"order flow", r"toxicity", r"high[- ]frequency",
    r"\bhft\b", r"adverse selection", r"\bkyle\b", r"hawkes", r"queue position", r"tick size",
    r"market microstructure", r"rough volatility", r"\bvwap\b", r"\btwap\b", r"implementation shortfall", r"dark pools?",
]
FILING_TERMS = [
    r"\b10-?k\b", r"\b10-?q\b", r"\bannual report\b", r"risk factors?", r"\brevenues?\b", r"\bnet income\b",
    r"gross (?:margin|profit)", r"operating (?:income|margin)", r"\bmd&a\b",
    r"management.s discussion", r"\blitigation\b", r"legal proceedings", r"\bsegments?\b", r"\bfiling\b",
    r"\bsec\b", r"\bguidance\b", r"\bbalance sheet\b", r"cash flow statement", r"share repurchases?",
]
# Words that turn an ambiguous term technical when they appear with it.
MODELLING_WORDS = [
    r"\bmodel(?:s|led|ed|ling|ing)?\b", r"\bstochastic\b", r"\bempirical(?:ly)?\b", r"\bestimat(?:e|ed|ion|ing)\b",
    r"\bcalibrat(?:e|ed|ion|ing)\b", r"\bresearch\b", r"\bpapers?\b", r"\bderiv(?:e|ed|ation|ing)\b",
    r"\bprove\b", r"\bproof\b", r"\bdynamics\b", r"\bsimulat(?:e|ed|ion)\b", r"\bregression\b", r"\bpdes?\b",
    r"\bclosed[- ]form\b", r"\bequilibrium\b", r"\bmicrostructur(?:e|al)\b",
]
AMBIGUOUS_TERMS = [
    r"\bduration\b", r"\bspreads?\b", r"\boptions?\b", r"\bformulas?\b", r"\bvolatility\b", r"\bliquidity\b",
    r"\bpricing\b", r"\barbitrage\b", r"\byield curve\b", r"\bconvexity\b",
]
# Filing words that are also part of basic concept names ("price-to-earnings
# ratio"): technical only when a corpus company is named or another filing
# term is present ("Apple's earnings", "earnings in the 10-K").
FILING_AMBIGUOUS = [r"\bearnings\b"]
# "calculate/compute ... <a number>" — a worked numeric problem.
NUMERIC_CALC_RE = re.compile(
    r"\b(?:calculate|compute|work out|find)\b.*?(?:\$\s?\d|\d+(?:\.\d+)?\s?%|\b\d{3,}\b)"
    r"|(?:\$\s?\d|\d+(?:\.\d+)?\s?%).*?\b(?:calculate|compute|work out)\b",
    re.I,
)
# "what is X" about an instrument — portfolio queries also get Tier 1 concept chunks.
# Performance / valuation questions about your own holdings — answered with
# prices.py numbers computed by code, never by the model.
PERFORMANCE_RE = re.compile(
    r"\bhow (?:is|are|has|have) my\b.*\b(?:doing|performing|performed|done)\b"
    r"|\b(?:performance|p&l|p/l|profit|loss(?:es)?|gains?|returns?|worth|market value|valuation)\b"
    r"|\bup or down\b|\bhow much (?:is|are|have)\b",
    re.I,
)
DEFINITION_RE = re.compile(r"\b(?:what (?:is|are|does)|explain|define|meaning of|how does .* work)\b", re.I)


@dataclass
class RetrievalPlan:
    tiers: list[int] | None          # None = all tiers
    k: int                           # distinct passages to keep
    overfetch: int                   # candidates fetched per kept passage (near-dup suppression)
    distance_threshold: float | None = None   # concept: fall back if the best hit is worse
    fallback_tiers: list[int] | None = None   # tiers to search on fallback (None = all)
    fallback: bool = True            # False: drop weak matches instead of searching wider
    tickers: list[str] = field(default_factory=list)  # restrict Tier 2 to these companies
    concept_k: int = 0               # extra Tier 1 passages (portfolio "what is X" questions),
                                     # kept only if within distance_threshold
    max_per_doc: int | None = None   # cap passages per document (source diversity)


@dataclass
class RouteDecision:
    route: str
    confidence: float
    matched_signals: dict[str, list[str]]
    secondary_routes: list[str]
    plan: RetrievalPlan
    held_tickers: list[str] = field(default_factory=list)
    company_tickers: list[str] = field(default_factory=list)  # corpus companies mentioned (held or not)
    wants_valuation: bool = False  # portfolio performance question -> show prices.py valuation

    def describe(self) -> str:
        signals = "; ".join(f"{r}: {', '.join(s)}" for r, s in self.matched_signals.items() if s) or "none"
        extra = f" (also: {', '.join(self.secondary_routes)})" if self.secondary_routes else ""
        if self.wants_valuation:
            extra += " [valuation]"
        p = self.plan
        tiers = "all" if p.tiers is None else "+".join(map(str, p.tiers))
        where = f", tickers={'+'.join(p.tickers)}" if p.tickers else ""
        concept = f", +{p.concept_k} tier-1 concept" if p.concept_k else ""
        return (f"route={self.route}{extra} confidence={self.confidence:.2f} | "
                f"tiers={tiers} k={p.k}{where}{concept} | signals: {signals}")


# Retrieval defaults per route. CONCEPT_DISTANCE_THRESHOLD, measured on the live
# store: questions Tier 1 covers have a best Tier 1 hit <= 0.206 (EMI); questions
# it doesn't cover are >= 0.257 (credit spread -> "credit score" article 0.297,
# liquidity 0.310, share buyback 0.257, Hawkes process 0.470). 0.23 splits the
# gap; above it the concept route falls back to searching all tiers.
CONCEPT_DISTANCE_THRESHOLD = 0.23
# Portfolio questions carry a personal clause ("...and do I own any?") that pulls
# the embedding away from the concept, so Tier 1 material gets more slack:
# "What is an ETF and do I own any?" -> ETF article 0.232 (keep), "Is my
# portfolio diversified enough?" 0.196 (keep), "How is my portfolio doing?"
# -> asset-allocation article 0.324 (drop: it doesn't answer that).
PORTFOLIO_CONCEPT_THRESHOLD = 0.28
# Broad technical questions (no company named): at most this many passages per
# paper, so answers draw on several sources. Over-fetch more to backfill.
TECHNICAL_MAX_PER_DOC = 2
TECHNICAL_OVERFETCH = 5


def _plan(route: str, held: list[str], companies: list[str], query: str) -> RetrievalPlan:
    if route == "concept":
        return RetrievalPlan(tiers=[1], k=4, overfetch=3,
                             distance_threshold=CONCEPT_DISTANCE_THRESHOLD, fallback_tiers=None)
    if route == "technical":
        if companies:  # a named company -> its filings
            return RetrievalPlan(tiers=[2], k=6, overfetch=3, tickers=companies)
        return RetrievalPlan(tiers=[2, 3], k=6, overfetch=TECHNICAL_OVERFETCH, max_per_doc=TECHNICAL_MAX_PER_DOC)
    # portfolio
    concept_k = 2 if DEFINITION_RE.search(query) else 0
    if held:
        return RetrievalPlan(tiers=[2], k=5, overfetch=3, tickers=held, concept_k=concept_k,
                             distance_threshold=PORTFOLIO_CONCEPT_THRESHOLD)
    # "my portfolio" with no specific ticker: every position + Tier 1 material only
    # if it genuinely covers the question. No fallback — for "how is my portfolio
    # doing?" a wider search only returns unrelated papers (best hit 0.31, HFT).
    return RetrievalPlan(tiers=[1], k=4, overfetch=3, distance_threshold=PORTFOLIO_CONCEPT_THRESHOLD,
                         fallback=False)


def _find(patterns: list[str], text: str) -> list[str]:
    hits = []
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            hits.append(m.group(0).lower())
    return hits


def route_query(query: str, holdings: list[dict] | None = None, allow_portfolio: bool = True) -> RouteDecision:
    """
    Deterministically route `query`. `holdings` defaults to holdings.csv;
    pass allow_portfolio=False (--no-portfolio) to never pick the portfolio route.
    """
    if holdings is None:
        holdings = load_holdings() if allow_portfolio else []
    held_set = {h["ticker"] for h in holdings} if allow_portfolio else set()

    # Corpus companies mentioned by ticker (uppercase) or by name
    companies = []
    for ticker, names in CORPUS_COMPANIES.items():
        if re.search(rf"(?<![A-Za-z]){ticker}(?![A-Za-z])", query) or \
           any(re.search(rf"\b{re.escape(n)}\b", query, re.I) for n in names):
            companies.append(ticker)

    held = find_held_tickers(query, holdings) if allow_portfolio else []
    portfolio_words = _find(PORTFOLIO_PHRASES, query) if allow_portfolio else []
    # A held company counts whether named by ticker ("JPM") or by name
    # ("JPMorgan") — otherwise the two spellings would route differently.
    held_by_name = [t for t in companies if t in held_set and t not in held]
    held = held + held_by_name

    signals: dict[str, list[str]] = {r: [] for r in ROUTES}
    signals["portfolio"] = [f"held:{t}" for t in held] + portfolio_words

    tech = _find(MICROSTRUCTURE_TERMS, query) + _find(FILING_TERMS, query)
    non_held = [t for t in companies if t not in held]
    tech += [f"company:{t}" for t in non_held]
    if tech or companies:
        tech += _find(FILING_AMBIGUOUS, query)
    ambiguous = _find(AMBIGUOUS_TERMS, query)
    modelling = _find(MODELLING_WORDS, query)
    if ambiguous and modelling:
        tech += [f"{a}+{modelling[0]}" for a in ambiguous]
    elif modelling and not ambiguous and tech:
        tech += modelling
    elif modelling and not ambiguous and re.search(r"\b(?:derive|derivation|prove|proof|stochastic|closed[- ]form)\b", query, re.I):
        tech += modelling  # pure maths requests are technical on their own
    if NUMERIC_CALC_RE.search(query):
        tech.append("calculate-with-numbers")
    signals["technical"] = tech
    signals["concept"] = [f"ambiguous:{a}" for a in ambiguous if not modelling] + \
                         (["definition-question"] if DEFINITION_RE.search(query) else [])

    matched = [r for r in ROUTES if signals[r] and r != "concept"]
    route = matched[0] if matched else "concept"
    secondary = [r for r in ROUTES if r != route and signals[r]]

    if route == "portfolio":
        confidence = 0.95 if held else 0.8
    elif route == "technical":
        confidence = min(0.95, 0.6 + 0.1 * len(tech))
    else:
        confidence = 0.8 if DEFINITION_RE.search(query) else 0.6

    return RouteDecision(
        route=route,
        confidence=round(confidence, 2),
        matched_signals=signals,
        secondary_routes=secondary,
        plan=_plan(route, held, non_held if route == "technical" else companies, query),
        held_tickers=held,
        company_tickers=companies,
        wants_valuation=route == "portfolio" and bool(PERFORMANCE_RE.search(query)),
    )


def forced_route(route: str, query: str, holdings: list[dict] | None = None,
                 allow_portfolio: bool = True) -> RouteDecision:
    """--route concept|technical|portfolio: keep the detected signals, force the route."""
    auto = route_query(query, holdings, allow_portfolio)
    if route == auto.route:
        return auto
    held = auto.held_tickers if route == "portfolio" else []
    companies = [t for t in auto.company_tickers if t not in auto.held_tickers]
    return RouteDecision(
        route=route, confidence=1.0, matched_signals=auto.matched_signals,
        secondary_routes=[r for r in ROUTES if r != route and auto.matched_signals[r]],
        plan=_plan(route, held, companies if route == "technical" else auto.company_tickers, query),
        held_tickers=held, company_tickers=auto.company_tickers,
        wants_valuation=route == "portfolio" and bool(PERFORMANCE_RE.search(query)),
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "explain duration for my bond fund"
    print(route_query(q).describe())
