"""
Router accuracy test — labelled queries across all three routes, including
ambiguous "trap" words that appear in both concept and technical contexts.

    python tests/test_router.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from router import forced_route, route_query  # noqa: E402

# Fixed holdings so the test doesn't depend on data/portfolio/holdings.csv
HOLDINGS = [
    {"ticker": "AAPL", "quantity": 25, "avg_cost": 182.5, "purchase_date": "2024-03-15"},
    {"ticker": "MSFT", "quantity": 10, "avg_cost": 410.0, "purchase_date": "2024-06-03"},
    {"ticker": "JPM", "quantity": 15, "avg_cost": 195.2, "purchase_date": "2025-01-10"},
]

# (query, expected route, note)
LABELLED = [
    # --- concept ---
    ("What is compound interest?", "concept", ""),
    ("What does bond duration mean?", "concept", "trap: duration"),
    ("What is a credit spread?", "concept", "trap: spread"),
    ("How do call and put options work?", "concept", "trap: options"),
    ("What is the formula for compound interest?", "concept", "trap: formula"),
    ("Explain volatility in simple terms", "concept", "trap: volatility"),
    ("What is an EMI and how is it calculated?", "concept", "trap: calculated, no numbers"),
    ("What is a systematic investment plan?", "concept", ""),
    ("What is diversification?", "concept", ""),
    ("How does an index fund work?", "concept", ""),
    ("What is liquidity?", "concept", "trap: liquidity"),
    ("What is arbitrage?", "concept", "trap: arbitrage"),
    # --- technical ---
    ("How is bid-ask spread modeled in market microstructure research?", "technical", "trap: spread"),
    ("What is latency arbitrage in high-frequency trading?", "technical", "trap: arbitrage"),
    ("How does a limit order book work?", "technical", ""),
    ("What is optimal execution and price impact?", "technical", ""),
    ("Derive the duration of a zero-coupon bond", "technical", "trap: duration"),
    ("What option pricing models are used for volatility estimation?", "technical", "trap: options"),
    ("What risk factors does Tesla disclose in its 10-K?", "technical", "non-held company"),
    ("What was Exxon's revenue last quarter?", "technical", "non-held company"),
    ("Calculate the EMI for a $500,000 loan at 3.5% for 10 years", "technical", "calculate-with-numbers"),
    ("How do market makers manage order flow toxicity?", "technical", ""),
    ("How is volatility estimated empirically from high-frequency data?", "technical", "trap: volatility"),
    # --- portfolio ---
    ("What risks does my AAPL position face?", "portfolio", ""),
    ("How is my portfolio doing?", "portfolio", ""),
    ("Explain duration for my bond fund", "portfolio", "mixed: portfolio + concept"),
    ("What litigation risks does JPM face?", "portfolio", "mixed: held ticker + filing term"),
    ("Should I sell my Microsoft shares?", "portfolio", "held company by name"),
    ("What is MSFT's revenue by segment?", "portfolio", "mixed: held ticker + filing terms"),
    ("Compare my holdings with Tesla's risk factors", "portfolio", "mixed: portfolio + technical"),
    ("What is an ETF and do I own any?", "portfolio", ""),
    # Held company by *name* routes like its ticker (JPM -> portfolio, so JPMorgan must too).
    # Tier 1 has nothing on Apple, so "concept" would retrieve unrelated explainers.
    ("What is the price-to-earnings ratio?", "concept", "trap: earnings (filing word inside a concept name)"),
    ("What were Exxon's earnings last year?", "technical", "trap: earnings + company"),
    ("What is the microstructural origin of rough volatility?", "technical", "trap: volatility + microstructural"),
    ("Is Apple a good company?", "portfolio", "trap: held company by name, no 'my'"),
    ("What litigation risks does JPMorgan face?", "portfolio", "trap: held company by name, no 'my'"),
]


def test_labelled_accuracy() -> bool:
    per_route = defaultdict(lambda: [0, 0])
    failures = []
    for query, expected, note in LABELLED:
        d = route_query(query, HOLDINGS)
        per_route[expected][1] += 1
        if d.route == expected:
            per_route[expected][0] += 1
        else:
            failures.append((query, expected, d.route, note, d.describe()))

    print(f"{'route':<10} {'correct':>8} {'total':>6} {'accuracy':>9}")
    for route in ("concept", "technical", "portfolio"):
        ok, n = per_route[route]
        print(f"{route:<10} {ok:>8} {n:>6} {ok / n:>9.0%}")
    ok_all = sum(v[0] for v in per_route.values())
    print(f"{'ALL':<10} {ok_all:>8} {len(LABELLED):>6} {ok_all / len(LABELLED):>9.0%}")
    traps = [q for q in LABELLED if q[2].startswith("trap")]
    trap_ok = sum(route_query(q, HOLDINGS).route == e for q, e, _ in traps)
    print(f"traps: {trap_ok}/{len(traps)}")
    for q, exp, got, note, desc in failures:
        print(f"  FAIL [{note}] {q!r}: expected {exp}, got {got}\n       {desc}")
    return not failures


def test_mixed_queries_report_secondary_signals() -> None:
    d = route_query("Explain duration for my bond fund", HOLDINGS)
    assert d.route == "portfolio" and "concept" in d.secondary_routes, d.describe()
    d = route_query("What litigation risks does JPM face?", HOLDINGS)
    assert d.route == "portfolio" and "technical" in d.secondary_routes, d.describe()
    d = route_query("Compare my holdings with Tesla's risk factors", HOLDINGS)
    assert d.route == "portfolio" and "technical" in d.secondary_routes, d.describe()
    print("PASS: mixed queries pick the highest-priority route and list secondary routes")


def test_retrieval_plans() -> None:
    concept = route_query("What is compound interest?", HOLDINGS).plan
    assert concept.tiers == [1] and concept.k == 4 and concept.distance_threshold is not None
    tech = route_query("How does a limit order book work?", HOLDINGS).plan
    assert tech.tiers == [2, 3] and tech.k == 6
    tsla = route_query("What risk factors does Tesla disclose in its 10-K?", HOLDINGS).plan
    assert tsla.tiers == [2] and tsla.tickers == ["TSLA"]
    port = route_query("What risks does my AAPL position face?", HOLDINGS).plan
    assert port.tiers == [2] and port.tickers == ["AAPL"] and port.concept_k == 0
    port_def = route_query("What is an ETF and do I own any?", HOLDINGS).plan
    assert port_def.concept_k == 0 or port_def.tiers == [1]
    msft_def = route_query("What is MSFT's revenue by segment?", HOLDINGS).plan
    assert msft_def.tickers == ["MSFT"] and msft_def.concept_k == 2  # "what is" -> + tier-1 concept chunks
    print("PASS: retrieval plans (tiers / k / tickers / concept supplement) per route")


def test_no_portfolio_and_forced_route() -> None:
    d = route_query("What risks does my AAPL position face?", HOLDINGS, allow_portfolio=False)
    assert d.route != "portfolio", d.describe()
    f = forced_route("technical", "What is compound interest?", HOLDINGS)
    assert f.route == "technical" and f.plan.tiers == [2, 3]
    print("PASS: --no-portfolio never routes to portfolio; --route forces a route")


def test_valuation_flag_only_for_own_performance() -> None:
    yes = ["How is my portfolio doing?", "How has my AAPL position performed?", "What is my portfolio worth?",
           "What's my P&L on MSFT?"]
    no = ["What risks does my AAPL position face?", "What are capital gains?", "How do index funds perform?",
          "Is my portfolio diversified enough?"]
    assert all(route_query(q, HOLDINGS).wants_valuation for q in yes), [q for q in yes if not route_query(q, HOLDINGS).wants_valuation]
    assert not any(route_query(q, HOLDINGS).wants_valuation for q in no), [q for q in no if route_query(q, HOLDINGS).wants_valuation]
    assert not route_query("How is my portfolio doing?", HOLDINGS, allow_portfolio=False).wants_valuation
    print("PASS: valuation only for performance questions about your own holdings")


if __name__ == "__main__":
    ok = test_labelled_accuracy()
    test_mixed_queries_report_secondary_signals()
    test_retrieval_plans()
    test_no_portfolio_and_forced_route()
    test_valuation_flag_only_for_own_performance()
    if not ok:
        sys.exit("\nRouter accuracy test FAILED")
    print("\nAll router tests passed.")
