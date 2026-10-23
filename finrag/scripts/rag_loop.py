"""
Day 8: RAG query loop with streaming LLM responses.

Retrieves the top-k chunks from the ChromaDB vector store (built in Day 6),
injects them into a prompt, and streams the answer from a local Ollama model.
Supports both single-query mode and an interactive REPL.

Portfolio-aware path: if the query mentions a ticker held in
data/portfolio/holdings.csv, retrieval is restricted to that company's Tier 2
SEC filing chunks (Chroma `where` on company_ticker) and the position details
are added to the prompt.

PRIVACY (deliberate design decision): everything runs locally. Generation goes
only to the Ollama server on localhost (pinned below, so an OLLAMA_HOST env var
can't redirect it), Chroma telemetry is disabled, and portfolio data is never
sent to any cloud API.

Usage:
    # Single query
    python scripts/rag_loop.py --query "What is an EMI and how is it calculated?"

    # Interactive REPL
    python scripts/rag_loop.py --interactive

    # Override model or k
    python scripts/rag_loop.py --query "How does SIP work?" --model llama3.2 --k 6

    # Tier-filtered query (only search within one tier)
    python scripts/rag_loop.py --query "What is latency arbitrage?" --tier 3

    # Portfolio-aware query (ticker you hold in data/portfolio/holdings.csv)
    python scripts/rag_loop.py --query "What risks does my AAPL position face?"

    # Disable streaming (print full answer at once)
    python scripts/rag_loop.py --query "Explain bond duration" --no-stream
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np

# Local helper — same scripts/ directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))  # ensure scripts/ is on path
from citations import (  # noqa: E402
    Citation,
    citations_from_chroma_results,
    format_citations_block,
    deduplicate_citations,
    suppress_near_duplicates,
    check_citations,
    format_citation_report,
    order_key,
)
from portfolio_store import format_position, get_holding, load_holdings  # noqa: E402
from router import RetrievalPlan, RouteDecision, forced_route, route_query  # noqa: E402
import llm_backends as backends  # noqa: E402
import prices  # noqa: E402
import prompts  # noqa: E402

# ── Paths & constants ────────────────────────────────────────────────────────
CHROMA_DIR = Path.home() / ".finrag_chroma_db"
COLLECTION_NAME = "finrag_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Chosen by the Day 12 evaluation (43 questions, identical retrieval for both):
# qwen2.5:7b cites correctly 92% vs 78%, mis-cites 0.11 vs 1.00 claims/answer,
# refuses 5/5 unanswerable questions vs 3/5 (llama3.2 invented an Exxon Q3
# figure and a Roth IRA answer), at ~2x the latency. Use --model llama3.2 for speed.
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_K = 5
# Pinned to localhost on purpose — prompts may contain private portfolio data.
OLLAMA_HOST = "http://127.0.0.1:11434"
# Ollama "cloud" models (e.g. "gemma4:31b-cloud") are served through the same
# localhost address but run on Ollama's servers — the prompt (and any portfolio
# data in it) leaves the machine. They are refused.
# Upper bound on generated tokens: llama3.2 once looped to 27,000 characters.
MAX_ANSWER_TOKENS = 1024
# A generation that takes longer than this has hung; fail instead of blocking forever.
OLLAMA_TIMEOUT_SEC = 600
CLOUD_MODEL_RE = re.compile(r"(?:^|[:\-])cloud$", re.I)


class CloudModelError(ValueError):
    pass


def ensure_local_model(model: str) -> None:
    """Refuse Ollama cloud models (by name tag and by the server's own metadata) and a remote
    API server that wasn't explicitly allowed (PORTIQ_ALLOW_REMOTE_LLM=1)."""
    if backends.is_mlx(model):
        return  # MLX runs in-process on this Mac, bound to 127.0.0.1
    if backends.is_api(model):
        cfg = backends.api_config()
        if cfg is None:
            raise RuntimeError(f"'{model}' needs PORTIQ_LLM_URL and PORTIQ_LLM_MODEL to be set.")
        backends.check_api_allowed(cfg)
        return
    if CLOUD_MODEL_RE.search(model):
        raise CloudModelError(f"'{model}' is an Ollama cloud model — prompts would leave this machine. "
                              "FinRAG only runs local models (e.g. llama3.2, qwen2.5:7b).")
    try:
        import requests
        info = requests.post(f"{OLLAMA_HOST}/api/show", json={"model": model}, timeout=10).json()
    except Exception:
        return  # server unreachable: the generate call will report it
    if info.get("remote_host") or info.get("remote_model"):
        raise CloudModelError(f"'{model}' is served remotely ({info.get('remote_host')}) — refusing.")

# ── Prompt templates (per route — see prompts.py) ─────────────────────────────

def source_label(c: Citation) -> str:
    """Short per-tier label shown next to each [N] context block in the prompt."""
    if c.tier == 1:
        return f'"{c.doc_title}" — {c.source.title()}'
    if c.tier == 2:
        if "Annual Report FY" in c.doc_title and c.section_heading:  # which part of a long report
            return f"{c.doc_title} — {c.section_heading}"
        return c.doc_title  # already "AAPL 10-K filed 2025-10-31"
    arxiv = f" arXiv:{c.arxiv_id}" if c.arxiv_id else ""
    return f'"{c.doc_title}"{arxiv}'


def build_prompt(
    query: str,
    citations: list[Citation],
    positions: list[dict] | None = None,
    route: str = "concept",
) -> str:
    """
    Build the prompt for `route` (concept / technical / portfolio).

    [N] labels, source labels and chunk text all come from the same Citation,
    numbered by c.rank — the same list format_citations_block() prints — so
    an [N] in the answer always points at source [N] in the printed list.
    """
    return prompts.build_prompt(query, citations, source_label, route=route,
                                positions=positions, format_position=format_position)


# ── Embedder (lazy-loaded to keep startup fast) ──────────────────────────────

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        from build_vectorstore import EMBED_DEVICE
        import huggingface_hub.constants as hub_constants
        # Offline first: loading the cached model must not phone the Hugging Face Hub at question
        # time. local_files_only stops the update check, but huggingface_hub (1.x) still fetches an
        # "agent registry" from huggingface.co once a day while building request headers; only its
        # offline mode stops that. Online only on a first run, when the model isn't cached yet.
        was_offline, hub_constants.HF_HUB_OFFLINE = hub_constants.HF_HUB_OFFLINE, True
        try:
            model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE, local_files_only=True)
        except OSError:
            hub_constants.HF_HUB_OFFLINE = was_offline
            model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
            hub_constants.HF_HUB_OFFLINE = True  # downloaded once; stay offline from here on

        class _Embedder:
            def embed_query(self, text: str) -> list[float]:
                return model.encode([BGE_QUERY_PREFIX + text]).tolist()[0]

            def encode_texts(self, texts: list[str]):
                """Normalized passage-style embeddings (no query prefix)."""
                return model.encode(texts, normalize_embeddings=True)

        _embedder = _Embedder()
    return _embedder


# ── Retrieval: exact search over the vectors stored in Chroma ────────────────
# Chroma's approximate HNSW index missed true nearest neighbours (every SIP /
# YTM / CAGR chunk for some queries; recall@12 = 0.92 for "What is quantitative
# easing?" even with 32 neighbours). At this corpus size (~7k x 384 floats,
# ~10 MB) exact cosine search takes milliseconds, so queries compute it
# directly and Chroma is only the store. Revisit around ~100k chunks.

def _matches(meta: dict, where: dict | None) -> bool:
    """The subset of Chroma `where` syntax FinRAG uses: equality, $in, $and, $or."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(_matches(meta, w) for w in where["$or"])
    (key, cond), = where.items()
    if isinstance(cond, dict):
        if "$in" in cond:
            return meta.get(key) in cond["$in"]
        if "$eq" in cond:
            return meta.get(key) == cond["$eq"]
        raise ValueError(f"unsupported where operator: {cond}")
    return meta.get(key) == cond


def annual_titles(metadatas: list[dict], ticker: str) -> list[str]:
    """Titles of a company's annual filings (10-K) in the store. Indian annual reports return []:
    they are all annual already, so nothing needs filtering."""
    return sorted({m["doc_title"] for m in metadatas
                   if m.get("company_ticker") == ticker and " 10-K " in f" {m.get('doc_title', '')} "})


HYBRID_SUFFIXES = (".NS",)  # company tickers searched with hybrid retrieval (NSE annual reports)
HYBRID_DEPTH = 100   # candidates taken from each ranking before fusing
RRF_K = 60           # standard reciprocal-rank-fusion constant
BM25_K1, BM25_B = 1.5, 0.75
_STOPWORDS = set("a an and are as at be by did do does for from had has have how in is it its of on or "
                 "the their this to was were what when which who why will with end fy".split())


def _keyword_tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1 and w not in _STOPWORDS]


class ExactIndex:
    """Exact cosine search with Chroma's query() result shape (ids/documents/metadatas/distances)."""

    def __init__(self, collection: chromadb.Collection):
        stored = collection.get(include=["embeddings", "documents", "metadatas"])
        self.ids = stored["ids"]
        self.documents = stored["documents"]
        self.metadatas = stored["metadatas"]
        matrix = np.asarray(stored["embeddings"], dtype=np.float32)
        self.matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        self._masks: dict[str, np.ndarray] = {}
        self._bm25: dict[str, tuple] = {}

    def _mask(self, where: dict | None) -> np.ndarray:
        key = json.dumps(where, sort_keys=True)
        if key not in self._masks:
            self._masks[key] = np.array([_matches(m, where) for m in self.metadatas], dtype=bool)
        return self._masks[key]

    def query(self, query_embeddings: list[list[float]], n_results: int, where: dict | None = None) -> dict:
        q = np.asarray(query_embeddings[0], dtype=np.float32)
        q = q / np.linalg.norm(q)
        rows = np.flatnonzero(self._mask(where))
        sims = self.matrix[rows] @ q
        order = np.lexsort((rows, -sims))[:n_results]  # by similarity, ties broken by row: deterministic
        picked = rows[order]
        return {
            "ids": [[self.ids[i] for i in picked]],
            "documents": [[self.documents[i] for i in picked]],
            "metadatas": [[self.metadatas[i] for i in picked]],
            "distances": [[float(1.0 - s) for s in sims[order]]],  # Chroma's cosine distance
        }


    # ── Hybrid search (Day 18) ──────────────────────────────────────────────
    # In company filings the words carry the meaning ("consolidated revenues",
    # "total deposits"); the small embedding model ranked the passage with the
    # figure 7th–17th, below subsidiaries' tables. Fusing a keyword (BM25) ranking
    # with the dense one put it in the top 4 for 4 of 5 test figures.
    def _bm25_index(self, where: dict | None) -> tuple:
        key = json.dumps(where, sort_keys=True)
        if key not in self._bm25:
            rows = np.flatnonzero(self._mask(where))
            tfs = [Counter(_keyword_tokens(self.documents[i])) for i in rows]
            lengths = np.array([sum(tf.values()) for tf in tfs], dtype=np.float32)
            df = Counter(w for tf in tfs for w in tf)
            self._bm25[key] = (rows, tfs, lengths, df, float(lengths.mean()) if len(rows) else 1.0)
        return self._bm25[key]

    def hybrid_query(self, query_text: str, query_embeddings: list[list[float]], n_results: int,
                     where: dict | None = None, depth: int = HYBRID_DEPTH) -> dict:
        """Reciprocal-rank fusion of the exact dense ranking and a BM25 keyword ranking."""
        rows, tfs, lengths, df, avg_len = self._bm25_index(where)
        q = np.asarray(query_embeddings[0], dtype=np.float32)
        q = q / np.linalg.norm(q)
        sims = self.matrix[rows] @ q
        dense_order = np.lexsort((rows, -sims))
        terms = set(_keyword_tokens(query_text))
        n = len(rows)
        bm25 = np.zeros(n, dtype=np.float32)
        for w in terms:
            if df[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                tf = np.array([t[w] for t in tfs], dtype=np.float32)
                bm25 += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * lengths / avg_len))
        keyword_order = np.lexsort((rows, -bm25))
        fused: dict[int, float] = {}
        for order in (dense_order[:depth], keyword_order[:depth] if terms else []):
            for rank, pos in enumerate(order):
                fused[int(pos)] = fused.get(int(pos), 0.0) + 1.0 / (RRF_K + rank)
        picked = sorted(fused, key=lambda pos: (-fused[pos], rows[pos]))[:n_results]
        distances = [float(1.0 - sims[pos]) for pos in picked]
        return {
            "ids": [[self.ids[rows[p]] for p in picked]],
            "documents": [[self.documents[rows[p]] for p in picked]],
            "metadatas": [[self.metadatas[rows[p]] for p in picked]],
            "distances": [distances],
            # i-th fused result gets the i-th smallest distance: keeps fused order, distance scale
            "sort_keys": [sorted(distances)],
        }


_GENERIC_NAME_WORDS = {"limited", "ltd", "private", "pvt", "company", "inc", "llc", "gmbh", "the", "and"}


def _without_unasked_subsidiaries(res: dict, query: str, k: int) -> dict:
    """
    Drop a subsidiary's own statements ("Subsidiary: Jio Platforms Limited") unless the
    question names it. Even labelled, a 7B model quoted Jio's revenue as Reliance's; asking
    about "Jio" or "Reliance Retail" still gets them.
    """
    q = query.lower()
    keep = []
    for i, meta in enumerate(res["metadatas"][0]):
        heading = meta.get("section_heading") or ""
        if "Subsidiary:" in heading:
            name = heading.split("Subsidiary:", 1)[1].split(" (")[0]
            words = [w for w in re.findall(r"[a-z]{3,}", name.lower()) if w not in _GENERIC_NAME_WORDS]
            parent = (meta.get("doc_title") or "").lower().split()[0] if meta.get("doc_title") else ""
            if not any(w in q for w in words if w != parent):
                continue
        keep.append(i)
    keep = keep[:k]
    out = {key: [[vals[0][i] for i in keep]] for key, vals in res.items()}
    out["sort_keys"] = [sorted(out["distances"][0])]
    return out


_indexes: dict[int, ExactIndex] = {}


def exact_index(collection: chromadb.Collection) -> ExactIndex:
    if id(collection) not in _indexes:
        _indexes[id(collection)] = ExactIndex(collection)
    return _indexes[id(collection)]


def retrieve(
    query: str,
    collection: chromadb.Collection,
    k: int = DEFAULT_K,
    tier: int | None = None,
    ticker: str | None = None,
    tiers: list[int] | None = None,
    annual_only: bool = False,
) -> dict:
    """
    Embed the query and retrieve the exact top-k chunks (see ExactIndex).

    Args:
        query:      The user's natural language question.
        collection: An open Chroma collection handle.
        k:          Number of results to retrieve.
        tier:       If set, filter results to only this tier (1, 2, or 3).
        ticker:     If set, restrict to Tier 2 SEC filing chunks for this company.
        tiers:      If set, restrict to these tiers (e.g. [2, 3]).
    """
    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)

    if ticker is not None:
        where = {"$and": [{"tier": 2}, {"company_ticker": ticker}]}
        if annual_only:  # an annual question: the company's 10-K(s), when it has any
            annual = annual_titles(exact_index(collection).metadatas, ticker)
            if annual:
                where = {"$and": [{"tier": 2}, {"company_ticker": ticker}, {"doc_title": {"$in": annual}}]}
        # Indian annual reports: dense + keyword. Measured on the golden set, hybrid search
        # lifted annual-report figures into the top 4 but cost SEC filings MRR (0.94 -> 0.90:
        # quarterly 10-Q tables repeat "revenues" and outranked the 10-K), so SEC stays dense.
        if ticker.endswith(HYBRID_SUFFIXES):
            res = exact_index(collection).hybrid_query(query, [query_embedding], n_results=k * 2, where=where)
            return _without_unasked_subsidiaries(res, query, k)
    elif tier is not None:
        where = {"tier": tier}
    elif tiers:
        where = {"tier": tiers[0]} if len(tiers) == 1 else {"tier": {"$in": tiers}}
    else:
        where = None
    return exact_index(collection).query(query_embeddings=[query_embedding], n_results=k, where=where)


# ── Ollama LLM call ──────────────────────────────────────────────────────────

def _check_ollama() -> bool:
    """Return True if the ollama Python package is importable."""
    try:
        import ollama  # noqa: F401
        return True
    except ImportError:
        return False


def stream_tokens(prompt: str, model: str = DEFAULT_MODEL):
    """Yield answer tokens from the backend: Ollama, MLX ("mlx:" models) or the API server ("api:")."""
    model = backends.resolve(model)
    if backends.is_mlx(model):
        yield from backends.mlx_stream(prompt, model, MAX_ANSWER_TOKENS)
        return
    if backends.is_api(model):
        ensure_local_model(model)
        yield from backends.api_stream(prompt, model, MAX_ANSWER_TOKENS)
        return
    import ollama
    ensure_local_model(model)
    client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT_SEC)
    for chunk in client.generate(model=model, prompt=prompt, stream=True,
                                 options={"num_predict": MAX_ANSWER_TOKENS}):
        yield chunk.get("response", "")


def stream_answer(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Stream tokens from Ollama, print them to stdout in real-time, and return
    the complete answer string.
    """
    if not _check_ollama():
        print("\n[ERROR] 'ollama' Python package not installed.")
        print("  Run:  pip install ollama")
        return ""

    ensure_local_model(model)
    full_response = []
    print()  # blank line before answer
    try:
        for token in stream_tokens(prompt, model=model):
            print(token, end="", flush=True)
            full_response.append(token)
    except Exception as e:
        print(f"\n[ERROR] Ollama call failed: {e}")
        print(f"  Is 'ollama run {model}' running in another terminal?")
        return ""

    print()  # newline after streamed output
    return "".join(full_response)


def generate(prompt: str, model: str = DEFAULT_MODEL, options: dict | None = None) -> str:
    """Silent, non-streaming generation (used by get_answer and evaluate.py)."""
    model = backends.resolve(model)
    if backends.is_mlx(model) or backends.is_api(model):
        opts = options or {}
        if backends.is_api(model):
            ensure_local_model(model)
        run = backends.mlx_generate if backends.is_mlx(model) else backends.api_generate
        return run(prompt, model, opts.get("num_predict", MAX_ANSWER_TOKENS), opts.get("temperature", 0.0))
    import ollama
    ensure_local_model(model)
    response = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT_SEC).generate(model=model, prompt=prompt, stream=False,
                                                          options={"num_predict": MAX_ANSWER_TOKENS, **(options or {})})
    return response.get("response", "")


def get_answer(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Non-streaming version — prints the full answer at once."""
    if not _check_ollama():
        print("\n[ERROR] 'ollama' Python package not installed.")
        print("  Run:  pip install ollama")
        return ""

    import ollama

    try:
        answer = generate(prompt, model=model)
        print(f"\n{answer}")
        return answer
    except Exception as e:
        print(f"\n[ERROR] Ollama call failed: {e}")
        return ""


# ── Core RAG step ────────────────────────────────────────────────────────────

def _rerank(citations: list[Citation]) -> list[Citation]:
    citations = sorted(citations, key=order_key)
    for i, c in enumerate(citations, start=1):
        c.rank = i
    return citations


def retrieve_for_plan(query: str, collection: chromadb.Collection, plan: RetrievalPlan) -> list[Citation]:
    """
    Execute a router RetrievalPlan and return the final, deduped, ranked
    citations (the single list both the prompt and the printed sources use).
    """
    fetch = plan.k * plan.overfetch

    if plan.tickers:  # Tier 2 filings of specific companies, k distinct passages each
        print(f"🔍 Retrieving top-{plan.k} SEC filing chunks per company: {', '.join(plan.tickers)}…")
        if plan.annual_only and any(annual_titles(exact_index(collection).metadatas, t) for t in plan.tickers):
            print("↪ Annual question — reading the annual filings (10-K) only.")
        found: list[Citation] = []
        for t in plan.tickers:
            found += citations_from_chroma_results(retrieve(query, collection, k=fetch, ticker=t,
                                                            annual_only=plan.annual_only), preview_chars=150)
        citations = suppress_near_duplicates(deduplicate_citations(found), k=plan.k * len(plan.tickers))
        if not citations:
            print("[WARN] No filing chunks for those companies — falling back to all tiers.")
    else:
        citations = []

    if not citations:
        tiers = plan.tiers if not plan.tickers else None
        label = "all tiers" if tiers is None else "tier " + "+".join(map(str, tiers))
        print(f"🔍 Retrieving top-{plan.k} chunks ({label})…")
        found = citations_from_chroma_results(retrieve(query, collection, k=fetch, tiers=tiers), preview_chars=150)
        best = min((c.distance for c in found), default=None)
        if plan.distance_threshold is not None and not plan.fallback:
            found = [c for c in found if c.distance <= plan.distance_threshold]
            if not found:
                print(f"↪ No {label} passage within {plan.distance_threshold} — the corpus doesn't cover this.")
        elif plan.distance_threshold is not None and (best is None or best > plan.distance_threshold):
            fb = "all tiers" if plan.fallback_tiers is None else "tier " + "+".join(map(str, plan.fallback_tiers))
            shown = "none" if best is None else f"{best:.3f}"
            print(f"↪ Best {label} match {shown} > {plan.distance_threshold} — not covered there; searching {fb}…")
            found = citations_from_chroma_results(
                retrieve(query, collection, k=fetch, tiers=plan.fallback_tiers), preview_chars=150)
        citations = suppress_near_duplicates(deduplicate_citations(found), k=plan.k, max_per_doc=plan.max_per_doc)

    if plan.concept_k:  # portfolio "what is X" questions also get plain-language Tier 1 passages
        concept = citations_from_chroma_results(retrieve(query, collection, k=plan.concept_k * plan.overfetch, tiers=[1]),
                                                preview_chars=150)
        if plan.distance_threshold is not None:  # only if Tier 1 really covers it
            concept = [c for c in concept if c.distance <= plan.distance_threshold]
        concept = suppress_near_duplicates(deduplicate_citations(concept), k=plan.concept_k)
        seen = {c.chunk_id for c in citations}
        citations = _rerank(citations + [c for c in concept if c.chunk_id not in seen])
    return citations


def decide_route(query: str, route: str = "auto", use_portfolio: bool = True) -> RouteDecision:
    if route == "auto":
        return route_query(query, allow_portfolio=use_portfolio)
    return forced_route(route, query, allow_portfolio=use_portfolio)


def plan_query(
    query: str,
    k: int | None = None,
    tier: int | None = None,
    use_portfolio: bool = True,
    route: str = "auto",
) -> tuple[RouteDecision, RetrievalPlan, list[dict]]:
    """
    Route the query and resolve its retrieval plan and portfolio positions.
    Explicit flags override the route's plan: `k` replaces its k, `tier`
    searches only that tier (no portfolio filtering).
    """
    decision = decide_route(query, route, use_portfolio=use_portfolio and tier is None)
    plan = decision.plan
    if k is not None:
        plan.k = k
    if tier is not None:
        plan = RetrievalPlan(tiers=[tier], k=plan.k, overfetch=plan.overfetch)

    positions: list[dict] = []
    if decision.route == "portfolio":
        holdings = load_holdings()
        # every position in a held stock — one per source (manual file, Upstox, …), each with its as-of
        positions = ([h for h in holdings if h["ticker"] in decision.held_tickers]
                     if decision.held_tickers else holdings)
    return decision, plan, positions


def valuation_for(decision: RouteDecision, positions: list[dict]) -> list[dict] | None:
    """Portfolio performance questions: value the positions with local prices (computed by code)."""
    if decision.route != "portfolio" or not decision.wants_valuation or not positions:
        return None
    return prices.value_positions(positions, prices.load_prices())


def rag_query(
    query: str,
    collection: chromadb.Collection,
    model: str = DEFAULT_MODEL,
    k: int | None = None,
    tier: int | None = None,
    stream: bool = True,
    verbose_citations: bool = False,
    use_portfolio: bool = True,
    check: bool = True,
    route: str = "auto",
    verbose: bool = False,
) -> str:
    """
    Run one full RAG cycle: route → retrieve → build prompt → call LLM →
    print citations → check citations.

    The deterministic router (router.py) picks portfolio / technical / concept
    and its retrieval plan. Explicit flags override the plan: `k` replaces the
    route's k, and `tier` searches only that tier (no portfolio filtering).

    Returns the LLM answer string.
    """
    decision, plan, positions = plan_query(query, k=k, tier=tier, use_portfolio=use_portfolio, route=route)

    print(f"\n🧭 Route: {decision.route} (confidence {decision.confidence:.2f})"
          + (f" — also matched: {', '.join(decision.secondary_routes)}" if decision.secondary_routes else ""))
    if verbose:
        print(f"   {decision.describe()}")

    valued = valuation_for(decision, positions)
    citations = retrieve_for_plan(query, collection, plan)
    if not citations and not positions:
        print("[WARN] No relevant passages found — the corpus doesn't cover this question.")
        return ""

    # `citations` is now final (deduped, sorted, ranked). The prompt and the
    # printed source list below are both built from it — never from the raw
    # Chroma `results`, whose order differs after dedup.
    prompt = build_prompt(query, citations, positions=positions, route=decision.route)

    def print_positions() -> None:
        print("\nYour position(s), from local holdings:")
        for h in positions:
            print(f"  • {format_position(h)}")
        if valued:  # market value and gain — computed by prices.py, never by the model
            print("\nValuation (local prices, computed by code):")
            for line in prices.format_valuation(valued):
                print(f"  • {line}")

    if not citations:  # portfolio question the corpus doesn't cover — don't let the model guess
        if positions:
            print_positions()
        answer = prices.summary_sentence(valued) if valued else prompts.NO_CONTEXT_ANSWER
        print(f"\n{answer}")
        return answer

    print(f"💬 Querying {model}…")
    if positions:
        # Position facts are printed verbatim from holdings.csv — never written
        # (or recomputed) by the model.
        print_positions()
    if stream:
        answer = stream_answer(prompt, model=model)
    else:
        answer = get_answer(prompt, model=model)

    if decision.route == "portfolio" and answer:
        print(f"\n{prompts.PORTFOLIO_DISCLAIMER}")

    print()
    print(format_citations_block(citations, verbose=verbose_citations))
    if check and answer:
        report = check_citations(answer, citations, get_embedder().encode_texts,
                                 extra_sources=[query] + [format_position(h) for h in positions])
        print()
        print(format_citation_report(report))
    return answer


# ── CLI ──────────────────────────────────────────────────────────────────────

def local_models() -> list[str]:
    """Local models: Ollama's (excluding cloud models) plus MLX's when its server is up."""
    return [n for n in backends.available_models() if not CLOUD_MODEL_RE.search(n)]


def open_collection() -> chromadb.Collection | None:
    """Open the persistent Chroma collection, with helpful error messages."""
    if not CHROMA_DIR.exists():
        print(f"[ERROR] Chroma DB not found at {CHROMA_DIR}")
        print("  Run:  python scripts/build_vectorstore.py")
        return None
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"[ERROR] Could not open collection '{COLLECTION_NAME}': {e}")
        print("  Run:  python scripts/build_vectorstore.py")
        return None

    count = collection.count()
    if count == 0:
        print(f"[ERROR] Collection '{COLLECTION_NAME}' is empty.")
        print("  Run:  python scripts/build_vectorstore.py")
        return None

    print(f"✅ Loaded collection '{COLLECTION_NAME}' ({count} chunks)")
    return collection


def interactive_repl(
    collection: chromadb.Collection,
    model: str,
    k: int | None,
    tier: int | None,
    stream: bool,
    use_portfolio: bool = True,
    check: bool = True,
    route: str = "auto",
    verbose: bool = False,
) -> None:
    """Run an interactive question-answering REPL until the user exits."""
    print("\n" + "=" * 60)
    print("  finrag Interactive RAG Loop")
    print(f"  Model: {model}  |  k={k or 'per route'}  |  tier={tier or 'per route'}  |  route={route}")
    print("  Type 'exit' or Ctrl-C to quit, 'help' for commands.")
    print("=" * 60)

    while True:
        try:
            query = input("\n❓ Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "help":
            print(textwrap.dedent("""
                Commands:
                  exit / quit / q  — exit the REPL
                  help             — show this message
                  Any other text   — run as a RAG query
            """))
            continue

        rag_query(query, collection, model=model, k=k, tier=tier, stream=stream,
                  use_portfolio=use_portfolio, check=check, route=route, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="finrag RAG query loop — retrieve from ChromaDB, answer with Ollama."
    )
    parser.add_argument(
        "--query", "-q",
        type=str, default=None,
        help="Single question to answer (use quotes for multi-word queries).",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive REPL for multi-turn querying.",
    )
    parser.add_argument(
        "--k",
        type=int, default=None,
        help="Number of chunks to retrieve (default: per route — concept 4, technical 6, portfolio 5).",
    )
    parser.add_argument(
        "--tier",
        type=int, choices=[1, 2, 3], default=None,
        help="Restrict retrieval to a single corpus tier (1=basics, 2=SEC, 3=research).",
    )
    parser.add_argument(
        "--model",
        type=str, default=DEFAULT_MODEL,
        help=f"Ollama model name to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming; print the full answer at once.",
    )
    parser.add_argument(
        "--verbose-citations",
        action="store_true",
        help="Show URL and text preview under each source citation.",
    )
    parser.add_argument(
        "--no-portfolio",
        action="store_true",
        help="Ignore data/portfolio/holdings.csv (no portfolio-aware retrieval).",
    )
    parser.add_argument(
        "--route",
        choices=["auto", "concept", "technical", "portfolio"], default="auto",
        help="Query route (default: auto — deterministic rules in router.py).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the routing decision: matched signals and retrieval plan.",
    )
    parser.add_argument(
        "--no-citation-check",
        action="store_true",
        help="Skip checking the answer's [N] markers against their source passages.",
    )
    args = parser.parse_args()

    try:
        ensure_local_model(args.model)
    except CloudModelError as e:
        parser.error(str(e))

    if not args.query and not args.interactive:
        parser.error("Provide --query 'your question here' or --interactive to start the REPL.")

    collection = open_collection()
    if collection is None:
        sys.exit(1)

    stream = not args.no_stream

    if args.interactive:
        interactive_repl(collection, model=args.model, k=args.k, tier=args.tier, stream=stream,
                         use_portfolio=not args.no_portfolio, check=not args.no_citation_check,
                         route=args.route, verbose=args.verbose)
    else:
        rag_query(
            args.query,
            collection,
            model=args.model,
            k=args.k,
            tier=args.tier,
            stream=stream,
            verbose_citations=args.verbose_citations,
            use_portfolio=not args.no_portfolio,
            check=not args.no_citation_check,
            route=args.route,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
