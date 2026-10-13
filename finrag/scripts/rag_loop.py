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
import sys
import textwrap
from pathlib import Path

import chromadb

# Local helper — same scripts/ directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))  # ensure scripts/ is on path
from citations import (  # noqa: E402
    Citation,
    citations_from_chroma_results,
    format_citations_block,
    deduplicate_citations,
)
from portfolio_store import find_held_tickers, format_position, get_holding, load_holdings  # noqa: E402

# ── Paths & constants ────────────────────────────────────────────────────────
CHROMA_DIR = Path.home() / ".finrag_chroma_db"
COLLECTION_NAME = "finrag_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_MODEL = "llama3.2"
DEFAULT_K = 5
# Pinned to localhost on purpose — prompts may contain private portfolio data.
OLLAMA_HOST = "http://127.0.0.1:11434"

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a helpful finance research assistant with access to curated financial \
documents, SEC filings, and market microstructure research papers. \
Answer the user's question accurately and concisely using ONLY the context \
provided below. If the context does not contain enough information, say so — \
do not make up facts. Cite the source numbers (e.g. [1], [2]) in your answer \
where relevant."""

PORTFOLIO_INSTRUCTIONS = """The "My Portfolio" section is the user's own position data from their local \
records. Refer to it as "your position" where relevant (e.g. size and cost \
basis), but cite [N] numbers only for facts taken from the numbered context \
passages. Do not invent current prices or returns — they are not provided."""

def source_label(c: Citation) -> str:
    """Short per-tier label shown next to each [N] context block in the prompt."""
    if c.tier == 1:
        return f'"{c.doc_title}" — {c.source.title()}'
    if c.tier == 2:
        return c.doc_title  # already "AAPL 10-K filed 2025-10-31"
    arxiv = f" arXiv:{c.arxiv_id}" if c.arxiv_id else ""
    return f'"{c.doc_title}"{arxiv}'


def build_prompt(query: str, citations: list[Citation], positions: list[dict] | None = None) -> str:
    """
    Build the full prompt string to send to the LLM.

    [N] labels, source labels and chunk text all come from the same Citation,
    numbered by c.rank — the same list format_citations_block() prints — so
    an [N] in the answer always points at source [N] in the printed list.
    """
    ctx_parts = [f"[{c.rank}] {source_label(c)}\n{c.text}" for c in citations]

    context_str = "\n\n".join(ctx_parts)

    portfolio_str = ""
    if positions:
        lines = "\n".join(f"- {format_position(h)}" for h in positions)
        portfolio_str = f"{PORTFOLIO_INSTRUCTIONS}\n\n=== My Portfolio ===\n{lines}\n\n"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{portfolio_str}"
        f"=== Context ===\n{context_str}\n\n"
        f"=== Question ===\n{query}\n\n"
        f"=== Answer ===\n"
    )


# ── Embedder (lazy-loaded to keep startup fast) ──────────────────────────────

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_NAME)

        class _Embedder:
            def embed_query(self, text: str) -> list[float]:
                return model.encode([BGE_QUERY_PREFIX + text]).tolist()[0]

        _embedder = _Embedder()
    return _embedder


# ── ChromaDB retrieval ───────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection: chromadb.Collection,
    k: int = DEFAULT_K,
    tier: int | None = None,
    ticker: str | None = None,
) -> dict:
    """
    Embed the query and retrieve the top-k chunks from Chroma.

    Args:
        query:      The user's natural language question.
        collection: An open Chroma collection handle.
        k:          Number of results to retrieve.
        tier:       If set, filter results to only this tier (1, 2, or 3).
        ticker:     If set, restrict to Tier 2 SEC filing chunks for this company.
    """
    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)

    if ticker is not None:
        where = {"$and": [{"tier": 2}, {"company_ticker": ticker}]}
    elif tier is not None:
        where = {"tier": tier}
    else:
        where = None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )
    return results


# ── Ollama LLM call ──────────────────────────────────────────────────────────

def _check_ollama() -> bool:
    """Return True if the ollama Python package is importable."""
    try:
        import ollama  # noqa: F401
        return True
    except ImportError:
        return False


def stream_answer(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Stream tokens from Ollama, print them to stdout in real-time, and return
    the complete answer string.
    """
    if not _check_ollama():
        print("\n[ERROR] 'ollama' Python package not installed.")
        print("  Run:  pip install ollama")
        return ""

    import ollama

    client = ollama.Client(host=OLLAMA_HOST)
    full_response = []
    print()  # blank line before answer
    try:
        for chunk in client.generate(model=model, prompt=prompt, stream=True):
            token = chunk.get("response", "")
            print(token, end="", flush=True)
            full_response.append(token)
    except Exception as e:
        print(f"\n[ERROR] Ollama call failed: {e}")
        print(f"  Is 'ollama run {model}' running in another terminal?")
        return ""

    print()  # newline after streamed output
    return "".join(full_response)


def get_answer(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Non-streaming version — prints the full answer at once."""
    if not _check_ollama():
        print("\n[ERROR] 'ollama' Python package not installed.")
        print("  Run:  pip install ollama")
        return ""

    import ollama

    try:
        response = ollama.Client(host=OLLAMA_HOST).generate(model=model, prompt=prompt, stream=False)
        answer = response.get("response", "")
        print(f"\n{answer}")
        return answer
    except Exception as e:
        print(f"\n[ERROR] Ollama call failed: {e}")
        return ""


# ── Core RAG step ────────────────────────────────────────────────────────────

def rag_query(
    query: str,
    collection: chromadb.Collection,
    model: str = DEFAULT_MODEL,
    k: int = DEFAULT_K,
    tier: int | None = None,
    stream: bool = True,
    verbose_citations: bool = False,
    use_portfolio: bool = True,
) -> str:
    """
    Run one full RAG cycle: retrieve → build prompt → call LLM → print citations.

    If the query mentions a held ticker (and no --tier was forced), retrieval
    is restricted to that company's SEC filing chunks and the position is
    added to the prompt.

    Returns the LLM answer string.
    """
    holdings = load_holdings() if use_portfolio and tier is None else []
    held = find_held_tickers(query, holdings)
    positions = [get_holding(t, holdings) for t in held]

    citations: list[Citation] = []
    if held:
        print(f"\n📁 Portfolio-aware query — position(s) from local holdings:")
        for h in positions:
            print(f"   {format_position(h)}")
        for t in held:
            print(f"🔍 Retrieving top-{k} {t} SEC filing chunks (tier=2, company_ticker={t})…")
            results = retrieve(query, collection, k=k, ticker=t)
            citations += citations_from_chroma_results(results, preview_chars=150)
        if not citations:
            print("[WARN] No filing chunks for those tickers — falling back to unfiltered retrieval.")

    if not citations:
        print(f"\n🔍 Retrieving top-{k} chunks…")
        results = retrieve(query, collection, k=k, tier=tier)
        citations = citations_from_chroma_results(results, preview_chars=150)

    citations = deduplicate_citations(citations)

    if not citations:
        print("[WARN] No results returned — is the vector store populated?")
        return ""

    # `citations` is now final (deduped, sorted, ranked). The prompt and the
    # printed source list below are both built from it — never from the raw
    # Chroma `results`, whose order differs after dedup.
    prompt = build_prompt(query, citations, positions=positions)

    print(f"💬 Querying {model}…")
    if stream:
        answer = stream_answer(prompt, model=model)
    else:
        answer = get_answer(prompt, model=model)

    print()
    print(format_citations_block(citations, verbose=verbose_citations))
    return answer


# ── CLI ──────────────────────────────────────────────────────────────────────

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
    k: int,
    tier: int | None,
    stream: bool,
    use_portfolio: bool = True,
) -> None:
    """Run an interactive question-answering REPL until the user exits."""
    print("\n" + "=" * 60)
    print("  finrag Interactive RAG Loop")
    print(f"  Model: {model}  |  k={k}  |  tier={tier or 'all'}")
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
                  use_portfolio=use_portfolio)


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
        type=int, default=DEFAULT_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_K}).",
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
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("Provide --query 'your question here' or --interactive to start the REPL.")

    collection = open_collection()
    if collection is None:
        sys.exit(1)

    stream = not args.no_stream

    if args.interactive:
        interactive_repl(collection, model=args.model, k=args.k, tier=args.tier, stream=stream,
                         use_portfolio=not args.no_portfolio)
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
        )


if __name__ == "__main__":
    main()
