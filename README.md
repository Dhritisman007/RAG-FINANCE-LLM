# FinRAG — Financial Knowledge RAG Agent

A fully local Retrieval-Augmented Generation (RAG) pipeline over a three-tier financial corpus — plain-language explainers, SEC filings (10-K / 10-Q) and arXiv market-microstructure papers — answered by a local LLM via Ollama, with optional portfolio-aware queries over your own holdings.

Built day-by-day as a structured learning project — each script corresponds to one pipeline stage.

> **Privacy by design:** everything runs on this machine. Embeddings run locally (sentence-transformers), the vector store is local ChromaDB with telemetry disabled, and generation goes only to the Ollama server on `localhost` (pinned in code, so `OLLAMA_HOST` can't redirect it). Portfolio data is never sent to any cloud API, and `holdings.csv` is git-ignored.

---

## Progress

| Day | Stage | Script | Status |
|-----|-------|--------|--------|
| 1 | Corpus manifest validation | `validate_manifest.py` | ✅ Done |
| 2 | Tier 1 explainer scraping | `scrape_tier1.py` | ✅ Done — 37 Investopedia docs |
| 3 | SEC EDGAR + arXiv scraping | `scrape_tier2_edgar.py`, `scrape_tier3_arxiv.py` | ✅ Done |
| 4 | Structure-aware chunking | `chunk_documents.py` | ✅ Done |
| 5 | Metadata enrichment | `enrich_metadata.py` | ✅ Done |
| 6 | ChromaDB vector store | `build_vectorstore.py` | ✅ Done |
| 7 | Retrieval sanity check | `sanity_check_retrieval.py` | ✅ Done — 15/15 queries pass |
| 8 | RAG query loop (Ollama) | `rag_loop.py`, `citations.py` | ✅ Done — citation bugs fixed + tested |
| — | Portfolio-aware queries | `portfolio_store.py`, `rag_loop.py` | ✅ Done |
| 9+ | UI, filtered retrieval, eval | — | ⬜ Not started |

---

## Architecture

```
corpus_manifest.yaml
        │
        ▼
 Day 2–3  Scrapers ──────────► data/raw/tier{1,2,3}/*.txt + *.meta.json
        │
        ▼
 Day 4    chunk_documents.py ─► data/processed/chunks.jsonl
        │
        ▼
 Day 5    enrich_metadata.py ─► data/processed/chunks_enriched.jsonl
        │
        ▼
 Day 6    build_vectorstore.py ► ChromaDB  (~/.finrag_chroma_db)
        │                         BAAI/bge-small-en-v1.5, cosine
        ▼
 Day 7    sanity_check_retrieval.py ► data/eval/retrieval_sanity_check.json
        │
        ▼
 Day 8    rag_loop.py ─► retrieve top-k → prompt → Ollama (llama3.2) → answer + sources
                ▲
                └── portfolio_store.py ◄── data/portfolio/holdings.csv  (structured lookup, not embedded)
```

---

## Corpus (current)

| Tier | Source | Content | Docs | Chunks |
|------|--------|---------|------|--------|
| 1 | Investopedia | Finance explainers (EMI/amortization, compound interest, SIP, bonds, options, ETFs, NPV, …) | 37 | 734 |
| 2 | SEC EDGAR | 10-K / 10-Q — AAPL, MSFT, JPM, TSLA, XOM | 15 | 5,541 |
| 3 | arXiv (q-fin) | Market microstructure, HFT, optimal execution | 79 | 3,267 |

**Pipeline stats:**
- 131 documents → 9,542 chunks
- 737 exact duplicates flagged and dropped → **8,805 chunks embedded**
- 466 chunks contain tables (4.9%) · 2,540 contain formulas (26.6%)

---

## What each stage does

### Day 1 — Manifest validation
Validates `config/corpus_manifest.yaml` (sources, topics, metadata schema) before scraping.

### Day 2 — Tier 1 scraping
Fetches one Investopedia article per manifest topic (37 topics, mapped in `INVESTOPEDIA_SLUG_MAP`). The extractor selects the article container *first* and strips nav/ads/boilerplate only inside it.

### Day 3 — SEC + arXiv scraping
Fetches filings from SEC EDGAR and papers from arXiv; writes `{slug}.txt` + `{slug}.meta.json` per document.

### Day 4 — Chunking
- Splits documents on `## Heading` markers first
- Paragraphs, list items and tables are kept whole and greedily packed into chunks (max ~450 tokens, min ~40)
- Oversized units fall back to sentence splitting (`was_sub_split=True`)
- `chunk_id` is unique per document + index (e.g. `sec_edgar_AAPL-10-K-filed-2025-10-31_45`)

### Day 5 — Metadata enrichment
Adds `contains_table`, `contains_formula`, `normalized_date`, `topic_tags` (keyword-based) and `content_hash` / `is_duplicate` (MD5 dedup).

### Day 6 — Vector store
Embeds non-duplicate chunks with `BAAI/bge-small-en-v1.5` (384-dim) into a persistent ChromaDB collection `finrag_chunks`. Metadata (`tier`, `company_ticker`, `topic_tags`, …) is flattened for `where`-filtered retrieval. Re-running fully rebuilds the collection, and the build fails loudly if any `chunk_id` is not unique.

> ChromaDB lives at `~/.finrag_chroma_db` — SQLite needs POSIX file locking, which exFAT external drives don't support.

### Day 7 — Retrieval sanity check
15 hand-picked queries across all tiers, with automated flags for tier drift, weak matches and short chunks. **All 15 return 5/5 results from the expected tier.**

| Tier | Queries | Top-1 distance |
|------|---------|----------------|
| 1 | 6 | 0.06–0.35 |
| 2 | 4 | 0.23–0.28 |
| 3 | 5 | 0.13–0.22 |

### Day 8 — RAG query loop
Retrieves top-k chunks, builds a grounded prompt with numbered sources, and streams an answer from local Ollama, followed by the source list. The prompt's `[N]` labels and the printed source list are built from the same final (deduped, ranked) citation list, so `[N]` in an answer always points to source `[N]`.

```bash
python scripts/rag_loop.py --query "What is latency arbitrage?" --tier 3
python scripts/rag_loop.py --interactive
python scripts/rag_loop.py --query "..." --model llama3.2 --k 6 --no-stream --verbose-citations
```

### Portfolio-aware queries
Holdings live in `finrag/data/portfolio/holdings.csv` (`ticker,quantity,avg_cost,purchase_date`) and are looked up as **structured data**, not embedded. When a query mentions a held ticker in uppercase (e.g. "my AAPL position"):

1. The position is loaded from `holdings.csv` via `portfolio_store.get_holding()`
2. Retrieval is restricted to that company's filings: `where={"$and": [{"tier": 2}, {"company_ticker": "AAPL"}]}`
3. The prompt includes a "My Portfolio" section plus the filtered filing chunks

```bash
python scripts/rag_loop.py --query "What risks does my AAPL position face?"
python scripts/rag_loop.py --query "..." --no-portfolio     # ignore holdings
python scripts/portfolio_store.py                           # list holdings
```

---

## Tests

```bash
python finrag/tests/test_citations.py
```

Covers the two fixed Day 8 citation bugs: distinct chunks from the same filing section aren't merged, and every `[N]` marker maps to the same chunk in the prompt and the printed list, even after dedup re-orders results.

---

## Known issues

**Fixed:**
- ~~Tier 1 empty~~ — the extractor stripped `<header>` (which wraps Investopedia's whole article) and `ad-` classes (which match the `<article>` itself) before selecting content; 3 slugs were also stale (404).
- ~~chunk_id collisions~~ — IDs were keyed on ticker, so a company's 10-K and 10-Qs collided and Chroma silently skipped ~1,900 chunks (store had 6,133 instead of 8,071).
- ~~Citation dedup over-collapses~~ — now keyed on `chunk_id`.
- ~~Prompt/label misalignment~~ — prompt and source list now come from one list.
- ~~Stray `scripts/"citations.py   "` draft~~ — deleted.
- ~~`ollama` missing from requirements~~ — added.

**Open:**
1. **EMI retrieval is weak** — "EMI" is an Indian-market term Investopedia doesn't use; the query hits the right tier, but the top chunk is about P/E ratios (dist 0.35). Needs an EMI-specific source (RBI/SEBI or a hand-written explainer).
2. **Near-duplicate SEC chunks** — EDGAR extraction sometimes repeats a paragraph (plus page footers like `Apple Inc. | 2025 Form 10-K | 6`); exact-MD5 dedup misses these, and they can take 2 of the top-k slots.
3. **Small Tier 1 chunks** — Investopedia sections are short and the chunker doesn't merge across headings (median 122 tokens; 86 chunks under 40).
4. **Tier 1 text includes trailing references/category tags** from each article.
5. **Portfolio detection is simple** — uppercase whole-word ticker match only; no live prices, so no P&L questions.
6. **Repo duplicates** — root-level `scrape_tier1.py`, `validate_manifest.py` and `corpus_manifest.yaml` are stale copies of the `finrag/` versions; `finrag/scripts/validate_manifest.py` is a stub. Old Chroma dirs (`finrag/data/chroma_db`, `finrag/data/vectorstore`) are unused.

---

## Setup

```bash
pip install -r finrag/requirements.txt
ollama pull llama3.2
```

### Run the pipeline

```bash
cd finrag
python scripts/scrape_tier1.py
python scripts/scrape_tier2_edgar.py
python scripts/scrape_tier3_arxiv.py
python scripts/chunk_documents.py
python scripts/enrich_metadata.py
python scripts/build_vectorstore.py --test-query "what is compound interest"
python scripts/sanity_check_retrieval.py
python scripts/rag_loop.py --interactive
```

---

## Roadmap

- [x] Fix Tier 1 scraping and rebuild the vector store
- [x] Fix citation dedup + prompt alignment
- [x] Portfolio-aware queries (local holdings + ticker-filtered retrieval)
- [ ] Add an EMI source; near-duplicate (fuzzy) dedup for SEC chunks
- [ ] Citation marker validation (map `[N]` → source, flag invalid/uncited markers)
- [ ] Streamlit / Gradio chat UI
- [ ] Metadata-filtered retrieval (tier-aware, table-aware) beyond tickers
- [ ] Portfolio: local price data for P&L / exposure questions
- [ ] Evaluation — recall@k, MRR against a golden QA set
