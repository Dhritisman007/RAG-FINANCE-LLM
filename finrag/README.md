# FinRAG — Financial Knowledge RAG Agent

A fully local Retrieval-Augmented Generation (RAG) pipeline over a three-tier financial corpus — plain-language explainers, SEC filings (10-K / 10-Q) and arXiv market-microstructure papers — answered by a local LLM via Ollama, with optional portfolio-aware queries over your own holdings.

Built day-by-day as a structured learning project — each script corresponds to one pipeline stage.

> All paths and commands below are relative to `finrag/`.

> **Privacy by design:** everything runs on this machine. Embeddings run locally (sentence-transformers), the vector store is local ChromaDB with telemetry disabled, and generation goes only to the Ollama server on `localhost` (pinned in code, so `OLLAMA_HOST` can't redirect it). Portfolio data is never sent to any cloud API, and `holdings.csv` is git-ignored.

---

## Progress

| Day | Stage | Script | Status |
|-----|-------|--------|--------|
| 1 | Corpus manifest validation | `validate_manifest.py` | ✅ Done |
| 2 | Tier 1 explainer scraping | `scrape_tier1.py` | ✅ Done — 38 Investopedia docs |
| 3 | SEC EDGAR + arXiv scraping | `scrape_tier2_edgar.py`, `scrape_tier3_arxiv.py` | ✅ Done |
| 4 | Structure-aware chunking | `chunk_documents.py` | ✅ Done |
| 5 | Metadata enrichment | `enrich_metadata.py` | ✅ Done |
| 6 | ChromaDB vector store | `build_vectorstore.py`, `check_index_recall.py` | ✅ Done — index recall@5 = 1.0 |
| 7 | Retrieval sanity check | `sanity_check_retrieval.py` | ✅ Done — 15/15 queries pass |
| 8 | RAG query loop (Ollama) | `rag_loop.py`, `citations.py` | ✅ Done — with automatic citation checking |
| — | Portfolio-aware queries | `portfolio_store.py`, `rag_loop.py` | ✅ Done |
| 10 | Deterministic query router | `router.py` | ✅ Done — 33/33 labelled queries, 15/15 traps |
| 11 | Per-route prompt templates | `prompts.py` | ✅ Done — + number-grounding check |
| 12+ | Eval set, UI, model comparison | — | ⬜ Not started |

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
 Day 8–11 rag_loop.py ─► route (router.py) → retrieve per route plan → route prompt (prompts.py)
                │          → Ollama (llama3.2) → answer + sources → citation & number check
                ▲
                └── portfolio_store.py ◄── data/portfolio/holdings.csv  (structured lookup, not embedded)
```

---

## Corpus (current)

| Tier | Source | Content | Docs | Chunks |
|------|--------|---------|------|--------|
| 1 | Investopedia | Finance explainers (EMI, amortization, compound interest, SIP, bonds, options, ETFs, NPV, …) | 38 | 758 |
| 2 | SEC EDGAR | 10-K / 10-Q — AAPL, MSFT, JPM, TSLA, XOM | 15 | 3,205 (2,927 after dedup) |
| 3 | arXiv (q-fin) | Market microstructure, HFT, optimal execution | 79 | 3,267 |

**Pipeline stats:**
- 132 documents → 7,230 chunks
- 278 exact duplicates flagged and dropped → **6,952 chunks embedded**
- 821 chunks contain tables (11.4%) · 2,509 contain formulas (34.8%)

---

## What each stage does

### Day 1 — Manifest validation
Validates `config/corpus_manifest.yaml` (sources, topics, metadata schema) before scraping.

### Day 2 — Tier 1 scraping
Fetches one Investopedia article per manifest topic (38 topics, mapped in `INVESTOPEDIA_SLUG_MAP`). The extractor selects the article container *first* and strips nav/ads/boilerplate only inside it.

### Day 3 — SEC + arXiv scraping
Fetches filings from SEC EDGAR and papers from arXiv; writes `{slug}.txt` + `{slug}.meta.json` per document.

The EDGAR extractor emits each piece of text exactly once: only innermost block elements (`<span>` is treated as inline), each table as a single flattened line, and the hidden `ix:header` XBRL block dropped. Running headers/footers (short lines repeating 5+ times, digits ignored — e.g. `Apple Inc. | 2025 Form 10-K | 6`) are removed. `--reextract` re-fetches the filings already on disk (by the URL in each `.meta.json`) to apply extractor fixes without changing which filings are in the corpus.

### Day 4 — Chunking
- Splits documents on `## Heading` markers first
- Paragraphs, list items and tables are kept whole and greedily packed into chunks (max ~450 tokens, min ~40)
- Oversized units fall back to sentence splitting, then a word-boundary hard split for text with no sentences, like flattened tables (`was_sub_split=True`)
- `chunk_id` is unique per document + index (e.g. `sec_edgar_AAPL-10-K-filed-2025-10-31_45`)

### Day 5 — Metadata enrichment
Adds `contains_table`, `contains_formula`, `normalized_date`, `topic_tags` (keyword-based) and `content_hash` / `is_duplicate` (MD5 dedup).

### Day 6 — Vector store
Embeds non-duplicate chunks with `BAAI/bge-small-en-v1.5` (384-dim) into a persistent ChromaDB collection `finrag_chunks`. Metadata (`tier`, `company_ticker`, `topic_tags`, …) is flattened for `where`-filtered retrieval. Re-running fully rebuilds the collection, and the build fails loudly if any `chunk_id` is not unique.

The HNSW index uses `max_neighbors=32, ef_construction=400, ef_search=200` (Chroma's default 16 neighbours left whole topic clusters unreachable). Verify after any rebuild:

```bash
python scripts/check_index_recall.py   # recall@5 of the index vs exact search, ~200 queries
```

> ChromaDB lives at `~/.finrag_chroma_db` — SQLite needs POSIX file locking, which exFAT external drives don't support.

### Day 7 — Retrieval sanity check
15 hand-picked queries across all tiers, with automated flags for tier drift, weak matches and short chunks. **All 15 return 5/5 results from the expected tier.**

| Tier | Queries | Top-1 distance |
|------|---------|----------------|
| 1 | 6 | 0.06–0.21 |
| 2 | 4 | 0.22–0.36 |
| 3 | 5 | 0.13–0.22 |

### Day 8 — RAG query loop
Retrieves top-k chunks, builds a grounded prompt with numbered sources, and streams an answer from local Ollama, followed by the source list. It over-fetches 3×k candidates and greedily skips near-copies (5-word-shingle Jaccard ≥ 0.8 to an already-chosen passage, e.g. a 10-Q restating a 10-K risk factor), so each answer gets k distinct passages. The prompt's `[N]` labels and the printed source list are built from the same final (deduped, ranked) citation list, so `[N]` in the prompt always matches source `[N]`.

The prompt requires a `[N]` after every sentence that uses the context (before this, llama3.2 left half of its answers with no citations at all).

**Citation check.** After each answer, every `[N]` marker is checked against its passage. A marker covers the text before it (`claim [2].`, `… (Source: [2])`) or, for lead-ins, the text after it (`As stated in [2], claim.`). Each claim is scored against every sentence of every passage with the local bge model, and the report flags:

| Flag | Meaning |
|------|---------|
| `⚠ [4] likely should be [5]` | another passage supports the claim much better (≥ 0.05 higher and ≥ 0.80) |
| `? [4] weakly supports this` | best cited support < 0.70 — the claim may not be in the sources |
| `✗ [7] is not one of the listed sources` | marker number doesn't exist |
| `⚠ the answer has no [N] citations` | nothing can be traced |

Thresholds were calibrated on llama3.2 answers; every flag spot-checked against the passage text was a real mis-citation (e.g. swapped `[4]`/`[5]`, a made-up "interest rate swaps" detail).

```bash
python scripts/rag_loop.py --query "What is latency arbitrage?" --tier 3
python scripts/rag_loop.py --interactive
python scripts/rag_loop.py --query "..." --model llama3.2 --k 6 --no-stream --verbose-citations
python scripts/rag_loop.py --query "..." --no-citation-check
```

### Day 10 — Deterministic query router
`router.py` routes every query with plain rules — no LLM call (a 3B model is unreliable at this):

| Route | Triggered by | Retrieval plan |
|-------|--------------|----------------|
| **portfolio** | a held ticker (`AAPL`) or held company name (`Apple`), or "my portfolio / position / shares / funds…", "I own…" | held companies' Tier 2 filings (k=5 each) + up to 2 Tier 1 passages for "what is X" questions; no ticker → Tier 1 only, nothing if not covered |
| **technical** | microstructure terms (limit order book, latency, price impact…), filing terms (10-K, risk factors, revenue…), a non-held corpus company, "calculate … with numbers", or an ambiguous word next to a modelling word | Tiers 2+3, k=6; a named company → its filings only |
| **concept** | everything else | Tier 1, k=4; falls back to all tiers if the best Tier 1 hit is worse than 0.23 |

Priority is portfolio > technical > concept; lower routes that also matched are reported as `secondary_routes`. Ambiguous words (duration, spread, options, formula, volatility, liquidity, arbitrage) only count as technical next to a modelling word ("bid-ask spread **modeled**", "option pricing **models**"). Thresholds were measured on the live store: Tier 1 covers a question when its best hit is ≤ 0.206 and doesn't when ≥ 0.257 (concept threshold 0.23); portfolio questions carry a personal clause that adds distance, so their Tier 1 material uses 0.28.

```bash
python scripts/rag_loop.py --query "..." --route technical      # auto | concept | technical | portfolio
python scripts/rag_loop.py --query "..." --verbose              # print route, signals and retrieval plan
python scripts/router.py "explain duration for my bond fund"    # route only, no retrieval
```

`--k` and `--tier` still work and override the route's plan; `--no-portfolio` never routes to portfolio.

### Day 11 — Per-route prompt templates
`prompts.py` — every template shares the grounding rules (answer only from the numbered context, cite `[N]` after every sentence, every number must appear in the context, say explicitly when the context doesn't cover the question) and the same `[N]` labelling as the printed source list.

| Route | Style |
|-------|-------|
| concept | plain language, ~150 words, define jargon, at most one worked example and only one from the context |
| technical | formulas and table values verbatim, never rounded, units and periods kept, source assumptions stated |
| portfolio | filing-based risks with citations; never restates or computes position numbers, never says buy/sell/hold |

Portfolio facts are never written by the model: the position lines are printed verbatim from `holdings.csv` above the answer, and the "educational information, not investment advice" note is appended by code. When retrieval finds nothing (e.g. "How is my portfolio doing?" — the corpus has no prices), the model isn't called at all; a fixed "the sources don't cover this" answer is printed instead (llama3.2 invented a `[1]` source in that case).

**Number check.** The citation check also lists every number in the answer that appears in none of the passages, the question or the positions — e.g. an invented "$1,000 at 5% → $52.50" example inside an otherwise well-cited answer, which semantic scores can't see.

### Portfolio-aware queries
Holdings live in `data/portfolio/holdings.csv` (`ticker,quantity,avg_cost,purchase_date`) and are looked up as **structured data**, not embedded. When the router picks the portfolio route (e.g. "my AAPL position", "Is Apple a good company?"):

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

Plain scripts (no pytest needed):

```bash
python tests/test_citations.py   # citation labels, dedup, checker, number check
python tests/test_router.py      # 33 labelled queries (accuracy table), mixed queries, plans, flags
python tests/test_prompts.py     # grounding/style rules per route, [N] label order, no-context answer
```

`test_citations.py` covers: distinct chunks from the same filing section aren't merged; every `[N]` marker maps to the same chunk in the prompt and the printed list, even after dedup re-orders results; near-copy chunks are skipped and the slot is backfilled; citation parsing (`[N]`, `[a][b]`, `(Source: [N])`, lead-ins, formulas, reference lists); and the checker flags swapped, invalid and missing citations.

---

## Known issues

**Fixed:**
- ~~Tier 1 empty~~ — the extractor stripped `<header>` (which wraps Investopedia's whole article) and `ad-` classes (which match the `<article>` itself) before selecting content; 3 slugs were also stale (404).
- ~~chunk_id collisions~~ — IDs were keyed on ticker, so a company's 10-K and 10-Qs collided and Chroma silently skipped ~1,900 chunks (store had 6,133 instead of 8,071).
- ~~Citation dedup over-collapses~~ — now keyed on `chunk_id`.
- ~~Prompt/label misalignment~~ — prompt and source list now come from one list.
- ~~Stray `scripts/"citations.py   "` draft~~ — deleted.
- ~~`ollama` missing from requirements~~ — added.
- ~~EMI retrieval weak~~ — no chunk contained "EMI" (the amortization article was only *labelled* EMI). Added Investopedia's dedicated EMI article and relabelled the old one "loan amortization"; top hit went from P/E ratio (0.35) to "How Is EMI calculated?" (0.21).
- ~~HNSW index hid whole topics~~ — with Chroma's default index settings every SIP / YTM / CAGR chunk was unreachable for some queries, even at `ef_search=1000` (5 of 199 test queries got 0/5 of their true top-5). Rebuilt with 32 neighbours: recall@5 = 1.0 on all 199.
- ~~Prompt example leaked into answers~~ — the citation rule's example ("An EMI is a fixed monthly loan payment [1].") was copied into a bid-ask-spread answer; it's now a content-free `<sentence> [2]`.
- ~~Hallucinated source on empty context~~ — with no passages llama3.2 cited a non-existent `[1]`; the model is no longer called when there is no context.
- ~~Invented example numbers undetected~~ — now flagged by the number check.
- ~~Near-duplicate SEC chunks~~ — the EDGAR extractor emitted both `<div>` and its inner `<span>`, doubling 23% of lines (up to 51% in some filings), split tables into one line per cell, and kept page headers/footers and the hidden XBRL header (a 58k-token chunk). Fixed at extraction; SEC text dropped from 136k to 15k lines with no content words lost. Cross-filing restatements (10-Q repeating 10-K text — real content, kept in the corpus) are suppressed at retrieval time instead.

**Open:**
1. **Mislabelled table numbers aren't detected** — a 3B model reading flattened filing tables picks the wrong row or column: asked for Exxon's revenue *last quarter*, it answered "$10.1 billion" (Q1 2025 *production expenses*) and, on a re-run, "339,247 million" (2024 *annual* revenue). The number exists in the source, so the number check passes it. Needs a stronger model or structured table parsing.
2. **Units dropped from table figures** — MSFT segment revenue was quoted as "$139,996" without the table's "(In millions)", despite the template rule.
3. **llama3.2 mis-cites ~20% of claims** — now *detected and reported* by the citation check (8 of 39 claims across 8 test answers), but not corrected; the answer text still carries the wrong number. A larger local model (e.g. `--model qwen2.5:7b`) or auto-correcting confident suggestions are the next steps.
4. **Low source diversity** — top-k can come entirely from one document (e.g. all 5 latency-arbitrage sources are chunks of one arXiv paper). A per-document cap in retrieval would broaden answers.
5. **Small Tier 1 chunks** — Investopedia sections are short and the chunker doesn't merge across headings (median 124 tokens).
6. **Tier 1 text includes trailing references/category tags** from each article; one Tier 3 chunk is a PDF artifact (a run of 13k `●` characters).
7. **No price data** — the router recognises held tickers and company names, but there are no prices, so performance / P&L questions get the "not covered" answer.
8. **Repo duplicates** — root-level `scrape_tier1.py`, `validate_manifest.py` and `corpus_manifest.yaml` are stale copies of the `finrag/` versions; `scripts/validate_manifest.py` is a stub.
9. **Stale vector stores (not deleted, git-ignored)** — `data/chroma_db/` (114 MB, Aug 13: the pre-fix build with only 6,133 chunks) and `data/vectorstore/` (empty). Nothing references either. `~/.finrag_chroma_db` also keeps orphaned index folders from earlier rebuilds.

---

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2
```

### Run the pipeline

All commands run from `finrag/`.

```bash
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
- [x] Fix near-duplicate SEC chunks (extractor + retrieval-time suppression)
- [x] Add an EMI source
- [x] Fix HNSW index recall
- [x] Citation checking (flag mis-cited, weak, invalid and missing `[N]` markers)
- [x] Deterministic query router (Day 10)
- [x] Per-route prompt templates + number-grounding check (Day 11)
- [ ] Per-document cap for retrieval diversity
- [ ] Compare a larger local model (e.g. `qwen2.5:7b`) on table questions and citation accuracy
- [ ] Streamlit / Gradio chat UI
- [ ] Metadata-filtered retrieval (tier-aware, table-aware) beyond tickers
- [ ] Portfolio: local price data for P&L / exposure questions
- [ ] Evaluation — recall@k, MRR against a golden QA set
