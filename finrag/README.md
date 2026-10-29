# PortIQ (FinRAG) — Financial Knowledge RAG Agent

**PortIQ** — *Upload. Ask. Get Insights.* — is the app's name; FinRAG is the retrieval pipeline underneath (scripts, store and evaluation keep that name).

A fully local Retrieval-Augmented Generation (RAG) pipeline over a three-tier financial corpus — plain-language explainers, company documents (US SEC 10-K / 10-Q filings and Indian annual reports) and arXiv market-microstructure papers — answered by a local LLM (Ollama, or Apple's MLX as the alternative), with a live portfolio tracker for Indian (NSE/BSE) and US stocks in ₹ and portfolio-aware queries over your own holdings.

Built day-by-day as a structured learning project — each script corresponds to one pipeline stage.

> All paths and commands below are relative to `finrag/`.

> **Privacy by design:** everything runs on this machine. Embeddings run locally (sentence-transformers), the vector store is local ChromaDB with telemetry disabled, and generation goes only to the Ollama or MLX server on `127.0.0.1` (pinned in code, so `OLLAMA_HOST` can't redirect it). Portfolio data is never sent to any cloud API, and `holdings.csv` is git-ignored. **Deliberate exceptions:** live prices come from Yahoo Finance, which sees the stock symbols you hold — never your quantities or prices paid; and an optional broker sync (`portfolio_sync.py`) reads your holdings from your own broker account, contacting only that broker's host. Answering a question makes no outbound connection.

---

## Progress

| Day | Stage | Script | Status |
|-----|-------|--------|--------|
| 1 | Corpus manifest validation | `validate_manifest.py` | ✅ Done |
| 2 | Tier 1 explainer scraping | `scrape_tier1.py` | ✅ Done — 38 Investopedia docs |
| 3 | SEC EDGAR + arXiv scraping | `scrape_tier2_edgar.py`, `scrape_tier3_arxiv.py` | ✅ Done |
| 4 | Structure-aware chunking | `chunk_documents.py` | ✅ Done |
| 5 | Metadata enrichment | `enrich_metadata.py` | ✅ Done |
| 6 | Vector store + exact search | `build_vectorstore.py`, `rag_loop.ExactIndex` | ✅ Done — exact cosine search, CPU embeddings |
| 7 | Retrieval sanity check | `sanity_check_retrieval.py` | ✅ Done — 15/15 queries pass |
| 8 | RAG query loop (Ollama) | `rag_loop.py`, `citations.py` | ✅ Done — with automatic citation checking |
| — | Portfolio-aware queries | `portfolio_store.py`, `rag_loop.py` | ✅ Done |
| 10 | Deterministic query router | `router.py` | ✅ Done — 33/33 labelled queries, 15/15 traps |
| 11 | Per-route prompt templates | `prompts.py` | ✅ Done — + number-grounding check |
| 12 | Golden eval set + model choice | `evaluate.py`, `data/eval/golden_qa.jsonl` | ✅ Done — qwen2.5:7b chosen (92% citation accuracy) |
| 13 | Chat UI (Streamlit) | `app.py` | ✅ Done — tested on every route, localhost-only |
| 14 | Local price data + valuation | `prices.py` | ✅ Done — whole-market download, no tickers ever sent |
| 15 | Corpus expansion: 25 companies | `corpus_manifest.yaml`, `router.py` | ✅ Done — 75 SEC filings, 16,946 passages |
| 16 | Tier 1 gaps + filing refresh | `scrape_tier1.py`, `refresh_filings.py` | ✅ Done — 47 explainers; refresh found a missed JPM 10-Q |
| 18 | Indian annual reports + dashboard UI | `scrape_tier2_india.py`, `router.py`, `app.py` | ✅ Done — 15 NSE companies (FY2025-26), two-pane research terminal |
| 17 | Live portfolio (NSE/BSE, ₹) + MLX backend | `app.py`, `prices.py`, `portfolio_store.py`, `llm_backends.py` | ✅ Done — add stocks in the UI, live prices every 60 s, MLX fallback when Ollama is down |
| 20 | Portfolio sources: CSV import + read-only broker sync | `scripts/brokers/`, `portfolio_sync.py`, `app.py` | ✅ Done — Upstox / Dhan / Groww fixture-tested, not yet on a live account |
| 21 | CAS statement import (all brokers) | `brokers/cas_adapter.py`, `app.py` | ✅ Done — not yet tried on a real statement |
| 22 | Website groundwork: sign-in, per-user data, privacy notice | `workspace.py`, `app.py` | ✅ Done locally — not deployed yet |

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
                │          → Ollama (qwen2.5:7b) → answer + sources → citation & number check
                ▲
                └── portfolio_store.py ◄── data/portfolio/holdings.csv  (structured lookup, not embedded)
```

---

## Corpus (current)

| Tier | Source | Content | Docs | Chunks |
|------|--------|---------|------|--------|
| 1 | Investopedia | Finance explainers (EMI, amortization, compound interest, SIP, bonds, options, ETFs, NPV, IRAs, 401(k), order types, …) | 47 | 936 |
| 2 | SEC EDGAR | 25 companies: latest 10-K + 2 10-Qs, plus newer filings via refresh (Day 16) | 76 | 15,309 |
| 3 | arXiv (q-fin) | Market microstructure, HFT, optimal execution | 79 | 3,267 |

**Pipeline stats:**
- 202 documents → 19,512 chunks
- 2,037 exact duplicates flagged and dropped (mostly boilerplate repeated across a company's 10-K and 10-Qs) → **17,475 chunks embedded**
- 2,089 chunks contain tables (11.0%) · 3,292 contain formulas (17.4%)

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

**Queries use exact search, not Chroma's index.** Chroma's approximate HNSW index missed true nearest neighbours — every SIP / YTM / CAGR chunk for some queries with default settings, and still recall@12 = 0.92 for "What is quantitative easing?" with 32 neighbours. At ~7k chunks × 384 dims (~10 MB) exact cosine search takes ~5 ms, so `rag_loop.ExactIndex` loads the stored vectors once and searches them directly (same result shape and distances as Chroma; deterministic tie-breaking). Chroma remains the store. Revisit around ~100k chunks; `scripts/check_index_recall.py` still measures Chroma's own index.

**Embeddings always run on the CPU** (`EMBED_DEVICE` in `build_vectorstore.py`). On an 8 GB M1 the GPU (MPS) shares memory with Ollama; with qwen2.5:7b loaded, Metal ran out of memory and PyTorch *silently* returned corrupted embeddings (cosine vs. the correct vector as low as 0.28) — unrelated retrieval and flipping citation checks, no exception raised. All 6,952 stored vectors were verified against a CPU recompute (cosine 1.0).

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

The prompt requires a `[N]` after every sentence that uses the context (before this, llama3.2 left half of its answers with no citations at all). The default model is **qwen2.5:7b** (see Day 12); Ollama *cloud* models (e.g. `gemma4:31b-cloud`) are refused — they answer on localhost but run remotely, so prompts with portfolio data would leave the machine.

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

### Day 12 — Evaluation and model choice
`data/eval/golden_qa.jsonl`: 43 questions (16 concept, 16 technical, 11 portfolio; 5 unanswerable), each labelled with its relevant documents — chosen from the raw text, not from retrieval output — and, for factual questions, the exact figure the answer must contain (e.g. ExxonMobil 2025 sales 323,905). `scripts/evaluate.py` runs every question through the same code path as `rag_loop.py`.

**Retrieval** (model-independent): route accuracy 100%, hit@k 100%, MRR 0.93.

**Answers** — both models with identical settings (`temperature 0, seed 0`, 1,024-token cap) and identical retrieval on all 43 questions:

| | llama3.2 (3B) | **qwen2.5:7b** |
|---|---|---|
| Citation accuracy (per answer) | 78% | **92%** |
| Mis-cited claims per answer | 1.00 | **0.11** |
| Numbers in no source, per answer | 0.39 | **0.18** |
| Required figures present ¹ | 5/6 | **6/6** |
| Unanswerable questions refused | 3/5 | **5/5** |
| Answerable questions refused | 0 | 0 |
| Answers without any citation | 4 | **0** |
| Time per answer (8 GB M1) | ~70 s | ~140 s |

¹ Corrected on Day 15: a required figure may be given in any form that appears verbatim in the filing (NVIDIA's 10-K states both "$215.9 billion" and "$ 215,938" million; Tesla's both "94,827" and "$94.83 billion"). The original key accepted only one form and marked verbatim quotes as wrong; a test now checks every accepted form appears in its filing.

llama3.2's misses on unanswerable questions were the dangerous kind: it invented "Q3 2026 revenue $116,017 million" for ExxonMobil (the corpus ends at Q2) and wrote a Roth IRA explanation with fabricated citations. qwen2.5:7b refused both — for Exxon it said Q3 isn't covered and offered the closest quarter, clearly labelled as Q1. **qwen2.5:7b is the default**; `--model llama3.2` is twice as fast.

How to read the metrics: citation accuracy is averaged per answer; a *refusal* is an answer that leads with "the sources don't cover this" (citations allowed); a cited answer that notes a gap later is a partial answer, not a refusal. Several scoring bugs were found by reading actual answers and fixed before the final comparison (e.g. arXiv ids counted as invented numbers; a runaway 27,000-character answer dominating per-claim scores).

**Per-document cap.** Broad technical questions prefer ≤ 2 passages per paper (soft cap — narrow questions still get k passages). On 10 such questions (llama3.2) citation accuracy was 70% with the cap vs 62% without: weak evidence in favour, so it's kept.

```bash
python scripts/evaluate.py                                   # retrieval only (fast, no LLM)
python scripts/evaluate.py --answers --model qwen2.5:7b      # + answers (~1.5 h on an 8 GB M1)
python scripts/evaluate.py --answers --resume                # continue an interrupted run
python scripts/evaluate.py --rescore data/eval/eval_answers_qwen2.5-7b.json
```

Answer runs checkpoint after every question: one run stalled for 50 minutes inside ChromaDB (intermittent, under memory pressure) and lost everything before checkpointing existed. Ollama calls also time out after 10 minutes.

### Day 13 — Chat UI
```bash
cd finrag && streamlit run app.py      # http://localhost:8501
```

`app.py` runs the same pipeline as `rag_loop.py` — router → retrieval plan → route prompt → local Ollama → sources → citation & number check — in a chat window:

- **Route first, then answer:** the route (and any "not covered in Tier 1 → searching all tiers" fallback) shows immediately; a spinner covers the model load and reading time (up to a minute for the first answer on an 8 GB M1), then the answer streams.
- **Your holdings in the sidebar**; on portfolio questions the position lines are printed verbatim from `holdings.csv` and the not-investment-advice note is added by code.
- **Sources** as expandable passages (with URLs), and the **citation check** inline (✓ / likely-miscited with the better source / weak / numbers found in no source).
- **Honest refusals:** a "the sources don't cover this" answer shows its sources as *closest passages found — not enough to answer* and skips the citation warning.
- **One question at a time:** the input is locked while an answer generates — otherwise a new question restarts the script and silently drops the answer in progress.
- Model picker (local models only), route override, holdings / citation-check / routing-details toggles.

**Local only, enforced** (`.streamlit/config.toml`): bound to `127.0.0.1` (Streamlit's default listens on every interface — LAN-visible holdings), usage statistics off, the "Deploy" button hidden (one click would publish the app to Streamlit Cloud), and the app refuses to start if the address or telemetry settings are changed. The file watcher is off: it imported dozens of `transformers` vision modules on every interaction.

### Day 14 — Local price data and valuation
"How is my portfolio doing?" now gets a real answer — **computed by code, not the model**:

```
AAPL: 25 × $343.46 = $8,586.50 (cost $4,562.50, +$4,024.00, +88.2%) — price as of 2026-10-19T17:04
...
Your priced holdings are worth $18,972.80 against a cost basis of $11,590.50: +$7,382.30 (+63.7%) ...
```

**Privacy (deliberate):** prices come from one download of Nasdaq's public stock screener covering the whole US market (~7,000 stocks). The request URL is fixed and contains **no tickers**, so it reveals nothing about what you hold (a test asserts no holding ever appears in the request). It only runs when you ask:

```bash
python scripts/prices.py --update     # download whole-market prices (or the UI's "Update prices" button)
python scripts/prices.py              # value your holdings
```

- The router flags portfolio *performance* questions ("how is my portfolio doing", "what is my portfolio worth", "P&L on MSFT"); those show a valuation table built by `prices.py`. If the corpus has nothing else relevant, the answer is a code-generated summary — no LLM call. Otherwise the model answers from filings as usual and never restates or computes position numbers.
- Every price carries its date; prices older than 3 days are flagged ⚠ stale.
- **Not in the screener** (ETFs, mutual funds, bond funds, non-US listings): add them to `data/portfolio/manual_prices.csv` (`ticker,price,date`) — never sent anywhere; a manual price wins when its date is the same or newer. Holdings without any price are listed as not included.
- The screener is Nasdaq's public website endpoint, not an official API; if it changes, `--update` fails loudly (it refuses to save a suspiciously small download) and manual prices keep working.
- Both price files are git-ignored. (Stooq, the first choice, now sits behind a JavaScript bot challenge; it is not used.)

### Day 15 — Corpus expansion: 25 companies
Added 20 large caps chosen for sector coverage, each with its latest 10-K and two latest 10-Qs (60 filings; only the new companies were fetched, so the original five companies' filings — and the evaluation's ground truth — didn't change):

| Sector | Companies |
|--------|-----------|
| Tech | AAPL, MSFT, NVDA, GOOGL, AMZN, META, AVGO, TSLA |
| Healthcare | LLY, JNJ, UNH |
| Financials | JPM, V, BAC, GS |
| Consumer | WMT, COST, KO, PG |
| Energy | XOM, CVX |
| Industrials | CAT, BA |
| Media / Utilities | NFLX, NEE |

- The new filings passed the extraction checks (0 duplicated lines, no repeated page furniture). The store grew 2.4× to 16,946 passages; exact search still takes ~6 ms (26 MB of vectors). A full CPU rebuild now takes ~22 minutes.
- The router recognises the new companies by ticker and name. Names that are also ordinary words only count when capitalised ("Visa" vs "a travel visa", "Meta" vs "meta-analysis"), and 1–2-letter tickers must stand alone ("V-shaped recovery" and "PG-13" don't match). Known limit: a capitalised "Amazon" always means the company.
- 10 new golden questions (53 total): exact figures checked against the raw filings — including Boeing's *Commercial Airplanes* backlog (a segment row; the phrase "total backlog" in the filing refers to that segment, not the company) — plus retrieval questions and one unanswerable (Netflix Q4 2026; the corpus ends at Q2).
- **Retrieval:** 53/53 route, 53/53 hit@k, MRR 0.94; sanity check 15/15. One question (q17, market makers' inventory risk) briefly lost its labelled papers to Goldman Sachs and Bank of America 10-Ks — which, on reading, describe their own market-making positions and risk limits, so they were added to its label.
- **qwen2.5:7b on the 10 new questions:** citation accuracy 98%, no invented numbers, 6/6 exact figures, the unanswerable one refused (it named the latest quarter available instead).

### Day 16 — Tier 1 gaps and filing refresh
**Nine explainers** for topics the evaluation found uncovered: credit spread, liquidity, arbitrage, Roth IRA, IRA, 401(k), market order, limit order, bid-ask spread. "What is a bid-ask spread?" now routes to concept (the phrase only counts as technical next to a modelling word, like "spread" or "duration"). Six concept questions were added to the golden set, the credit-spread question gained its Tier 1 article, and the unanswerable concept question is now "What is a 529 plan?" (no document mentions one).

**Filing refresh** — companies keep filing; without this the corpus silently goes stale:

```bash
python scripts/refresh_filings.py                    # dry run: which filings are new?
python scripts/refresh_filings.py --apply --rebuild  # download them, then chunk → enrich → embed
```

It asks the SEC for each company's current latest 10-K + two 10-Qs and downloads only what isn't on disk, with the same extractor as the scraper. **Older filings are kept** — the golden set cites specific filings, so answers stay reproducible; each passage's label carries its filing date. Its first run found a JPM 10-Q filed 2026-08-06 that the original download (Aug 5) had missed.

Retrieval after both: **59/59 route, 59/59 hit@k, MRR 0.94**. The raw Day 7 sanity check (which bypasses the router) now flags "How does a limit order book work?": the new limit-order articles rank 2nd, 4th and 5th among order-book papers. Through the router this question is technical and searches only filings and papers (q15 still hits), so the flag is expected.

### Full evaluation on the expanded corpus (Day 16)
All 59 golden questions with qwen2.5:7b after the corpus expansion (25 companies, 47 explainers):

| | Concept (22) | Technical (26) | Portfolio (11) | **All (59)** |
|---|---|---|---|---|
| Route / hit@k | 100% / 100% | 100% / 100% | 100% / 100% | **100% / 100%** (MRR 0.94) |
| Citation accuracy | 94% | 91% | 88% | **91%** |
| Required figures | — | 6/6 | 6/6 | **12/12** |
| Unanswerable refused | 2/2 | 3/3 | 1/1 | **6/6** (0 answerable refused) |
| Numbers in no source / answer | 0.52 | 0.17 | 0.00 | 0.28 |

Most "numbers in no source" are **invented worked examples in concept answers** — e.g. a "$50,000 salary" for the 401(k) and "$50 vs $51" for arbitrage, neither in the article — despite the template's "never make up an example". The number check catches them and the UI warns.

**Fix (Day 16):** the concept rule now names the failure concretely ("do NOT write illustrations like 'For example, if you earn $50,000…' with your own numbers") and the concept prompt ends with a one-line reminder right before the answer — small models weigh the end of the prompt most. Re-run on the 22 concept questions (`data/eval/eval_answers_qwen2.5-7b_concept_prompt_v2.json`):

| Concept answers | Before | After |
|---|---|---|
| Answers with numbers in no source | 6/22 (11 numbers) | **1/22** (1 number) |
| Numbers in no source / answer | 0.52 | **0.05** |
| Citation accuracy | 94% | **98%** |
| Unanswerable refused / answerable refused | 100% / 0% | 100% / 0% |
| Median answer length | 462 chars | 454 chars |

The one left (a made-up "0.04%" expense ratio) is flagged in the UI. Regenerating flagged answers was considered and not added: it would cost another model call for a small remaining gain. (Run note: an Ollama auto-update restarted the Mac mid-run; the checkpoint kept 49 answers and the last 10 ran on Ollama 0.40.6 instead of 0.33.2.)

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

### Day 17 — Live portfolio in ₹ (Indian stocks) and an Ollama alternative
**Portfolio panel.** The main screen now opens with a **📊 My portfolio** panel above the chat:
- **Add a stock:** symbol (`RELIANCE`, `TCS`, `HDFCBANK`…), exchange (NSE by default, BSE or US), quantity, average buy price and an optional purchase date. Buying more of a stock you hold merges the lots at the weighted average price. **Edit / remove** opens a table editor for everything else. Every save is validated (`portfolio_store.validate_holdings`: bad symbols, blank or negative numbers, duplicates, future dates) and written atomically, so a crash can't leave a half-written file.
- **Live prices:** with *Live prices* on, the panel refreshes from Yahoo Finance every 60 s without reloading the chat: Invested, Current value, Day's P&L, Total P&L, and per stock LTP, Day %, Current, P&L. Refresh pauses while an answer is generating, so it can't interrupt one.
- **Rupees:** NSE/BSE holdings are stored with Yahoo's suffixes (`RELIANCE.NS`, `TCS.BO`) and priced in ₹; totals use Indian digit grouping (₹12,34,567.80). US holdings stay in $ per share and are converted to ₹ at the live USD→INR rate (`USDINR=X`, a currency pair, so it reveals nothing about you). Without a rate they are left out of the ₹ totals rather than mixed in as dollars.
- The chat uses the same prices. "How is my portfolio doing?" gets a ₹ valuation computed by code; "my RELIANCE shares" (or "Wipro" for `WIPRO.NS`) routes to your position. Indian companies have no SEC filings in the corpus, so those questions use the explainers plus your position instead of an empty filing search.
- **Privacy:** Yahoo sees the symbols requested — never quantities or costs. The Nasdaq whole-market download (Day 14, no tickers sent, US stocks only) is still available as `--update-market`. Same-date precedence: manual price file > Yahoo > Nasdaq.

```bash
python scripts/prices.py --update          # Yahoo prices for your holdings (+ USD→INR if you hold US stocks)
python scripts/prices.py                   # ₹ valuation
```

**MLX: the alternative to Ollama.** `scripts/llm_backends.py` adds Apple's MLX next to Ollama:
- Models named `mlx:…` run on a local MLX server (`127.0.0.1:8080`, started by `scripts/start_mlx_server.sh`, offline mode, no telemetry); everything else runs on Ollama.
- If Ollama isn't running or comes up with no models (the restart-before-the-drive-mounts problem), FinRAG uses the MLX model instead of failing; the UI's model picker lists both.
- Same model (Qwen2.5-7B-Instruct, 4-bit), measured on the same six questions on the 8 GB M1: **decode is the same** (9–12 tokens/s on both), **prefill is slower on MLX** (first token 24–52 s on 3,000+-token technical/portfolio prompts vs ~30 s on Ollama), and the answers were equally correct and cited. So MLX is a working backup, not a speed-up; on this machine the bottleneck is memory, not the runtime.

**Leaner prompts.** Technical and portfolio retrieval went from k=6/5 to k=4 (hit@k still 100%, MRR 0.94, precision up), and technical/portfolio answers aim for about 200 words, which cuts both prefill and decode time.

### Day 18 — Indian annual reports and a research-terminal UI
**Annual reports.** `scripts/scrape_tier2_india.py` adds the latest (FY2025-26) annual report of 15 NSE large caps across sectors — RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, BHARTIARTL, ITC, HINDUNILVR, LT, BAJFINANCE, MARUTI, SUNPHARMA, TATASTEEL, ASIANPAINT (`config/corpus_manifest.yaml`, `tier2_india`):
- Reports come from NSE's public annual-report listing, requested with an honest User-Agent (BSE's API refuses scripted requests, and it isn't worked around). PDFs are cached in `data/pdf_cache/india/` (git-ignored, ~290 MB) and extracted with `pypdf`.
- **Sections from running headers.** Each page is labelled from its running header (or, for reports like Asian Paints', its "to the Consolidated Financial Statements … (Contd.)" footer). Kept: the integrated report, Board's report, MD&A, corporate governance and the **consolidated** statements. Skipped: the AGM notice, the BRSR / sustainability report and the **standalone** statements — parent-only figures a model could quote as the company's results. A skipped section only continues while its own header does, so a notice printed before the integrated report can't swallow it.
- **Rupee repair.** Report fonts map ₹ to `I`, `H`, `C`, `K` or a backtick; it is restored only right before an amount in crore/lakh and in unit headers ("(₹ in crore)").
- Same schema as the SEC filings (tier 2, `company_ticker` = `RELIANCE.NS`, title "Reliance Industries (RELIANCE) Annual Report FY2025-26", sections labelled with page numbers), so chunking, enrichment, retrieval and citation checking are unchanged. 16.8M characters after filtering.
- **Routing.** The router knows the 15 companies by symbol ("TCS", "INFY") and by name ("Infosys", "HDFC Bank", "L&T"). "Reliance" counts only when capitalised (lowercase "reliance" is an ordinary word). Known limitation: an uppercase "ITC" always means the company, even as "input tax credit". A BSE holding (`TCS.BO`) uses its NSE company's report.

**UI.** The app is now a two-pane research terminal instead of one long column:
- **Top bar:** NSE open/closed, the local model in use, corpus size, and a ⚙ Settings menu (model, route, citation check). The sidebar is gone.
- **Left — Portfolio:** KPI cards (Invested, Current value, Day's P&L, Total P&L), then tabs: **Holdings** (P&L coloured), **Allocation** (donut by value + return-on-cost bars) and **Stock view** (1-year price chart with your average buy price drawn on it; 52-week range, 1/3/12-month returns, volatility, max drawdown; one-click questions — key risks, revenue & margins, outlook — sent to the assistant). ＋ Add and Edit open a popover and a table editor.
- **Right — Research assistant:** a scrolling chat with suggested questions and the input underneath.
- Dark theme in `.streamlit/config.toml`, system fonts only (no web-font request). The stock statistics describe past prices; nothing is forecast.
- **PortIQ brand:** logo mark, "Port**IQ**" wordmark with the blue→violet gradient, tagline "Upload. Ask. Get Insights.", indigo accent. `assets/portiq_mark.svg` is an SVG *recreation* of the logo; drop the original in as `assets/portiq_logo.svg` or `.png` and the app uses it instead (embedded as a data URI — no network request). KPI cards show whole rupees (paise stay in the table).

**Getting annual-report numbers right.** Retrieval found the right report every time, but answers quoted the wrong figures. Diagnosed and fixed step by step, measuring each change:
- **Subsidiaries' statements.** Reliance prints Jio Platforms' and Reliance Retail's summary statements inside its consolidated section, and the model quoted Jio's revenue (₹1,46,885 crore) as Reliance's. Those pages are now labelled "Subsidiary: Jio Platforms Limited", and for a question that doesn't name the subsidiary they are left out of the context (asking about "Jio" or "Reliance Retail" still gets them — both route to Reliance's report).
- **Hybrid search for annual reports.** The small embedding model ranked the passage holding the asked-for figure 7th–17th within a company's 450–1,200 passages. Fusing a keyword (BM25) ranking with the dense one (reciprocal-rank fusion) lifted it into the top 4 for 4 of 5 test figures. A cross-encoder reranker was tried and made it worse (3 of 5), so it isn't used. On SEC filings hybrid search *cost* MRR (0.94 → 0.90: quarterly 10-Q tables repeat "revenues" and outranked the 10-K), so it applies to annual reports only — SEC results are identical to before.
- **Prompt rules** (only when an annual-report passage is in the context): amounts in ₹ crore with Indian grouping, prefer CONSOLIDATED figures and say so, current year is the first column, a subsidiary's statement isn't the company's total. Annual-report passages also show their section to the model ("— Board's Report", "— Consolidated Financial Statements").
- **Extraction fixes:** Board's reports that open "The Board of Directors hereby submits…" (Infosys), the "(In ` crore" unit header, and AGM-notice items ("Item No. 6 –") printed after the statements.
- **Remaining limit:** in tables with Standalone and Consolidated columns side by side, the 7B model still takes the first (standalone) column — Infosys ₹1,48,819 crore instead of ₹1,78,650 crore — and it rounds ₹31,05,250.47 crore to ₹31,05,251. This is model capability, which is what the model comparison below is for.
- **Golden label corrected:** Reliance's "net profit" is ₹95,754 crore for the year *or* ₹80,775 crore attributable to owners (the other ₹14,979 crore is non-controlling interest); both are accepted.

**Results.** Retrieval on all 66 golden questions: route accuracy 100%, hit@k 100%, MRR 0.95, precision 86% — all six answerable Indian questions rank their company's report first. Answers (qwen2.5-7b on MLX, 36-question comparison set — every required-figure and should-refuse question plus a concept/portfolio/technical sample): 10/16 required figures as scored (11/16 with the corrected Reliance label — re-score pending), 85% of citations correct, 0.03 invented numbers per answer, 7/7 out-of-corpus questions refused, no answerable question refused, 39 s per answer.

**Model comparison: Qwen2.5-7B vs Qwen3-8B.** Ollama is being replaced, and the model has to serve a hosted website, so two Hugging Face open models (both Apache 2.0) were compared on the same 36 questions (every required-figure and should-refuse question plus a concept/portfolio/technical sample), same retrieval, both 4-bit on MLX:

| | Qwen2.5-7B-Instruct | Qwen3-8B (thinking off) |
|---|---|---|
| Required figures correct | 11/16 (US 9/12, India 2/4) | **12/16** (US 10/12, India 2/4) |
| Refused out-of-corpus questions | **7/7** | 5/7 — two confidently wrong numbers |
| Refused answerable questions | **0/29** | 1/29 |
| Citations correct | 51/60 (85%) | **65/74 (88%)** |
| Invented numbers | 1 | 1 |
| Seconds per answer (8 GB M1) | **39** | 50 (needs memory workarounds) |

Qwen3 got NVIDIA's and NextEra's figures right where Qwen2.5 didn't, but it answered "Nvidia's revenue last year" with a wrong figure and gave ExxonMobil's **Q1** revenue as "Q3 2026" (not filed yet), where Qwen2.5 said the sources don't cover it. It also mislabelled Costco's total revenue as net sales. For a finance product a confident wrong number is the worst failure, so **Qwen3-8B is not a clear upgrade**; Qwen2.5-7B stays the default. Neither fixes the Indian Standalone|Consolidated table problem (both 2/4). A 24 GB server GPU can run a 14B model (Qwen2.5-14B-Instruct / Qwen3-14B in 8-bit), which can't be tested on this Mac — evaluate it there with the same harness (`evaluate.py --answers --model api:<model> --only <ids>`) before choosing the production model. Results: `data/eval/eval_answers_mlx-qwen2.5-7b_rescored.json`, `data/eval/eval_answers_mlx-qwen3-8b_subset.json`.

**Stock search when adding.** "＋ Add" now has a type-ahead dropdown: typing "relian" lists *RELIANCE — Reliance Industries Limited · NSE*, Reliance Power, Reliance Infrastructure… (matching symbol or company name). The list (`data/reference/stocks.csv`, 13,396 rows, committed so it works offline) holds every NSE equity and ETF and US companies from the SEC's list (largest first), each with the Yahoo ticker the price feed uses — so a picked stock always gets prices. Anything not listed can still be typed (NSE by default; `TCS.BO` for BSE). Refresh with `python scripts/stock_list.py --update` — whole lists are downloaded, nothing about your holdings is sent.

**Faster answers (measured).** On the 8 GB Mac, search takes 0.1–0.3 s and the citation check ~1 s; the time is the model *reading* the ~2,500-token prompt (~120–150 tokens/s, 20–25 s) and *writing* (~12 words/s). Changes, each measured on the same 36 questions:
- Answers are shorter and lead with the answer (~120 words), but say "not covered" first when the sources don't answer — and never swap in a nearby period or figure. **39 s → 28 s per answer**, figures correct 10/17 → 11/17, out-of-corpus questions still all refused (6/6), no answerable question refused (was 1/30), citations 85% → 83% (within run-to-run noise). The first wording ("lead with the answer") made the model answer a question it should have refused, so it was revised before keeping it.
- The MLX server keeps the last prompt's cache, so the ~600 tokens of fixed instructions aren't re-read (a few seconds per question).
- Tried and **rejected**: trimming long passages to the lines matching the question. It cut prompts by up to 55% but dropped the very row with the answer when the question said "revenues" and the table said "Revenue" (Reliance, Infosys) — a missing figure is worse than a slower answer.
- In the UI: the model is warmed up when the app opens (an idle model took ~20 s extra to start), repeated questions are answered instantly, and progress names the documents being read. On this Mac heavy swapping (other apps open) can still make an answer take a minute; on a GPU server the same pipeline answers in seconds.
- Relative periods ("last year", "latest quarter") get a note to use the most recent period in the sources and name it — Nvidia "last year" was answered as fiscal 2025 instead of 2026. For an annual question the model could quote a year-to-date figure from a 10-Q (Nvidia $177,837 million instead of the 10-K's $215,938 million) — fixed next (below). Golden q39 relabelled: NVIDIA has been in the corpus since Day 15, so "last year" is answerable (fiscal 2026).

**Annual questions read the 10-K.** The router flags a question about a whole year ("fiscal year 2026", "in 2025", "last year", "annual", "10-K") — but not one about a quarter, half-year or year-to-date — and retrieval then searches only that company's 10-K filings (all filings if it has none; Indian annual reports are annual already). Measured:
- Retrieval on all 66 questions: MRR 0.95 → 0.96, precision 86% → **92%**; every annual question now gets 10-K passages only (Caterpillar's 10-K moved from 3rd to 1st). No question got worse.
- Answers on the 36-question set: **figures correct 11/17 → 14/17** (Nvidia ×2, NextEra, Caterpillar fixed), out-of-corpus 6/6 refused, no wrongful refusals, citations 84%, 25 s per answer. One new slip: Tesla's $94,827 million was converted to "$94.827 billion" — flagged by the number check ("not found in the sources"), so the answer card shows ⚠. Still open: Infosys (standalone column) and HDFC Bank (rounded ₹31,05,250.47 crore).

**Standalone | Consolidated tables labelled.** PDF text flattens these side-by-side tables into rows of bare numbers ("Revenue from operations 1,48,819 1,36,592 9.0 1,78,650 1,62,990 9.6"), and the model quoted the first, standalone column as Infosys' revenue. The extractor now finds each table's header ("Standalone Consolidated", either order; prose like "the Standalone and Consolidated Financial Statements" is ignored), splits every row with an even number of values into the two blocks (the standalone block's columns always precede the consolidated block's), and names each value when the table has a period line: *"Revenue from operations — Standalone: 2026 1,48,819, 2025 1,36,592, YoY % 9.0; Consolidated: 2026 1,78,650, 2025 1,62,990, YoY % 9.6"*. Odd rows are left as they are, not guessed. Inside these rows a number broken by the PDF ("29, 211") is rejoined — which is also why the number check had called Infosys' standalone profit "not in the sources". 231 rows across 8 reports are labelled.
- `build_vectorstore.py --only-source nse_annual_report` replaces just the annual reports' passages (10,748) instead of re-embedding all 28,000 — after checking that all 19,512 other passages came out byte-identical from re-chunking.
- Result: Infosys answered with the consolidated figures (₹1,78,650 crore, ₹29,440 crore); Indian figures 3/4; on the 36-question set **figures correct 15/17**, citations 86%, out-of-corpus 6/6 refused, no wrongful refusals, retrieval unchanged (MRR 0.96, precision 92%). Left: Tesla's "$94.827 billion" conversion (flagged on the answer) and HDFC Bank's rounding.

**A UI a customer can trust.**
- Every answer shows what was verified — "✓ 3 of 3 statements match their sources", "✓ All figures found in the sources" (or ⚠ with the details) — plus the documents used and the time taken; a "not covered" answer is marked as such ("it didn't guess").
- Sources are cards (📘 annual report, 🧾 SEC filing, 📖 explainer, 📄 paper — with section, page and link), and the cited passages are one click away. Routing jargon moved behind Settings.
- "ⓘ How it works": where answers come from (document counts read from the corpus), how they're checked and what the check can't catch, what is computed by code, privacy, and that PortIQ is not a SEBI-registered adviser. A disclaimer sits under the question box.
- **Insights** tab (first): plain-English observations, the 1-year return of today's holdings vs Nifty 50 and S&P 500, volatility, worst drop, largest holding, and a growth-of-₹100 chart — all computed by code (`scripts/insights.py`, tested), describing the past, never advising.

**Hosting groundwork.** The backend now talks to any OpenAI-compatible server by URL (vLLM, Hugging Face Inference Endpoints — see Setup), tested end to end against a live OpenAI-compatible server. A server that isn't on the same machine is refused unless `PORTIQ_ALLOW_REMOTE_LLM=1`, and the UI then shows "☁ <host>" with a plain statement of what is sent there. Before a public launch the app still needs to become multi-user (accounts + per-user holdings; today there is one `holdings.csv` and the app refuses to bind beyond 127.0.0.1).

### Day 20 — Portfolio sources: CSV import and read-only broker sync
Holdings can now come from a broker export (CSV) or a broker's read-only API used with **your own credentials**, through one adapter interface (`scripts/brokers/`). See **Portfolio sources** under Setup. Design notes:
- `BrokerAdapter` (`fetch_holdings`, `fetch_prices`, `auth_status`) with typed errors — `AuthExpired` / `CredentialsMissing` (re-authenticate) vs `RateLimited` / `BrokerUnavailable` (try later) vs `UnsupportedOperation` vs `MalformedResponse` — and a registry (`get_adapter`, `list_adapters`).
- Every broker endpoint and field name was taken from the brokers' official docs (Upstox, Dhan, Groww — Oct 2026); what the docs don't state is a TODO in each adapter's header.
- `scripts/portfolio_sync.py` writes an atomic SQLite snapshot (`data/portfolio/portfolio.db`); the router, prompts, UI and valuation read it through the same `load_holdings()` as before, and every synced position says where and when it came from ("— from Upstox, as of 2026-10-23T14:05:11").
- In-app **⇅ Import** dialog (upload + column guesses + preview, per-broker Check / Sync now, synced sources with Remove) driving the same CLI in a subprocess; several named CSV imports can coexist (`csv:<name>`).
- Found and fixed on the way: loading the embedding model at question time contacted huggingface.co (a sentence-transformers update check). It now loads offline from the cache; `tests/test_security.py` proves answering makes no outbound connection.

### Day 21 — CAS statement import (every broker, no approval needed)
Most brokers can't be connected from another app (Groww documents no way at all), but every investor's shares sit at a depository — NSDL or CDSL — which e-mails a **Consolidated Account Statement (CAS)** each month covering every demat account under their PAN. **⇅ Import → CAS statement** reads that PDF (see **Portfolio sources** under Setup). Design notes:
- Parsed on this machine by [casparser](https://github.com/codereverser/casparser) (MIT); its only network code is an explicit "update the ISIN database" command, never called. `tests/test_cas.py` runs parsing under the network guard.
- A CAS identifies shares by ISIN. The NSE symbol comes from NSE's own equity and ETF lists, whose ISINs are now stored in `data/reference/stocks.csv` (2,613 equities + 355 ETFs; refresh with `stock_list.py --update`), falling back to casparser's bundled ISIN database. NSE's list wins where they differ: casparser's still has pre-rename symbols (ICICIPRULI for ICICILIFE, GUJGASLTD for GUJENERGY — 9 in all). ETFs count whether a statement lists them under equities or mutual funds. BSE-only shares, other mutual funds, bonds and NPS are listed as skipped with the reason — nothing is dropped silently.
- **A CAS has no purchase price.** `avg_cost` can now be unknown (NULL) end to end: such holdings are valued, but Invested, P&L and answers leave them out and say so — never counted as zero. The import screen lets you type average prices, and pre-fills them from your previous import of the same statement.
- Privacy: the password (your PAN) goes to the import process through stdin, never on a command line, and is never logged or stored; the PDF exists only in a private temp folder while it is read; only holdings are kept — not your name, PAN, address, contact details or account numbers (tested).
- Found and fixed on the way: empty cells in the holdings table showed Streamlit's "None" instead of "—" (Streamlit ignores formatters for empty numbers), now plain text.
- Found and fixed on the way: huggingface_hub 1.x fetches an "agent registry" from huggingface.co once a day while loading even a cached model (`local_files_only` doesn't stop it), so the app contacted huggingface.co daily. The embedder now loads in the hub's offline mode; `tests/test_security.py` caught it when the day-old cache expired.

### Day 22 — Website groundwork (hosted mode)
Decided: host on Hugging Face Spaces, Google sign-in, a Hugging Face Inference Endpoint for the model, and users' holdings stored on the site's server **only after a clear notice**. This step makes the app safe for many users; deploying it is next. See **Hosting** under Setup.
- `scripts/workspace.py`: each script run says whose data it handles (a context variable — Streamlit runs every session's script in its own thread, so a global would leak between users). Holdings file, snapshot DB, uploaded CSVs and saved column mappings all resolve to the signed-in user's own folder, named by a one-way hash of their sign-in ID. With nobody signed in, reading holdings **fails** instead of falling back to a shared file — the fail-safe that caught dialogs and the live-price refresh rerunning without the sign-in gate (now fixed: they re-apply the session's user).
- Shared answer cache keyed per user (an answer can carry the asker's positions).
- Broker connections are off on the website: `.env` tokens there would be the operator's, so every visitor's "sync" would import the operator's accounts. The sync refuses broker sources and `get_secret` returns nothing in hosted mode.
- Privacy notice before anything can be added (asked again when it changes); ⚙ Settings → Download my data / Delete my data (type DELETE) / Sign out.
- Local mode is unchanged (tested: same paths, Brokers tab, "Runs on this machine").

## Tests

Plain scripts (no pytest needed):

```bash
python tests/test_citations.py   # citation labels, dedup, checker, number check
python tests/test_router.py      # 33 labelled queries (accuracy table), mixed queries, plans, flags
python tests/test_prompts.py     # grounding/style rules per route, [N] label order, no-context answer, cloud-model refusal
python tests/test_evaluate.py    # refusal detection, label matching, golden-set consistency
python tests/test_exact_index.py # exact search ranking, distances, filters
python tests/test_prices.py      # no tickers in the price request, parsing, manual overrides, valuation math
python tests/test_refresh.py     # only filings missing from disk are fetched; nothing is deleted
python tests/test_portfolio_store.py  # holdings validation, atomic save, add/merge lots
python tests/test_llm_backends.py     # Ollama/MLX/API switch and fallback, remote opt-in, streaming, think-block removal (no servers needed)
python tests/test_india_reports.py    # annual-report sections kept/skipped, rupee repair, NSE listing
python tests/test_stock_list.py       # stock-list parsing, Yahoo tickers, typed symbols, refusing tiny downloads
python tests/test_insights.py         # allocation, concentration, holding periods, past-year performance, wording
python tests/test_build_update.py     # --only-source replaces one source's passages and leaves the rest
python tests/test_brokers.py          # adapter contract: parsing, expired token, backoff, malformed payloads, CSV validation
python tests/test_portfolio_sync.py   # atomic snapshots, failed sync keeps the last good one, NULL prices, source + as-of, CLI --json/--label/--remove, column guesses
python tests/test_security.py         # no secrets anywhere, broker-host-only sync, no network when answering, .gitignore
python tests/test_cas.py              # CAS → holdings, no personal data stored, wrong password / not a CAS, stdin password, DB migration, unknown costs
python tests/test_workspace.py        # hosted mode: users isolated (incl. answer context, threads), nobody signed in reads nothing, consent, delete, no brokers
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
- ~~Silently corrupted embeddings~~ — the embedder ran on the GPU, which ran out of memory with qwen2.5:7b loaded and returned garbage without an error; embeddings now always run on the CPU.
- ~~Approximate-index misses~~ — queries now use exact search (see Day 6).
- ~~Mislabelled table numbers~~ (with qwen2.5:7b) — llama3.2 reported Q1 2025 *expenses* and 2024 *annual* revenue as "last quarter's revenue"; qwen2.5:7b labelled periods correctly on every table question in the eval and keeps units ("$139,996 million (fiscal 2026)").
- ~~Repo duplicates and stale stores~~ — root-level copies of `scrape_tier1.py`, `validate_manifest.py` and `corpus_manifest.yaml` (only old, already-fixed content), the unused `data/chroma_db/` (the pre-fix 6,133-chunk build) and `data/vectorstore/`, and 10 orphaned index folders (~118 MB) in `~/.finrag_chroma_db` were removed (moved to the Trash). Ollama's models now live only on the external drive.
- ~~No price data~~ — local whole-market prices (Day 14); performance questions get a code-computed valuation.
- ~~Cloud models could leak prompts~~ — Ollama cloud models are refused by name and by server metadata.
- ~~Near-duplicate SEC chunks~~ — the EDGAR extractor emitted both `<div>` and its inner `<span>`, doubling 23% of lines (up to 51% in some filings), split tables into one line per cell, and kept page headers/footers and the hidden XBRL header (a 58k-token chunk). Fixed at extraction; SEC text dropped from 136k to 15k lines with no content words lost. Cross-filing restatements (10-Q repeating 10-K text — real content, kept in the corpus) are suppressed at retrieval time instead.

**Open:**
1. **Occasional invented examples in concept answers** — down from 6/22 to 1/22 answers after the Day 16 prompt fix; the remaining case is flagged by the number check. A regenerate-on-flag guard would close it at the cost of an extra model call.
2. **Mislabelled numbers aren't *detected*** — the number check only verifies a number exists in the sources, not that it's attached to the right row or period. (qwen2.5:7b labelled every table figure correctly in the evaluation; this is a safety-net gap, not an observed failure. An earlier entry here claimed qwen converted Tesla's revenue to "$94.83 billion" against the verbatim rule — wrong: the 10-K itself says "$94.83 billion".)
3. **Slow answers on 8 GB** — 15–60 s per qwen2.5:7b answer on an 8 GB M1 after the Day 17 trims (memory-bound; MLX is no faster); llama3.2 is twice as fast but much less reliable.
4. **Mis-citations are reported, not corrected** — rarer with qwen2.5:7b (0.11 per answer), but the answer text keeps the wrong number.
5. **Small Tier 1 chunks** — Investopedia sections are short and the chunker doesn't merge across headings (median 124 tokens).
6. **Tier 1 text includes trailing references/category tags** from each article; one Tier 3 chunk is a PDF artifact (a run of 13k `●` characters).
7. **Indian coverage is 15 companies** — other Indian holdings are answered from the explainers plus your position only. Annual reports are yearly, so there are no quarterly results yet (NSE's quarterly filings are the next step).

---

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b      # default model (~4.7 GB)
ollama pull llama3.2        # optional: faster, less reliable
python scripts/prices.py --update   # optional: prices for "how is my portfolio doing?"
```

**Optional: MLX (Apple Silicon) as an alternative to Ollama** — one-time ~4 GB download to the external drive, then start the server whenever you want it:

```bash
pip install mlx-lm
HF_HOME="/Volumes/Crucial X9/hf-cache" python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-7B-Instruct-4bit')"
scripts/start_mlx_server.sh              # qwen2.5-7b (default)
scripts/start_mlx_server.sh qwen3-8b     # Qwen3-8B, thinking off (download mlx-community/Qwen3-8B-4bit the same way)
```

One model per server (two 4–5 GB models can't share 8 GB); the served model is written to `data/mlx_serving.txt`, the UI offers exactly that one, and asking for another fails with a clear "restart with …" message instead of loading a second model. The server runs one request at a time with no prompt cache and 512-token prefill steps; for Qwen3 it also stores the attention cache in 8 bits and disables MLX's buffer cache — the defaults ran out of Metal memory on this 8 GB Mac.

The script sets `MLX_MPI_LIBNAME` so MLX doesn't load Anaconda's MPICH library (it expects Open MPI and aborts). On the exFAT drive macOS leaves `._*` files in the model cache, which crash the Hugging Face cache scan behind `/v1/models`; FinRAG checks `/health` instead, and `find "/Volumes/Crucial X9/hf-cache" -name "._*" -delete` cleans them.

**Optional: any OpenAI-compatible model server (for hosting)** — vLLM, a Hugging Face Inference Endpoint, or anything else that speaks the OpenAI chat API. Configure it with environment variables; the model then appears as `api:<model>`:

```bash
# On an NVIDIA GPU server (24 GB, e.g. L4 / A10G): serve Qwen3-8B with vLLM
pip install vllm
vllm serve Qwen/Qwen3-8B --max-model-len 8192
# Point PortIQ at it (same machine: no opt-in needed)
export PORTIQ_LLM_URL=http://127.0.0.1:8000/v1
export PORTIQ_LLM_MODEL=Qwen/Qwen3-8B
```

For a server that is **not** on the same machine (e.g. `https://<name>.endpoints.huggingface.cloud/v1` with `PORTIQ_LLM_API_KEY=<HF token>`), PortIQ refuses to use it until `PORTIQ_ALLOW_REMOTE_LLM=1` is set too — questions, retrieved passages and the details of holdings you ask about are sent there, and the UI then says so (top bar shows "☁ <host>"). Qwen3's thinking mode is switched off per request (`chat_template_kwargs`), a 503 from a scaled-to-zero endpoint is reported as "starting up", and the API server is offered without probing it (a sleeping endpoint would otherwise look down). Fallback order when the requested model isn't available: Ollama → MLX → API.

### Portfolio sources

| Source | Status | Holdings | Prices | Auth |
|---|---|---|---|---|
| **CAS statement** (NSDL / CDSL) | ✅ working; tested with casparser's data model and an encrypted test PDF — not yet with a real statement | every demat account under your PAN, all brokers, in one PDF | statement-date prices as a fallback (live prices from Yahoo) | the PDF password (your PAN) |
| **CSV / broker export** | ✅ working | any export, via a column-mapping JSON | — (Yahoo / manual prices) | none |
| **Upstox** | ✅ implemented, fixture-tested; not yet run against a live account | `GET /v2/portfolio/long-term-holdings` | `GET /v3/market-quote/ltp` | your access token (expires 3:30 AM daily) |
| **Dhan** | ✅ implemented, fixture-tested; not yet run against a live account | `GET /v2/holdings` | `POST /v2/marketfeed/ltp` (needs client ID) | your access token (24 h) |
| **Groww** | ✅ implemented, fixture-tested; not yet run against a live account | `GET /v1/holdings/user` | `GET /v1/live-data/ltp` | your access token (expires 6:00 AM daily) |
| **Hosted "Connect broker"** | ⏳ not built — needs broker partner approval | see `docs/hosted-oauth-design.md` | | |

"Fixture-tested" means against the official docs' example responses with a mocked HTTP layer — no broker sandbox exposes holdings, and no live account was available. Details the docs don't state are TODOs in each adapter (e.g. the HTTP status Dhan and Groww send with an expired token; whether Upstox's `quantity` includes `t1_quantity`; Groww holdings have no exchange field, so they're priced on NSE).

**In the app: ⇅ Import** (portfolio header) opens one dialog with four tabs:
- **CAS statement** — upload the monthly e-CAS PDF from NSDL or CDSL and type its password (the first holder's PAN in capital letters). You see every holding found, merged across demat accounts, with what was skipped and why. A CAS doesn't include what you paid, so you can type your average buy price per holding (pre-filled from your last import); holdings left blank are valued but kept out of Invested and P&L. Lost the e-mail? NSDL's and CDSL's websites let you download past statements. Several people's statements can be kept apart with "Whose statement?".
- **Upload CSV** — pick your broker's holdings export and name the import. PortIQ guesses which column is the symbol, quantity and average price from common header names (you confirm or change each), previews exactly what will be imported, and lists every row it will skip with the reason. Importing again under the same name replaces that import; other imports and your own holdings file are untouched. The column choices are saved (`config/broker_mappings/<name>.local.json`) so the next export of the same layout imports in one click; the uploaded file is kept in `data/portfolio/imports/` (last 3 per import, git-ignored).
- **Brokers** — per broker: whether a token is in `.env` (checked locally, without a network call), the current snapshot and last sync result, **Check** (is the token still accepted?) and **Sync now**. An expired token says so and keeps showing the last snapshot. With no token, "How to connect" shows the steps below.
- **Synced sources** — every import and broker snapshot with its holdings count and as-of time, and **Remove** (deletes that source's snapshots and PortIQ's stored copy of the upload; your original file and broker account are untouched).

Imports and syncs run `scripts/portfolio_sync.py` in a separate process, so the sync's broker-host-only network limit never applies to the app's live prices.

**Bring your own credentials.** Copy `.env.example` to `.env` (git-ignored) and fill in only the brokers you use. Tokens are read from `.env` only — never from the command line, never logged, never stored in the database.
- **Upstox:** create an app on the Upstox developer portal and generate an access token (OAuth flow, or from the app page). It expires at 3:30 AM the next day → `UPSTOX_ACCESS_TOKEN`. The 1-year read-only Analytics Token can also read Portfolio APIs, but only from a whitelisted static IP.
- **Dhan:** web.dhan.co → "Access DhanHQ APIs" → generate a token (valid 24 h) → `DHAN_ACCESS_TOKEN`; your client ID → `DHAN_CLIENT_ID` (needed for prices).
- **Groww:** groww.in → Profile → Trading APIs → generate an access token (expires 6:00 AM daily) → `GROWW_ACCESS_TOKEN`.

```bash
python scripts/portfolio_sync.py --source csv --file data/portfolio/imports/export.csv --mapping config/broker_mappings/my_broker.local.json
python scripts/portfolio_sync.py --source upstox --dry-run   # fetch and show, write nothing
python scripts/portfolio_sync.py --source upstox
python scripts/portfolio_sync.py --status                     # auth state + snapshot age per source
python scripts/portfolio_sync.py --source csv --file export.csv --mapping m.local.json --label zerodha   # a named import, kept beside others
python scripts/portfolio_sync.py --remove csv:zerodha           # delete one source's snapshots
python scripts/portfolio_sync.py --source cas --file CAS.pdf --costs costs.json   # asks for the PDF password; --password-stdin reads it from stdin
```
Add `--json` to any of these for one machine-readable line (what the app reads); the exit code is 0 on success and 1 on a failed sync.

For a CSV, copy `config/broker_mappings/generic_example.json` to a `*.local.json` file and set each column to your export's header. Only a generic example ships: no broker's export layout was verified from official documentation. Every row that isn't imported is listed with its line and reason (negative quantity, non-numeric price, bad symbol or exchange, blank row); repeated lots of one stock are merged and reported.

**Privacy design.**
- During a sync the process may reach **only that adapter's broker host** (`scripts/net_guard.py` blocks every other resolve/connect); a CSV import makes no network call.
- Calls are read-only GETs (plus Dhan's price POST, which only reads); no order endpoint is used.
- When answering a question nothing goes out at all — the models run on 127.0.0.1 (tested).
- Snapshots stay in `data/portfolio/portfolio.db` (git-ignored); a failed sync never wipes the last good snapshot; an unavailable price is stored as unavailable, never 0; every answer built from a snapshot names its source and as-of time.
- **Limitation:** one-click hosted "Connect broker" needs each broker's partner approval (Dhan documents a partner consent flow; Upstox multi-user access appears to require onboarding; Groww documents no third-party access). It is designed, not built — `docs/hosted-oauth-design.md`.

### Hosting (the website) — groundwork done, not deployed yet
Hosted mode is switched on by environment variables; without them PortIQ is the local app above.

| Setting | Meaning |
|---|---|
| `PORTIQ_MODE=hosted` | sign-in required, per-user data, privacy notice, no broker connections; may bind beyond 127.0.0.1 |
| `PORTIQ_DATA_DIR` | where users' folders live (default `/data/portiq` — a Space's persistent storage) |
| `PORTIQ_LLM_URL`, `PORTIQ_LLM_MODEL`, `PORTIQ_LLM_API_KEY`, `PORTIQ_ALLOW_REMOTE_LLM=1` | the model server (Inference Endpoint) — see the model section above |

Google sign-in uses Streamlit's built-in `st.login` (OpenID Connect). Create an OAuth client in Google Cloud and put it in `finrag/.streamlit/secrets.toml` (git-ignored):

```toml
[auth]
redirect_uri = "https://<your-space>.hf.space/oauth2callback"
cookie_secret = "<a long random string>"
client_id = "<from Google Cloud>"
client_secret = "<from Google Cloud>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

**Try hosted mode on this machine** (no Google client needed): `PORTIQ_DEV_LOGIN` signs you in as a test user, and is ignored unless the app is bound to 127.0.0.1:

```bash
PORTIQ_MODE=hosted PORTIQ_DATA_DIR=/tmp/portiq-users PORTIQ_DEV_LOGIN=alice@test streamlit run app.py --server.port 8502
```

Still to do (deploy): the Space, its persistent storage, shipping the document store (~720 MB) privately, the Google OAuth client, and the Inference Endpoint. Unverified: whether sign-in works when the Space is opened inside huggingface.co's page frame — use the Space's own `*.hf.space` address.

### Troubleshooting: "no models" after a restart
Ollama's models live on the external drive (Ollama → Settings → Model location). If the Mac restarts and Ollama launches before the drive mounts, it logs `models path not accessible, using default` and serves an empty default folder — `ollama list` shows nothing and FinRAG can't answer. Quit and reopen Ollama (or restart its server) once the drive is mounted.

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
- [x] Golden QA set + evaluation harness (Day 12)
- [x] Model comparison — qwen2.5:7b chosen (Day 12)
- [x] Per-document cap for retrieval diversity (soft cap, weak evidence)
- [x] Exact search + CPU embeddings (Day 12)
- [x] Streamlit chat UI (Day 13)
- [x] Portfolio: local price data and valuation (Day 14)
- [x] Expand the SEC corpus to 25 companies (Day 15)
- [x] Tier 1 gaps the eval exposed (Day 16)
- [x] Filing refresh (Day 16)
- [x] Full 59-question answer evaluation on the expanded corpus (Day 16)
- [x] Cut invented examples in concept answers from 6/22 to 1/22 (Day 16)
- [x] Live portfolio panel: add stocks in the UI, NSE/BSE in ₹, 60 s refresh (Day 17)
- [x] MLX backend as an Ollama alternative, with automatic fallback (Day 17)
- [x] Indian annual reports in the corpus — 15 NSE companies (Day 18)
- [x] Two-pane research-terminal UI (Day 18)
- [x] Getting annual-report figures right: subsidiary labels and filter, hybrid search for annual reports, annual-report prompt rules (Day 18)
- [x] PortIQ branding (Day 18)
- [x] Qwen2.5-7B vs Qwen3-8B comparison — Qwen3 is not a clear upgrade; Qwen2.5-7B stays the default (Day 19)
- [x] LLM backend accepts any OpenAI-compatible server (vLLM / Hugging Face endpoint) by URL, remote only with explicit opt-in (Day 19)
- [x] Type-ahead stock search when adding a holding (Day 19)
- [x] Faster answers (39 s → 28 s on the Mac), verifiable answer cards, Insights tab, How it works (Day 19)
- [x] Annual questions read the 10-K — figures correct 11/17 → 14/17 (Day 19)
- [x] Indian Standalone|Consolidated tables labelled — figures correct 14/17 → 15/17 (Day 19)
- [x] Portfolio sources: CSV import + read-only Upstox / Dhan / Groww sync with your own credentials (Day 20)
- [ ] Run each broker adapter once against a live account (only fixture-tested so far)
- [ ] Hosted "Connect broker" — needs broker partner approval (design: docs/hosted-oauth-design.md)
- [ ] Model rounding / unit conversion (HDFC Bank ₹31,05,250.47 → ₹31,05,251; Tesla → "$94.827 billion")
- [ ] Evaluate a 14B model on a rented 24 GB GPU with the same harness before choosing the production model
- [ ] Multi-user app before hosting: accounts, per-user holdings, a server-safe privacy policy
- [ ] Replace the SVG logo recreation with the original PortIQ artwork
- [ ] Indian quarterly results
