# FinRAG — Financial Knowledge RAG Agent

A production-style **Retrieval-Augmented Generation (RAG)** pipeline over a three-tier financial corpus: plain-language explainers, SEC filings (10-K / 10-Q), and arXiv market-microstructure research papers.

Built day-by-day as a structured learning project — each script corresponds to one pipeline stage.

---

## Architecture

```
corpus_manifest.yaml          ← single source of truth for what gets ingested
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                   DATA INGESTION                    │
│  Day 2 │ scrape_tier1.py       → Tier 1 explainers  │
│  Day 3 │ scrape_tier2_edgar.py → Tier 2 SEC filings │
│  Day 3 │ scrape_tier3_arxiv.py → Tier 3 arXiv papers│
└─────────────────────────────────────────────────────┘
         │  data/raw/{tier1,tier2,tier3}/*.{txt,meta.json}
         ▼
┌─────────────────────────────────────────────────────┐
│                   PROCESSING                        │
│  Day 4 │ chunk_documents.py  → chunks.jsonl         │
│  Day 5 │ enrich_metadata.py  → chunks_enriched.jsonl│
└─────────────────────────────────────────────────────┘
         │  data/processed/chunks_enriched.jsonl
         ▼
┌─────────────────────────────────────────────────────┐
│               VECTOR STORE (Day 6)                  │
│  build_vectorstore.py → ChromaDB collection         │
│  Model: BAAI/bge-small-en-v1.5                     │
└─────────────────────────────────────────────────────┘
```

---

## Corpus

| Tier | Source | Content | Target |
|------|--------|---------|--------|
| 1 | Investopedia, Khan Academy, RBI/SEBI | Finance explainers (EMI, compound interest, mutual funds, options…) | ~35 docs |
| 2 | SEC EDGAR | 10-K / 10-Q — AAPL, MSFT, JPM, TSLA, XOM | ~25 docs |
| 3 | arXiv (q-fin.TR / q-fin.MF / q-fin.ST) | Market microstructure, HFT, optimal execution, limit order books | ~35 papers |

**Pipeline stats (last run):**
- 94 documents → 8,808 raw chunks → 8,071 unique chunks embedded
- 457 chunks contain tables (5.2%) | 2,489 contain formulas (28.3%)
- 737 exact-text duplicates flagged and dropped before embedding

---

## Project Structure

```
RAG LLM/
├── corpus_manifest.yaml          # Ingestion config — sources, targets, metadata schema
├── README.md
├── .gitignore
└── finrag/
    ├── scripts/
    │   ├── validate_manifest.py  # Day 1 — validate corpus_manifest.yaml
    │   ├── scrape_tier1.py       # Day 2 — scrape finance explainer pages
    │   ├── scrape_tier2_edgar.py # Day 3 — fetch SEC EDGAR filings
    │   ├── scrape_tier3_arxiv.py # Day 3 — fetch arXiv q-fin papers
    │   ├── chunk_documents.py    # Day 4 — structure-aware chunking → chunks.jsonl
    │   ├── enrich_metadata.py    # Day 5 — topic_tags, contains_table, is_duplicate
    │   └── build_vectorstore.py  # Day 6 — embed & load into ChromaDB
    └── data/
        ├── raw/
        │   ├── tier1/            # {slug}.txt + {slug}.meta.json
        │   ├── tier2/            # SEC filings
        │   └── tier3/            # arXiv papers
        └── processed/
            ├── chunks.jsonl           # Day 4 output
            └── chunks_enriched.jsonl  # Day 5 output → consumed by Day 6
```

> **ChromaDB** is stored at `~/.finrag_chroma_db` (outside the repo). SQLite requires
> POSIX file locking which is not supported on exFAT external drives.

---

## Setup

### Prerequisites

- Python 3.12+
- macOS / Linux

### Install dependencies

```bash
pip install chromadb sentence-transformers requests pyyaml beautifulsoup4
```

### Run the full pipeline

```bash
cd "RAG LLM/finrag"

# Day 1 — validate manifest
python scripts/validate_manifest.py

# Day 2 — scrape finance explainers
python scripts/scrape_tier1.py

# Day 3 — scrape SEC filings & arXiv papers
python scripts/scrape_tier2_edgar.py
python scripts/scrape_tier3_arxiv.py

# Day 4 — chunk all raw documents
python scripts/chunk_documents.py

# Day 5 — enrich chunk metadata
python scripts/enrich_metadata.py

# Day 6 — embed & build ChromaDB vector store
python scripts/build_vectorstore.py --test-query "what is compound interest"
```

### Python path note

If you get `ModuleNotFoundError: No module named 'chromadb'`, your terminal `python`
points to a different installation. Use the explicit interpreter or the shebang in each script:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/build_vectorstore.py
```

---

## Chunking Strategy (Day 4)

`chunk_documents.py` implements **structure-aware chunking**:

- Documents split by `## Heading` section markers first
- Within each section, paragraphs / list-items / table blobs are atomic units
- Units greedily packed into token-budgeted chunks (max ~450 tokens, min ~40 tokens)
- Oversized single units are sentence-split as a fallback (`was_sub_split=True`)
- Tiny trailing chunks merged into their neighbour

---

## Metadata Enrichment (Day 5)

`enrich_metadata.py` adds three fields to every chunk:

| Field | Type | Description |
|-------|------|-------------|
| `contains_table` | `bool` | Pipe / ASCII-separator heuristic detection |
| `topic_tags` | `list[str]` | Rule-based labels: `derivatives`, `market_microstructure`, `sec_filings`, etc. |
| `is_duplicate` | `bool` | MD5 hash dedup — first occurrence kept, rest flagged and dropped at embedding time |

These are flattened into ChromaDB metadata for **filtered retrieval** (e.g. tier-3 only, tables only).

---

## Embedding Model

**BAAI/bge-small-en-v1.5** (via `sentence-transformers`)

- 33M parameters, 384-dim embeddings — fast on CPU, strong retrieval quality
- Asymmetric: queries need the prefix `"Represent this sentence for searching relevant passages: "` (handled automatically)
- Similarity: **cosine** (`hnsw:space=cosine` in ChromaDB)

---

## Example Query Output

```
=== Query: 'what is compound interest' ===

[dist=0.2911] tier=2 source=sec_edgar heading='JPM 10-K filed 2026-02-13'
  The Firm generally uses interest rate derivatives to manage the risk associated
  with changes in interest rates. Fixed-rate assets and liabilities appr...

[dist=0.2929] tier=2 source=sec_edgar heading='JPM 10-K filed 2026-02-13'
  Structured notes: financial instruments whose cash flows are linked to the
  movement in one or more indexes, interest rates, foreign exchanges...

[dist=0.3002] tier=2 source=sec_edgar heading='JPM 10-K filed 2026-02-13'
  The Firm evaluates its structural interest rate risk through earnings-at-risk...
```

---

## Roadmap

- [ ] Day 7: RAG query layer — connect ChromaDB retrieval to an LLM (Gemini / GPT)
- [ ] Day 8: Streamlit / Gradio chat UI
- [ ] Day 9: Metadata-filtered retrieval (tier-aware, table-aware queries)
- [ ] Day 10: Evaluation — recall@k, MRR against a golden QA set
