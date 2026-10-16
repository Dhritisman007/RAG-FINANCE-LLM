#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
Day 6: Build the ChromaDB vector store.

Reads data/processed/chunks_enriched.jsonl (from Day 5), embeds every chunk
with BAAI/bge-small-en-v1.5, and loads them into a persistent Chroma
collection at data/chroma_db/, with cleaned/flattened metadata attached
for later filtering (tier, contains_table, topic_tags, etc.).

Usage:
    python scripts/build_vectorstore.py
    python scripts/build_vectorstore.py --test-query "what is compound interest"
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import chromadb

ROOT = Path(__file__).parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks_enriched.jsonl"
# ChromaDB uses SQLite which requires POSIX file locking — not available on
# exFAT external drives. Store the DB locally to avoid "readonly database" errors.
CHROMA_DIR = Path.home() / ".finrag_chroma_db"
COLLECTION_NAME = "finrag_chunks"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE models are asymmetric: documents are embedded as-is, but queries need
# this instruction prefix prepended for the model to align them correctly
# with document embeddings. Skipping this on the query side measurably hurts
# retrieval quality with BGE models specifically.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BATCH_SIZE = 64

# Always embed on the CPU. On Apple Silicon, sentence-transformers defaults to
# the GPU (MPS), which shares memory with Ollama: with qwen2.5:7b resident on
# an 8 GB M1, Metal ran out of memory and PyTorch silently returned corrupted
# embeddings (cosine vs. the correct vector as low as 0.28) — wrong retrieval
# and flipping citation checks, with no exception raised. bge-small is fast
# enough on the CPU (a query embeds in tens of milliseconds).
EMBED_DEVICE = "cpu"

# HNSW index settings. Chroma's default (16 neighbours, ef_construction=100)
# left whole topic clusters unreachable from some queries — e.g. every SIP /
# YTM / CAGR chunk was missed even with ef_search=1000 (recall@5 = 0 on 5 of
# 199 test queries). 32 neighbours + ef_construction=400 gave recall@5 = 1.0
# against exact search on all 199. Cosine fits normalized sentence embeddings.
HNSW_CONFIG = {"space": "cosine", "max_neighbors": 32, "ef_construction": 400, "ef_search": 200}


# ---------- 1. Embedder (real model, lazily imported so this module stays
#               testable without the heavy sentence-transformers/torch stack) ----------

class BGEEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name, device=EMBED_DEVICE)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([BGE_QUERY_PREFIX + text]).tolist()[0]


# ---------- 2. Metadata cleaning ----------
# Chroma metadata values must be str/int/float/bool — no lists, no None.

def clean_metadata(chunk: dict) -> dict:
    cleaned = {}
    for key, value in chunk.items():
        if key == "text":
            continue  # stored separately as the document body
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, list):
            cleaned[key] = ",".join(str(v) for v in value)  # e.g. topic_tags -> "emi,interest"
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


# ---------- 3. Load chunks, embed, insert ----------

def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found — run scripts/enrich_metadata.py first (Day 5).")
    return [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]


def build_collection(embedder=None) -> chromadb.Collection:
    """
    embedder: object with .embed_documents(list[str]) -> list[list[float]].
    Defaults to BGEEmbedder (real model). Pass a stand-in for testing.
    """
    if embedder is None:
        embedder = BGEEmbedder()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    # dedup: skip chunks already flagged as duplicates by Day 5's enrichment
    before = len(chunks)
    chunks = [c for c in chunks if not c.get("is_duplicate", False)]
    print(f"Dropped {before - len(chunks)} duplicate chunks, embedding {len(chunks)}")

    # Chroma silently skips ids that already exist, so a collision here would
    # quietly drop chunks from the store. Fail loudly instead.
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        raise ValueError(f"{len(dupes)} chunk_ids are not unique (e.g. {dupes[:3]}) — fix chunk_documents.py")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # fresh collection each run — re-running this script should fully rebuild,
    # not silently accumulate stale/duplicate entries from prior runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": HNSW_CONFIG},
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [clean_metadata(c) for c in batch]

        embeddings = embedder.embed_documents(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  Inserted batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    print(f"\nCollection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
    return collection


# ---------- 4. Smoke-test query ----------

def test_query(query: str, embedder=None, n_results: int = 5) -> None:
    if embedder is None:
        embedder = BGEEmbedder()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    print(f"\n=== Query: '{query}' ===")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"\n[dist={dist:.4f}] tier={meta['tier']} source={meta['source']} heading='{meta['section_heading']}'")
        print(f"  {doc[:150]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-query", type=str, default=None, help="Run a test retrieval query after building")
    args = parser.parse_args()

    build_collection()
    if args.test_query:
        test_query(args.test_query)