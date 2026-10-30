"""
Index recall check: does Chroma's approximate (HNSW) search return the same
top-k as exact brute-force search?

Since Day 12 the RAG loop doesn't query the HNSW index at all — it runs exact
search over the stored vectors (rag_loop.ExactIndex) — so this is a diagnostic
of Chroma's own index, useful if the corpus ever grows past exact search.

HNSW can silently leave whole topic clusters unreachable (it once hid every
SIP / YTM / CAGR chunk), so run this after build_vectorstore.py whenever the
corpus changes. Queries = the Day 7 sanity-check set + every Tier 1 section
heading phrased as a question.

Usage:
    python scripts/check_index_recall.py
    python scripts/check_index_recall.py --k 10
"""
import argparse
import json
import sys
from pathlib import Path

import chromadb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_vectorstore import (  # noqa: E402
    BGE_QUERY_PREFIX, CHROMA_DIR, CHUNKS_PATH, COLLECTION_NAME, EMBED_DEVICE, EMBED_MODEL_NAME,
)
from sanity_check_retrieval import TEST_QUERIES  # noqa: E402


def load_queries() -> list[str]:
    headings = set()
    for line in open(CHUNKS_PATH, encoding="utf-8"):
        chunk = json.loads(line)
        if chunk["tier"] == 1 and chunk["section_heading"].endswith("?"):
            headings.add(chunk["section_heading"])
    return [q["query"] for q in TEST_QUERIES] + sorted(headings)


def run(k: int = 5) -> float:
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.get_collection(COLLECTION_NAME)
    from rag_loop import get_all
    stored = get_all(collection, ["embeddings"])
    ids = np.array(stored["ids"])
    doc_matrix = np.array(stored["embeddings"])
    doc_matrix /= np.linalg.norm(doc_matrix, axis=1, keepdims=True)

    queries = load_queries()
    model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    query_matrix = model.encode([BGE_QUERY_PREFIX + q for q in queries], normalize_embeddings=True)

    recalls, total_misses = [], []
    for query, emb in zip(queries, query_matrix):
        exact = set(ids[np.argsort(-(doc_matrix @ emb))[:k]])
        approx = set(collection.query(query_embeddings=[emb.tolist()], n_results=k)["ids"][0])
        recalls.append(len(exact & approx) / k)
        if recalls[-1] == 0:
            total_misses.append(query)

    mean = float(np.mean(recalls))
    print(f"=== Index recall@{k} vs exact search: {len(queries)} queries ===")
    print(f"Mean recall@{k}     : {mean:.3f}")
    print(f"Perfect queries    : {sum(r == 1 for r in recalls)}/{len(queries)}")
    print(f"Total misses (0/{k}): {len(total_misses)} {total_misses[:10]}")
    if mean < 0.98 or total_misses:
        print("\nWARNING: index is missing true nearest neighbours — raise HNSW_CONFIG "
              "max_neighbors / ef_construction in build_vectorstore.py and rebuild.")
    return mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    run(k=args.k)
