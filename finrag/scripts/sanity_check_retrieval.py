"""
Day 7: Retrieval sanity check.

Runs a fixed set of hand-picked queries spanning all three tiers against the
Chroma collection built in build_vectorstore.py (Day 6), prints the top-k
results with metadata, and auto-flags likely issues (tier drift, weak
matches, truncated chunks) so you can decide what to fix before Week 2.

Also writes a JSON report to data/eval/retrieval_sanity_check.json — this
becomes raw material for Day 15's hand-crafted eval set, so don't skip
saving it even if you're just eyeballing results today.

Usage:
    python scripts/sanity_check_retrieval.py
    python scripts/sanity_check_retrieval.py --k 3
"""
import argparse
import json
from pathlib import Path
import chromadb

ROOT = Path(__file__).parent.parent
# Must match the path used in build_vectorstore.py — stored in home dir to
# avoid SQLite POSIX file-locking issues on the exFAT external drive.
CHROMA_DIR = Path.home() / ".finrag_chroma_db"
COLLECTION_NAME = "finrag_chunks"
REPORT_PATH = ROOT / "data" / "eval" / "retrieval_sanity_check.json"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---- 15 manual test queries spanning all three tiers ----
# expected_tier is a *hypothesis* to check against, not ground truth — some
# queries deliberately straddle tiers (marked with notes) to test for drift.
TEST_QUERIES = [
    # Tier 1 — fundamentals
    {"query": "What is an EMI and how is it calculated?", "expected_tier": 1},
    {"query": "How does compound interest work?", "expected_tier": 1},
    {"query": "What is the difference between a mutual fund and a stock?", "expected_tier": 1},
    {"query": "What does bond duration mean?", "expected_tier": 1,
     "notes": "trap: 'duration' also appears in quant/microstructure contexts"},
    {"query": "How do call and put options work?", "expected_tier": 1},
    {"query": "What is a systematic investment plan (SIP)?", "expected_tier": 1},

    # Tier 2 — SEC filings
    {"query": "What risk factors does the company disclose in its 10-K?", "expected_tier": 2},
    {"query": "What was the company's revenue and gross profit last quarter?", "expected_tier": 2},
    {"query": "What does the management discussion and analysis section say about performance?", "expected_tier": 2},
    {"query": "What litigation risks are mentioned in the filing?", "expected_tier": 2},

    # Tier 3 — market microstructure / HFT
    {"query": "How does a limit order book work?", "expected_tier": 3},
    {"query": "What is latency arbitrage in high-frequency trading?", "expected_tier": 3},
    {"query": "How is bid-ask spread modeled in market microstructure research?", "expected_tier": 3},
    {"query": "What is optimal execution and price impact?", "expected_tier": 3},
    {"query": "How do market makers manage order flow toxicity?", "expected_tier": 3},
]


def load_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    class Embedder:
        def embed_query(self, text: str) -> list[float]:
            return model.encode([BGE_QUERY_PREFIX + text]).tolist()[0]
    return Embedder()


def flag_issues(query_spec: dict, results: dict) -> list[str]:
    """Cheap automated checks — not a replacement for reading the actual text."""
    flags = []
    tiers = [m["tier"] for m in results["metadatas"][0]]
    distances = results["distances"][0]

    if not tiers:
        flags.append("NO RESULTS RETURNED — check collection is populated")
        return flags

    expected = query_spec.get("expected_tier")
    if expected is not None and tiers[0] != expected:
        flags.append(f"TOP RESULT TIER MISMATCH: expected tier {expected}, got tier {tiers[0]}")

    if expected is not None:
        match_count = sum(1 for t in tiers if t == expected)
        if match_count == 0:
            flags.append(f"ZERO of top-{len(tiers)} results are from expected tier {expected}")
        elif match_count < len(tiers) / 2:
            flags.append(f"Only {match_count}/{len(tiers)} results from expected tier {expected}")

    if distances[0] > 0.5:
        flags.append(f"WEAK TOP MATCH: distance {distances[0]:.3f} is high for the best result")

    for doc in results["documents"][0]:
        if len(doc.strip()) < 30:
            flags.append("Suspiciously short chunk in results — check for over-aggressive splitting")
            break

    return flags


def run(k: int = 5) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"[ERROR] Collection '{COLLECTION_NAME}' not found at {CHROMA_DIR} — run build_vectorstore.py first (Day 6).")
        return

    if collection.count() == 0:
        print("[ERROR] Collection is empty — run build_vectorstore.py first (Day 6).")
        return

    embedder = load_embedder()
    report = []
    total_flags = 0

    print(f"=== Retrieval Sanity Check: {len(TEST_QUERIES)} queries, k={k} ===\n")

    for i, spec in enumerate(TEST_QUERIES, 1):
        query = spec["query"]
        query_embedding = embedder.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=k)

        flags = flag_issues(spec, results)
        total_flags += len(flags)

        print(f"[{i}] \"{query}\"  (expected tier: {spec.get('expected_tier', '?')})")
        if spec.get("notes"):
            print(f"    note: {spec['notes']}")

        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            print(f"    - [dist={dist:.3f}] tier={meta['tier']} src={meta['source']} "
                  f"heading='{meta['section_heading']}'")
            print(f"      {doc[:120].replace(chr(10), ' ')}...")

        if flags:
            print("    ⚠ FLAGS:")
            for f in flags:
                print(f"      - {f}")
        else:
            print("    ✅ no automated flags")
        print()

        report.append({
            "query": query,
            "expected_tier": spec.get("expected_tier"),
            "notes": spec.get("notes"),
            "top_k_results": [
                {"text": doc, "tier": meta["tier"], "source": meta["source"],
                 "section_heading": meta["section_heading"], "distance": dist}
                for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
            ],
            "flags": flags,
        })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    queries_with_flags = sum(1 for r in report if r["flags"])
    print("=== Summary ===")
    print(f"Queries run          : {len(TEST_QUERIES)}")
    print(f"Queries with flags   : {queries_with_flags}")
    print(f"Total flags raised   : {total_flags}")
    print(f"Report saved to      : {REPORT_PATH}")
    if queries_with_flags > len(TEST_QUERIES) / 3:
        print("\n⚠ More than a third of queries flagged — worth revisiting chunking/embedding "
              "before moving to Day 8, not just patching individually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Number of results per query")
    args = parser.parse_args()
    run(k=args.k)