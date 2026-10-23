"""
Per-source store update test (build_vectorstore --only-source). A throwaway Chroma
database and a fake embedder — no model, the real store is untouched.

    python tests/test_build_update.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import build_vectorstore as bv  # noqa: E402


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0, 0.5] for t in texts]


def chunk(cid, source, text, dup=False):
    return {"chunk_id": cid, "source": source, "text": text, "tier": 2, "is_duplicate": dup, "topic_tags": []}


def test_update_replaces_only_one_source() -> None:
    with tempfile.TemporaryDirectory() as d:
        bv.CHROMA_DIR, bv.CHUNKS_PATH = Path(d) / "db", Path(d) / "chunks.jsonl"
        write = lambda rows: bv.CHUNKS_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        write([chunk("sec-1", "sec_edgar", "10-K text"), chunk("nse-1", "nse_annual_report", "old row"),
               chunk("nse-2", "nse_annual_report", "old row 2")])
        bv.build_collection(FakeEmbedder())
        write([chunk("sec-1", "sec_edgar", "CHANGED but not re-embedded"),
               chunk("nse-1", "nse_annual_report", "Revenue — Standalone: 1; Consolidated: 2"),
               chunk("nse-3", "nse_annual_report", "new chunk"), chunk("nse-4", "nse_annual_report", "dup", dup=True)])
        col = bv.update_source("nse_annual_report", FakeEmbedder())
        got = col.get(include=["documents"])
        docs = dict(zip(got["ids"], got["documents"]))
        assert docs == {"sec-1": "10-K text", "nse-1": "Revenue — Standalone: 1; Consolidated: 2", "nse-3": "new chunk"}
    print("PASS: --only-source replaces that source's chunks (stale ones removed, duplicates skipped), others untouched")


if __name__ == "__main__":
    test_update_replaces_only_one_source()
    print("\nAll build-update tests passed.")
