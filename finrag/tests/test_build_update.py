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


def chunk(cid, source, text, dup=False, ticker=""):
    return {"chunk_id": cid, "source": source, "text": text, "tier": 2, "is_duplicate": dup, "topic_tags": [],
            "company_ticker": ticker}


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


def test_update_adds_only_given_companies() -> None:
    with tempfile.TemporaryDirectory() as d:
        bv.CHROMA_DIR, bv.CHUNKS_PATH = Path(d) / "db", Path(d) / "chunks.jsonl"
        write = lambda rows: bv.CHUNKS_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        write([chunk("infy-1", "nse_annual_report", "Infosys row", ticker="INFY.NS"),
               chunk("sec-1", "sec_edgar", "10-K text")])
        bv.build_collection(FakeEmbedder())
        write([chunk("infy-1", "nse_annual_report", "CHANGED but not re-embedded", ticker="INFY.NS"),
               chunk("sec-1", "sec_edgar", "10-K text"),
               chunk("coal-1", "nse_annual_report", "Coal India row", ticker="COALINDIA.NS"),
               chunk("ntpc-1", "nse_annual_report", "NTPC row", ticker="NTPC.NS")])
        col = bv.update_source("nse_annual_report", FakeEmbedder(), tickers=["COALINDIA.NS", "NTPC.NS"])
        got = col.get(include=["documents"])
        assert dict(zip(got["ids"], got["documents"])) == {
            "infy-1": "Infosys row", "sec-1": "10-K text", "coal-1": "Coal India row", "ntpc-1": "NTPC row"}
        col = bv.update_source("nse_annual_report", FakeEmbedder(), tickers=["COALINDIA.NS"])  # re-run: replaced, not doubled
        assert col.count() == 4
    print("PASS: --tickers adds just those companies' chunks (re-runs replace them); other companies untouched")


def test_update_reembeds_only_changed_chunks() -> None:
    with tempfile.TemporaryDirectory() as d:
        bv.CHROMA_DIR, bv.CHUNKS_PATH = Path(d) / "db", Path(d) / "chunks.jsonl"
        write = lambda rows: bv.CHUNKS_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        write([chunk("r-1", "nse_annual_report", "Revenue J3,71,085 Crore", ticker="RELIANCE.NS"),
               chunk("r-2", "nse_annual_report", "unchanged", ticker="RELIANCE.NS"),
               chunk("r-3", "nse_annual_report", "dropped later", ticker="RELIANCE.NS"),
               chunk("sec-1", "sec_edgar", "10-K text")])
        bv.build_collection(FakeEmbedder())

        class Counting(FakeEmbedder):
            seen: list = []

            def embed_documents(self, texts):
                self.seen += texts
                return super().embed_documents(texts)
        write([chunk("r-1", "nse_annual_report", "Revenue ₹3,71,085 Crore", ticker="RELIANCE.NS"),
               chunk("r-2", "nse_annual_report", "unchanged", ticker="RELIANCE.NS"),
               chunk("r-4", "nse_annual_report", "new", ticker="RELIANCE.NS"), chunk("sec-1", "sec_edgar", "10-K text")])
        emb = Counting()
        col = bv.update_source("nse_annual_report", emb, changed_only=True)
        got = col.get(include=["documents"])
        assert dict(zip(got["ids"], got["documents"])) == {
            "r-1": "Revenue ₹3,71,085 Crore", "r-2": "unchanged", "r-4": "new", "sec-1": "10-K text"}
        assert emb.seen == ["Revenue ₹3,71,085 Crore", "new"], emb.seen      # unchanged chunks not re-embedded
        assert bv.changed_chunks(col, [c for c in bv.load_chunks() if c["source"] == "nse_annual_report"],
                                 {"source": "nse_annual_report"}) == ([], [])  # a second run has nothing to do
    print("PASS: --changed-only re-embeds just changed/new chunks, drops removed ones, and leaves the rest")


if __name__ == "__main__":
    test_update_replaces_only_one_source()
    test_update_adds_only_given_companies()
    test_update_reembeds_only_changed_chunks()
    print("\nAll build-update tests passed.")
