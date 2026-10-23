"""
Exact-search index tests. No Chroma needed (a fake collection supplies vectors).

    python tests/test_exact_index.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rag_loop import ExactIndex, _matches, annual_titles  # noqa: E402


class FakeCollection:
    def __init__(self, vectors, metadatas):
        self._data = {"ids": [f"c{i}" for i in range(len(vectors))], "documents": [f"doc {i}" for i in range(len(vectors))],
                      "metadatas": metadatas, "embeddings": vectors}

    def get(self, include):
        return self._data


def test_exact_ranking_distances_and_ties() -> None:
    vecs = [[1, 0], [0.8, 0.6], [0, 1], [0.8, 0.6]]  # c1 and c3 tie exactly
    meta = [{"tier": 1}, {"tier": 2, "company_ticker": "AAPL"}, {"tier": 3}, {"tier": 2, "company_ticker": "MSFT"}]
    idx = ExactIndex(FakeCollection(vecs, meta))
    res = idx.query([[1, 0]], n_results=3)
    assert res["ids"][0] == ["c0", "c1", "c3"], res["ids"]           # exact order, tie broken by row
    assert np.allclose(res["distances"][0], [0.0, 0.2, 0.2])          # cosine distance = 1 - cos
    assert idx.query([[1, 0]], 3)["ids"] == res["ids"]                # deterministic
    print("PASS: exact cosine ranking, Chroma-style distances, deterministic tie-breaking")


def test_where_filters() -> None:
    meta = [{"tier": 1}, {"tier": 2, "company_ticker": "AAPL"}, {"tier": 3}, {"tier": 2, "company_ticker": "MSFT"}]
    idx = ExactIndex(FakeCollection([[1, 0], [0.8, 0.6], [0, 1], [0.8, 0.6]], meta))
    aapl = idx.query([[1, 0]], 5, {"$and": [{"tier": 2}, {"company_ticker": "AAPL"}]})
    assert aapl["ids"] == [["c1"]]
    assert idx.query([[1, 0]], 5, {"tier": {"$in": [2, 3]}})["ids"][0] == ["c1", "c3", "c2"]
    assert idx.query([[1, 0]], 5, {"tier": 1})["ids"] == [["c0"]]
    assert _matches({"tier": 3}, {"$or": [{"tier": 1}, {"tier": 3}]})
    try:
        _matches({"tier": 1}, {"tier": {"$gt": 0}})
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported operators must fail loudly, not match everything")
    print("PASS: where filters ($and, $in, $or, equality); unsupported operators raise")


def test_hybrid_keyword_fusion() -> None:
    """A passage that names the asked-for line item is promoted even when its vector is farther away."""
    vecs = [[1, 0], [0.98, 0.2], [0.95, 0.31], [0.6, 0.8], [0, 1]]
    docs = ["Jio Platforms revenue from services table", "Retail segment revenue table", "cash flow statement",
            "In FY 2025-26 we delivered consolidated revenues and net profit of the group", "unrelated paper"]
    meta = [{"tier": 2, "company_ticker": "RELIANCE.NS"}] * 4 + [{"tier": 3}]
    coll = FakeCollection(vecs, meta)
    coll._data["documents"] = docs
    idx = ExactIndex(coll)
    where = {"$and": [{"tier": 2}, {"company_ticker": "RELIANCE.NS"}]}
    dense = idx.query([[1, 0]], 2, where)["ids"][0]
    assert "c3" not in dense                                                   # dense alone misses it
    res = idx.hybrid_query("What were the consolidated revenues and net profit?", [[1, 0]], 2, where)
    assert "c3" in res["ids"][0] and "c4" not in res["ids"][0], res["ids"]      # keyword match promoted; filter kept
    i = res["ids"][0].index("c3")
    assert np.isclose(res["distances"][0][i], 1 - 0.6)                         # displayed distance is the true cosine
    assert res["sort_keys"][0] == sorted(res["distances"][0])                  # order-preserving key, same scale
    assert idx.hybrid_query("the of and", [[1, 0]], 2, where)["ids"] == [dense] # only stopwords: dense order
    print("PASS: hybrid search fuses keyword and dense rankings; distances stay true, order survives sorting")


def test_annual_titles() -> None:
    metas = [{"company_ticker": "NVDA", "doc_title": "NVDA 10-K filed 2026-02-25"},
             {"company_ticker": "NVDA", "doc_title": "NVDA 10-Q filed 2026-05-28"},
             {"company_ticker": "NVDA", "doc_title": "NVDA 10-K filed 2026-02-25"},
             {"company_ticker": "AAPL", "doc_title": "AAPL 10-K filed 2025-10-31"},
             {"company_ticker": "TCS.NS", "doc_title": "Tata Consultancy Services (TCS) Annual Report FY2025-26"}]
    assert annual_titles(metas, "NVDA") == ["NVDA 10-K filed 2026-02-25"]
    assert annual_titles(metas, "TCS.NS") == []          # annual reports are annual already: no filter
    assert annual_titles(metas, "MSFT") == []
    print("PASS: a company's 10-K titles are found for annual questions")


if __name__ == "__main__":
    test_exact_ranking_distances_and_ties()
    test_where_filters()
    test_hybrid_keyword_fusion()
    test_annual_titles()
    print("\nAll exact-index tests passed.")
