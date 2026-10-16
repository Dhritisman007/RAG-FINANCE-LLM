"""
Exact-search index tests. No Chroma needed (a fake collection supplies vectors).

    python tests/test_exact_index.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rag_loop import ExactIndex, _matches  # noqa: E402


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


if __name__ == "__main__":
    test_exact_ranking_distances_and_ties()
    test_where_filters()
    print("\nAll exact-index tests passed.")
