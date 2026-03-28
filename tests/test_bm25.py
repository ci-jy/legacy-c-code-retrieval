import math

import numpy as np

from codesearch.bm25 import BM25


def test_hand_computed_scores():
    docs = [["a", "b"], ["a"], ["c", "c", "b"]]
    bm = BM25(docs, k1=1.2, b=0.75)
    avgdl = 2.0
    idf_b = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    idf_c = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))

    def term(tf, dl, idf):
        return idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl))

    s = bm.scores(["b"])
    np.testing.assert_allclose(s, [term(1, 2, idf_b), 0.0, term(1, 3, idf_b)])
    s = bm.scores(["c", "b"])
    np.testing.assert_allclose(s, [term(1, 2, idf_b), 0.0, term(1, 3, idf_b) + term(2, 3, idf_c)])


def test_repeated_and_unknown_terms():
    bm = BM25([["x", "y"], ["y"]])
    np.testing.assert_array_equal(bm.scores(["x", "x"]), bm.scores(["x"]))
    np.testing.assert_array_equal(bm.scores(["zzz"]), [0.0, 0.0])


def test_empty_index():
    assert BM25([]).scores(["a"]).shape == (0,)


def test_bm25_ranking_is_deterministic(fixture_root):
    from codesearch.index import CodeIndex

    q = "insert a value into the hash table"
    a = CodeIndex.build(fixture_root)
    b = CodeIndex.build(fixture_root)
    np.testing.assert_array_equal(a.bm25.scores(["hash"]), b.bm25.scores(["hash"]))
    assert a.ranking(q, "bm25") == b.ranking(q, "bm25")
    assert a.ranking(q, "bm25") == a.ranking(q, "bm25")
    assert a.units[a.ranking(q, "bm25")[0]].name == "table_put"
