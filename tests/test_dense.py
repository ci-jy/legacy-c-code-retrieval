import numpy as np

from codesearch.dense import Embedder
from codesearch.index import CodeIndex, dense_text


def test_embeddings_are_normalised_and_repeatable(units):
    e = Embedder()
    texts = [dense_text(u, True) for u in units[:4]]
    a = e.encode_documents(texts)
    b = e.encode_documents(texts)
    assert a.shape[0] == 4 and a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_dense_ranking_is_deterministic(fixture_root):
    q = "grow a dynamic string buffer when it runs out of space"
    a = CodeIndex.build(fixture_root)
    b = CodeIndex.build(fixture_root)
    ra, rb = a.ranking(q, "dense"), b.ranking(q, "dense")
    assert ra == rb
    np.testing.assert_allclose(a.scores(q, "dense"), b.scores(q, "dense"), atol=1e-6)
    assert a.units[ra[0]].file == "src/strbuf.c"


def test_query_and_batch_encoding_agree():
    e = Embedder()
    qs = ["free all entries", "parse a config line"]
    batch = e.encode_queries(qs)
    np.testing.assert_allclose(batch[1], e.encode_query(qs[1]), atol=1e-5)
