import numpy as np
import pytest

from codesearch.index import METHODS, CodeIndex


@pytest.mark.parametrize("method", METHODS)
def test_search_returns_ranked_units(index, method):
    hits = index.search("look up a key in the hash table", k=5, method=method)
    assert len(hits) == 5
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({u.id for u, _ in hits}) == 5


def test_hybrid_combines_both_rankings(index):
    q = "release memory of the string buffer"
    bm = index.scores(q, "bm25")
    dn = index.scores(q, "dense")
    fused = index.combine("hybrid", bm, dn)
    np.testing.assert_allclose(fused, index.scores(q, "hybrid"))
    top_bm, top_dn = int(np.argmax(bm)), int(np.argmax(dn))
    assert fused[top_bm] >= 1 / 61 and fused[top_dn] >= index.dense_weight / 61
    assert fused.max() <= (1 + index.dense_weight) / 61 + 1e-12


def test_unknown_method_rejected(index):
    with pytest.raises(ValueError):
        index.scores("x", "magic")


def test_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = CodeIndex.load(tmp_path / "idx")
    assert [u.id for u in loaded.units] == [u.id for u in index.units]
    assert loaded.graph.edges() == index.graph.edges()
    np.testing.assert_array_equal(loaded.embeddings, index.embeddings)
    q = "parse key value line"
    assert loaded.ranking(q) == index.ranking(q)


def test_open_reuses_cache(tmp_path, fixture_root):
    import shutil

    repo = tmp_path / "repo"
    shutil.copytree(fixture_root, repo)
    first = CodeIndex.open(repo)
    assert (repo / ".codesearch" / "embeddings.npy").exists()
    marker = np.full_like(first.embeddings, 0.5)
    np.save(repo / ".codesearch" / "embeddings.npy", marker)
    again = CodeIndex.open(repo)
    np.testing.assert_array_equal(again.embeddings, marker)  # loaded, not recomputed
    (repo / "src" / "extra.c").write_text("int extra(void) { return 1; }\n")
    rebuilt = CodeIndex.open(repo)
    assert len(rebuilt.units) == len(first.units) + 1
    assert not np.allclose(rebuilt.embeddings[: len(first.units)], 0.5)


def test_find_by_name_and_id(index):
    assert [u.id for u in index.find("log_msg")] == ["src/main.c::log_msg", "src/strbuf.c::log_msg"]
    assert [u.id for u in index.find("src/strbuf.c::log_msg")] == ["src/strbuf.c::log_msg"]
    assert index.find("missing") == []
