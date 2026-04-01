import pytest

from codesearch.evaluate import KS, build_queries, evaluate_index, metrics
from codesearch.index import CodeIndex, dense_text, lexical_text
from codesearch.parser import parse_repository


@pytest.fixture(scope="module")
def eval_index(fixture_root):
    return CodeIndex(parse_repository(fixture_root), include_comments=False)


def test_queries_are_comment_function_pairs(eval_index):
    qs = build_queries(eval_index)
    pairs = {eval_index.units[q.target].id: q.text for q in qs}
    # sb_len's comment has enough words; functions without comments produce no query
    assert len(qs) == 13
    assert pairs["src/hash.c::free_chain"] == "Recursively free a chain of hash table entries."
    assert "src/hash.c::table_free" not in pairs and "src/main.c::log_msg" not in pairs


def test_comments_are_removed_from_the_evaluation_index(eval_index):
    for q in build_queries(eval_index):
        u = eval_index.units[q.target]
        assert q.text not in dense_text(u, eval_index.include_comments)
        assert u.comment_raw not in lexical_text(u, eval_index.include_comments)


def test_short_and_duplicate_comments_are_skipped():
    from codesearch.parser import parse_source

    src = b"""/* Unused. */
int a(void) { return 0; }
/* Same words describe two different functions. */
int b(void) { return 0; }
/* Same words describe two different functions. */
int c(void) { return 0; }
/* A unique description of this one function. */
int d(void) { return 0; }
"""
    units = parse_source(src, "x.c")
    for u in units:
        u.id = f"x.c::{u.name}"
    idx = CodeIndex(units, include_comments=False)
    qs = build_queries(idx)
    assert [(q.text, idx.units[q.target].name) for q in qs] == [("A unique description of this one function.", "d")]


def test_metrics_hand_computed():
    m = metrics([1, 2, 6, 20])
    assert m.recall == {1: 0.25, 5: 0.5, 10: 0.75}
    assert m.mrr == pytest.approx((1 + 1 / 2 + 1 / 6 + 1 / 20) / 4)
    assert metrics([]).mrr == 0.0


def test_evaluate_index_reports_all_methods(eval_index):
    qs = build_queries(eval_index)
    res = evaluate_index(eval_index, qs)
    assert set(res) == {"bm25", "dense", "hybrid", "hybrid_graph"}
    for r in res.values():
        assert 0.0 < r.mrr <= 1.0
        assert [r.recall[k] for k in KS] == sorted(r.recall[k] for k in KS)


def test_evaluate_index_matches_search_path(eval_index):
    qs = build_queries(eval_index)[:3]
    res = evaluate_index(eval_index, qs, methods=("hybrid_graph",))
    ranks = [eval_index.ranking(q.text, "hybrid_graph").index(q.target) + 1 for q in qs]
    assert res["hybrid_graph"].mrr == pytest.approx(metrics(ranks).mrr)
