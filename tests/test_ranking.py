import numpy as np
import pytest

from codesearch.ranking import graph_expand, rank_order, rrf


def test_rank_order_breaks_ties_by_index():
    assert rank_order(np.array([0.1, 0.5, 0.5, 0.0])) == [1, 2, 0, 3]
    assert rank_order(np.array([])) == []


def test_rrf_hand_computed():
    s = rrf([[2, 0, 1], [0, 1, 2]], n=3, k=60)
    assert s[0] == pytest.approx(1 / 62 + 1 / 61)
    assert s[1] == pytest.approx(1 / 63 + 1 / 62)
    assert s[2] == pytest.approx(1 / 61 + 1 / 63)
    assert rank_order(s) == [0, 2, 1]


def test_rrf_with_k_one_and_disjoint_lists():
    s = rrf([[0], [1, 2]], n=4, k=1)
    np.testing.assert_allclose(s, [1 / 2, 1 / 2, 1 / 3, 0.0])


def test_rrf_depth_truncates_each_ranking():
    s = rrf([[2, 0, 1], [0, 1, 2]], n=3, k=60, depth=1)
    np.testing.assert_allclose(s, [1 / 61, 0.0, 1 / 61])
    assert rank_order(s) == [0, 2, 1]


def test_weighted_rrf_hand_computed():
    s = rrf([[2, 0, 1], [0, 1, 2]], n=3, k=60, weights=[1.0, 5.0])
    assert s[0] == pytest.approx(1 / 62 + 5 / 61)
    assert s[1] == pytest.approx(1 / 63 + 5 / 62)
    assert s[2] == pytest.approx(1 / 61 + 5 / 63)
    assert rank_order(s) == [0, 1, 2]
    with pytest.raises(ValueError):
        rrf([[0]], n=1, weights=[1.0, 2.0])


def test_graph_expand_hand_computed():
    scores = np.array([0.5, 0.4, 0.1, 0.05, 0.0])
    nbrs = {0: [3], 1: [3, 4], 2: [], 3: [0, 1], 4: [1]}
    out = graph_expand(scores, nbrs.__getitem__, seeds=2, weight=0.5)
    # doc 3 gets 0.5*0.5/1 from seed 0 and 0.5*0.4/2 from seed 1; doc 4 gets 0.1 from seed 1
    assert out[3] == pytest.approx(0.05 + 0.25 + 0.1)
    assert out[4] == pytest.approx(0.1)
    assert out[2] == pytest.approx(0.1)
    # doc 3 ties seed 1, so seeds are lifted just enough to keep their place
    assert rank_order(out) == [0, 1, 3, 2, 4]
    np.testing.assert_array_equal(scores, [0.5, 0.4, 0.1, 0.05, 0.0])  # input untouched


def test_graph_expand_no_lift_needed_and_seed_neighbours_not_boosted():
    scores = np.array([1.0, 0.9, 0.0])
    out = graph_expand(scores, {0: [1, 2], 1: [0], 2: [0]}.__getitem__, seeds=2, weight=0.5)
    np.testing.assert_allclose(out, [1.0, 0.9, 0.25])


def test_graph_expand_ignores_zero_scores():
    scores = np.zeros(3)
    out = graph_expand(scores, lambda i: [0, 1, 2], seeds=3)
    np.testing.assert_array_equal(out, scores)
