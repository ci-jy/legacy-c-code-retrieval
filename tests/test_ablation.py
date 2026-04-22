import numpy as np
import pytest

from codesearch.ablation import ablation_report, paired_bootstrap, recall_at, reciprocal_ranks


def test_identical_systems_have_a_zero_interval():
    a = [np.array([1.0, 0.5, 0.2]), np.array([0.1, 1.0])]
    r = paired_bootstrap(a, a, resamples=200)
    assert r == {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_le_0": 1.0}


def test_constant_improvement_is_recovered_exactly():
    a = [np.array([0.2, 0.4, 0.6]), np.array([0.0, 1.0])]
    b = [x + 0.1 for x in a]
    r = paired_bootstrap(a, b, resamples=500)
    assert r["diff"] == pytest.approx(0.1) and r["lo"] == pytest.approx(0.1) and r["hi"] == pytest.approx(0.1)
    assert r["p_le_0"] == 0.0


def test_difference_is_the_macro_average_over_groups():
    a = [np.zeros(4), np.zeros(100)]
    b = [np.ones(4), np.zeros(100)]  # +1 on the small group only
    r = paired_bootstrap(a, b, resamples=1000, seed=3)
    assert r["diff"] == pytest.approx(0.5)  # not the pooled 4/104
    assert r["lo"] == pytest.approx(0.5) and r["hi"] == pytest.approx(0.5)


def test_interval_covers_the_observed_difference_and_is_seeded():
    rng = np.random.default_rng(0)
    a = [rng.random(300)]
    b = [a[0] + rng.normal(0.05, 0.2, 300)]
    r = paired_bootstrap(a, b, resamples=2000, seed=7)
    assert r["lo"] < r["diff"] < r["hi"]
    assert r["hi"] - r["lo"] == pytest.approx(2 * 1.96 * 0.2 / np.sqrt(300), rel=0.15)
    assert paired_bootstrap(a, b, resamples=2000, seed=7) == r
    with pytest.raises(ValueError):
        paired_bootstrap(a, [b[0][:-1]])


def test_rank_transforms():
    assert reciprocal_ranks([1, 4]).tolist() == [1.0, 0.25]
    assert recall_at([1, 10, 11]).tolist() == [1.0, 1.0, 0.0]
    assert recall_at([1, 2], k=1).tolist() == [1.0, 0.0]


def test_report_lists_every_model_and_method():
    ranks = {
        "base": {"r1": {"dense": [1, 2, 20], "hybrid": [1, 1, 5]}, "r2": {"dense": [3], "hybrid": [2]}},
        "tuned": {"r1": {"dense": [1, 1, 20], "hybrid": [1, 1, 5]}, "r2": {"dense": [1], "hybrid": [2]}},
    }
    text = ablation_report(ranks, {"tuned": "tuned (abc)"}, resamples=100)
    assert "over r1, r2 (4 queries)" in text
    # base dense: r1 R@1 1/3, MRR (1 + 0.5 + 0.05)/3; r2 R@1 0, MRR 1/3
    assert "| base | Dense | 0.167 | 0.833 | 0.425 | — | — | — |" in text
    # tuned dense: r1 MRR (1 + 1 + 0.05)/3, r2 MRR 1 -> macro diff (0.5/3 + 2/3)/2
    assert "| tuned (abc) | Dense | 0.833 | 0.833 | 0.842 | +0.417 [" in text
    assert "| tuned (abc) | Hybrid (RRF) | 0.333 | 1.000 | 0.617 | +0.000 [+0.000, +0.000] | +0.000" in text
    assert "| tuned (abc) | Dense | 0.683 | 1.000 |" in text  # per-repository MRR
