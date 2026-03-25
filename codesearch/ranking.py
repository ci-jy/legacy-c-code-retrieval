"""Rank helpers: score-to-ranking, reciprocal-rank fusion and call-graph expansion."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

RRF_K = 60


def rank_order(scores: np.ndarray) -> list[int]:
    """Indices sorted by descending score; ties broken by ascending index (deterministic)."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(len(scores)), -scores)).tolist()


def rrf(rankings: Sequence[Sequence[int]], n: int, k: int = RRF_K, depth: int | None = None) -> np.ndarray:
    """Reciprocal-rank fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1.

    `depth` truncates each ranking before fusing; documents absent from a ranking get nothing from it.
    """
    out = np.zeros(n, dtype=np.float64)
    for ranking in rankings:
        for r, doc in enumerate(ranking[:depth] if depth else ranking, start=1):
            out[doc] += 1.0 / (k + r)
    return out


def graph_expand(
    scores: np.ndarray,
    neighbours: Callable[[int], Sequence[int]],
    seeds: int = 3,
    weight: float = 0.5,
) -> np.ndarray:
    """Promote call-graph neighbours of the strongest hits.

    The top `seeds` documents ("seeds") keep their order and stay on top. Every other
    document that is a caller or callee of a seed gains `weight * score(seed) / degree(seed)`,
    dividing by degree so that hub functions do not flood the ranking. If boosted documents
    would overtake a seed, the seeds are lifted by a constant so the seed order is preserved.
    Only one hop is used.
    """
    base = np.asarray(scores, dtype=np.float64)
    out = base.copy()
    top = [s for s in rank_order(base)[:seeds] if base[s] > 0]
    if not top:
        return out
    top_set = set(top)
    for seed in top:
        nbrs = [x for x in neighbours(seed) if x != seed]
        if not nbrs:
            continue
        share = weight * base[seed] / len(nbrs)
        for nb in nbrs:
            if nb not in top_set:
                out[nb] += share
    mask = np.ones(len(base), dtype=bool)
    mask[top] = False
    if mask.any():
        highest_rest = out[mask].max()
        lowest_seed = min(base[s] for s in top)
        if highest_rest >= lowest_seed:
            out[top] += highest_rest - lowest_seed + 1e-9
    return out
