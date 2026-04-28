"""Embedding-model ablation on the benchmark with paired bootstrap confidence intervals.

Every model is evaluated on the same queries of the pinned benchmark repositories, dense-only and
inside the hybrid fusion. Differences to the reference model (the first one) are reported per
query, so the bootstrap is paired: each resample draws queries with replacement *within each
repository* and recomputes the macro-averaged difference, which keeps the interval consistent
with the macro averages in the table.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .evaluate import METHOD_LABELS, build_queries, query_ranks
from .index import CodeIndex
from .parser import parse_repository

ABLATION_METHODS = ("dense", "hybrid", "hybrid_graph")
CONFIDENCE = 0.95


def reciprocal_ranks(ranks) -> np.ndarray:
    return 1.0 / np.asarray(ranks, dtype=np.float64)


def recall_at(ranks, k: int = 10) -> np.ndarray:
    return (np.asarray(ranks) <= k).astype(np.float64)


def paired_bootstrap(a: list[np.ndarray], b: list[np.ndarray], resamples: int = 10000, seed: int = 0,
                     confidence: float = CONFIDENCE) -> dict:
    """Bootstrap the macro-averaged difference mean(b) - mean(a) over groups of paired per-query values.

    a[g] and b[g] hold one value per query of group g (a repository) for two systems. Returns the
    observed difference, the percentile interval and the share of resamples with a difference <= 0.
    """
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("paired bootstrap needs the same queries for both systems")
    rng = np.random.default_rng(seed)
    diffs = [np.asarray(y, dtype=np.float64) - np.asarray(x, dtype=np.float64) for x, y in zip(a, b)]
    observed = float(np.mean([d.mean() for d in diffs]))
    boot = np.zeros(resamples)
    for d in diffs:
        idx = rng.integers(0, len(d), size=(resamples, len(d)))
        boot += d[idx].mean(axis=1)
    boot /= len(diffs)
    alpha = (1.0 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return {"diff": observed, "lo": float(lo), "hi": float(hi), "p_le_0": float((boot <= 0).mean())}


def collect_ranks(model: str, repos: dict[str, Path], methods=ABLATION_METHODS, log=print) -> dict:
    """{repo: {method: [rank per query]}} for one embedding model (queries in a fixed order)."""
    out = {}
    for name, path in repos.items():
        t0 = time.time()
        index = CodeIndex(parse_repository(path), include_comments=False, root=str(path), model_name=model)
        queries = build_queries(index)
        out[name] = query_ranks(index, queries, methods)
        log(f"[ablation] {model} on {name}: {len(queries)} queries ({time.time() - t0:.0f}s)")
    return out


def _fmt_ci(r: dict) -> str:
    return f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def ablation_report(ranks: dict[str, dict], labels: dict[str, str] | None = None, resamples: int = 10000,
                    seed: int = 0) -> str:
    """Markdown tables from {model: {repo: {method: ranks}}}; the first model is the reference."""
    labels = labels or {}
    models = list(ranks)
    ref = models[0]
    repos = list(ranks[ref])
    methods = [m for m in ABLATION_METHODS if m in ranks[ref][repos[0]]]
    n = sum(len(ranks[ref][r][methods[0]]) for r in repos)
    lab = lambda m: labels.get(m, m)  # noqa: E731

    lines = [f"Macro average over {', '.join(repos)} ({n} queries). Intervals are {CONFIDENCE:.0%} paired "
             f"bootstrap intervals ({resamples} resamples, stratified by repository) of the difference to "
             f"`{lab(ref)}`; P(Δ≤0) is the share of resamples in which the model did not improve.", "",
             "| Model | Method | Recall@1 | Recall@10 | MRR | ΔMRR [95% CI] | ΔRecall@10 [95% CI] | P(ΔMRR≤0) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in models:
        for m in methods:
            per = [ranks[model][r][m] for r in repos]
            r1 = np.mean([recall_at(x, 1).mean() for x in per])
            r10 = np.mean([recall_at(x, 10).mean() for x in per])
            mrr = np.mean([reciprocal_ranks(x).mean() for x in per])
            if model == ref:
                delta = ["—", "—", "—"]
            else:
                base = [ranks[ref][r][m] for r in repos]
                d_mrr = paired_bootstrap([reciprocal_ranks(x) for x in base], [reciprocal_ranks(x) for x in per],
                                         resamples, seed)
                d_r10 = paired_bootstrap([recall_at(x) for x in base], [recall_at(x) for x in per], resamples, seed)
                delta = [_fmt_ci(d_mrr), _fmt_ci(d_r10), f"{d_mrr['p_le_0']:.3f}"]
            lines.append(f"| {lab(model)} | {METHOD_LABELS[m]} | {r1:.3f} | {r10:.3f} | {mrr:.3f} | "
                         + " | ".join(delta) + " |")

    lines += ["", "Per repository (MRR):", "",
              "| Model | Method | " + " | ".join(repos) + " |", "|---|---|" + "---:|" * len(repos)]
    for model in models:
        for m in methods:
            lines.append(f"| {lab(model)} | {METHOD_LABELS[m]} | "
                         + " | ".join(f"{reciprocal_ranks(ranks[model][r][m]).mean():.3f}" for r in repos) + " |")
    return "\n".join(lines)


def run_ablation(models: list[str], data_dir: str | None = None, tracking_dir: str | None = None,
                 resamples: int = 10000, seed: int = 0, ranks_out: str | Path | None = None, log=print) -> str:
    from .repos import PINNED_REPOS, fetch

    kw = {"data_dir": Path(data_dir)} if data_dir else {}
    repos = {r.name: fetch(r, **kw) for r in PINNED_REPOS}
    ranks, labels, resolved = {}, {}, {}
    for spec in models:
        path = spec
        if spec.startswith("mlflow:"):
            from .finetune import resolve_model

            path = resolve_model(spec, tracking_dir)
            run_id = Path(path).parent.parent.name
            labels[spec] = f"{spec} ({run_id[:8]})"
        resolved[spec] = os.path.relpath(path) if os.path.isabs(path) else path
        ranks[spec] = collect_ranks(path, repos, log=log)
    if ranks_out:
        Path(ranks_out).write_text(json.dumps({"models": resolved, "ranks": ranks}))
    return ablation_report(ranks, labels, resamples, seed)
