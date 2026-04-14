"""Comment-to-function retrieval benchmark.

Each function with a descriptive leading comment yields one query: the comment text is the
query and the function is the single relevant answer. The index is built with every leading
comment removed, so a method can only succeed by matching the comment's intent to the code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .index import METHODS, CodeIndex
from .parser import parse_repository
from .ranking import rank_order
from .tokenize import tokenize

KS = (1, 5, 10)
MIN_QUERY_WORDS = 4
MAX_QUERY_WORDS = 64


@dataclass
class Query:
    text: str
    target: int  # index into CodeIndex.units


@dataclass
class MethodResult:
    recall: dict[int, float]
    mrr: float


@dataclass
class RepoResult:
    name: str
    n_functions: int
    n_queries: int
    methods: dict[str, MethodResult] = field(default_factory=dict)
    seconds: float = 0.0
    source: str = ""


def build_queries(index: CodeIndex) -> list[Query]:
    """One query per function whose cleaned leading comment is a usable description.

    Comments shorter than MIN_QUERY_WORDS words are skipped (mostly license banners
    and section markers are filtered by the parser; very short ones such as "unused"
    carry no intent), long ones are cut to MAX_QUERY_WORDS words. When two functions
    share the same comment (e.g. #ifdef alternatives) the comment cannot identify a
    single target, so those queries are dropped.
    """
    by_text: dict[str, list[int]] = {}
    for i, u in enumerate(index.units):
        text = query_text(u.comment)
        if text:
            by_text.setdefault(text, []).append(i)
    return [Query(text, ids[0]) for text, ids in by_text.items() if len(ids) == 1]


def query_text(comment: str) -> str | None:
    """The query a cleaned leading comment yields, or None when it is not a usable description."""
    words = comment.split()
    if len(words) < MIN_QUERY_WORDS or not tokenize(comment):
        return None
    if "copyright" in comment.lower() or "license" in comment.lower():
        return None
    return " ".join(words[:MAX_QUERY_WORDS])


def metrics(ranks: list[int]) -> MethodResult:
    """ranks are 1-based positions of the target in each query's ranking."""
    r = np.asarray(ranks, dtype=np.float64)
    if r.size == 0:
        return MethodResult({k: 0.0 for k in KS}, 0.0)
    return MethodResult({k: float((r <= k).mean()) for k in KS}, float((1.0 / r).mean()))


def query_ranks(index: CodeIndex, queries: list[Query], methods=METHODS) -> dict[str, list[int]]:
    """1-based rank of each query's target under every method."""
    q_emb = index.embedder.encode_queries([q.text for q in queries])
    ranks: dict[str, list[int]] = {m: [] for m in methods}
    for qi, q in enumerate(queries):
        bm = index.bm25.scores(tokenize(q.text))
        dn = index.embeddings @ q_emb[qi]
        for m in methods:
            order = rank_order(index.combine(m, bm, dn))
            ranks[m].append(order.index(q.target) + 1)
    return ranks


def evaluate_index(index: CodeIndex, queries: list[Query], methods=METHODS) -> dict[str, MethodResult]:
    ranks = query_ranks(index, queries, methods)
    return {m: metrics(ranks[m]) for m in methods}


def evaluate_repo(path: str | Path, name: str | None = None, model_name: str | None = None) -> RepoResult:
    t0 = time.time()
    kwargs = {"model_name": model_name} if model_name else {}
    index = CodeIndex(parse_repository(path), include_comments=False, root=str(path), **kwargs)
    queries = build_queries(index)
    res = RepoResult(name or Path(path).name, len(index.units), len(queries))
    res.methods = evaluate_index(index, queries)
    res.seconds = time.time() - t0
    return res


METHOD_LABELS = {
    "bm25": "BM25 (baseline)",
    "dense": "Dense",
    "hybrid": "Hybrid (RRF)",
    "hybrid_graph": "Hybrid + call graph",
}


def markdown_table(results: list[RepoResult]) -> str:
    lines = [
        "| Repository | Method | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        for m, mr in r.methods.items():
            lines.append(
                f"| {r.name} | {METHOD_LABELS.get(m, m)} | {mr.recall[1]:.3f} | {mr.recall[5]:.3f} | "
                f"{mr.recall[10]:.3f} | {mr.mrr:.3f} |"
            )
    return "\n".join(lines)


def corpus_table(results: list[RepoResult]) -> str:
    lines = ["| Repository | Functions indexed | Queries | Time (s) |", "|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r.name} | {r.n_functions} | {r.n_queries} | {r.seconds:.1f} |")
    return "\n".join(lines)


def macro_average(results: list[RepoResult]) -> dict[str, MethodResult]:
    out = {}
    for m in METHODS:
        rs = [r.methods[m] for r in results if m in r.methods]
        if rs:
            out[m] = MethodResult({k: float(np.mean([x.recall[k] for x in rs])) for k in KS},
                                  float(np.mean([x.mrr for x in rs])))
    return out
