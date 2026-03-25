"""Markdown report for benchmark runs."""

from __future__ import annotations

import platform

from .evaluate import (
    KS,
    MAX_QUERY_WORDS,
    METHOD_LABELS,
    MIN_QUERY_WORDS,
    RepoResult,
    corpus_table,
    macro_average,
    markdown_table,
)
from .index import FUSION_DEPTH
from .ranking import RRF_K


def write_report(results: list[RepoResult], model: str) -> str:
    out = ["# Retrieval benchmark results", ""]
    out += [
        "Task: each function's leading comment is used as a natural-language query; the function it "
        "documents is the only correct answer. All leading comments are removed from the index first, "
        f"so a method has to connect the description to code. Comments with fewer than {MIN_QUERY_WORDS} "
        f"words, licence banners and comments shared by several functions are skipped; queries are cut to "
        f"{MAX_QUERY_WORDS} words. Ranks are over every function in the repository.",
        "",
        f"- Embedding model: `{model}` (CPU)",
        f"- Hybrid: reciprocal-rank fusion (k={RRF_K}) of the top {FUSION_DEPTH} BM25 and dense results",
        "- Hybrid + call graph: hybrid, then callers/callees of the top 3 hits are promoted (top 3 kept in place)",
        f"- Python {platform.python_version()}",
        "",
        "## Corpora",
        "",
    ]
    out.append("| Repository | Source |")
    out.append("|---|---|")
    for r in results:
        out.append(f"| {r.name} | {r.source or '-'} |")
    out += ["", corpus_table(results), "", "## Results", "", markdown_table(results), ""]
    if len(results) > 1:
        avg = macro_average(results)
        out += ["## Macro average over repositories", "", "| Method | " + " | ".join(f"Recall@{k}" for k in KS)
                + " | MRR |", "|---|" + "---:|" * (len(KS) + 1)]
        for m, mr in avg.items():
            out.append(f"| {METHOD_LABELS[m]} | " + " | ".join(f"{mr.recall[k]:.3f}" for k in KS) + f" | {mr.mrr:.3f} |")
        out.append("")
    out += ["## Hybrid vs. BM25 (MRR)", ""]
    for r in results:
        b, h, g = r.methods["bm25"].mrr, r.methods["hybrid"].mrr, r.methods["hybrid_graph"].mrr
        out.append(f"- {r.name}: BM25 {b:.3f}, hybrid {h:.3f} ({h - b:+.3f}), hybrid + graph {g:.3f} ({g - b:+.3f})")
    out.append("")
    return "\n".join(out)
