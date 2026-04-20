"""Mine (leading comment, function body) training pairs from C repositories, free of benchmark leakage.

A pair is the benchmark-style query built from a function's leading comment and the text the
dense index embeds for that function with its comment removed (``dense_text(unit, False)``).
Training repositories must be different projects from the benchmark ones, and every pair whose
code or comment is an exact or near duplicate of any benchmark function is dropped, so a
fine-tuned model cannot have seen a benchmark answer.

Negatives for MultipleNegativesRankingLoss are drawn from the same repository and split:
either uniformly at random or among the top BM25 hits for the query (hard negatives).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .bm25 import BM25
from .evaluate import query_text
from .index import dense_text, lexical_text
from .parser import FunctionUnit, parse_repository
from .ranking import rank_order
from .repos import PINNED_REPOS, PinnedRepo
from .tokenize import tokenize

SHINGLE = 5  # tokens per code shingle
COMMENT_SHINGLE = 3  # words per comment shingle
CODE_JACCARD = 0.5  # code at or above this shingle Jaccard with a benchmark function is a near duplicate
COMMENT_JACCARD = 0.6
MIN_BODY_LINES = 3  # one- and two-line functions carry too little signal to learn from
VAL_FRACTION = 0.1
HARD_NEGATIVE_DEPTH = 10  # hard negatives are sampled from this many top BM25 hits

_CODE_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|\S")


@dataclass
class TrainingPair:
    repo: str
    unit_id: str
    query: str
    code: str
    split: str = "train"  # "train" or "val"
    negative: str = ""


@dataclass
class MiningStats:
    repo: str
    functions: int = 0
    with_query: int = 0
    dropped: Counter = field(default_factory=Counter)
    kept: int = 0


# ---------------------------------------------------------------- similarity
def code_body(code: str) -> str:
    """Drop the "name (path)" header that dense_text puts in front, so copies under another path still match."""
    head, sep, body = code.partition("\n")
    return body if sep and head.endswith(")") else code


def normalise_code(code: str) -> str:
    return " ".join(_CODE_TOKEN.findall(code_body(code)))


def code_hash(code: str) -> str:
    return hashlib.sha1(normalise_code(code).encode("utf-8", "replace")).hexdigest()


def code_shingles(code: str, k: int = SHINGLE) -> set[str]:
    toks = _CODE_TOKEN.findall(code_body(code))
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def comment_shingles(text: str, k: int = COMMENT_SHINGLE) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class ShingleIndex:
    """Inverted index over shingle sets that finds the highest Jaccard similarity to any stored set."""

    def __init__(self, sets: list[set[str]]):
        self.sizes = [len(s) for s in sets]
        self.postings: dict[str, list[int]] = {}
        for i, s in enumerate(sets):
            for sh in s:
                self.postings.setdefault(sh, []).append(i)

    def max_jaccard(self, s: set[str]) -> tuple[float, int]:
        overlap: Counter = Counter()
        for sh in s:
            overlap.update(self.postings.get(sh, ()))
        best, best_i = 0.0, -1
        for i, inter in overlap.items():
            j = inter / (len(s) + self.sizes[i] - inter)
            if j > best:
                best, best_i = j, i
        return best, best_i


class BenchmarkGuard:
    """Everything known about the benchmark functions that training data must not overlap with."""

    def __init__(self, bench_units: list[FunctionUnit]):
        self.units = bench_units
        bodies = [dense_text(u, False) for u in bench_units]
        self.code_hashes = {code_hash(b) for b in bodies}
        self.code_index = ShingleIndex([code_shingles(b) for b in bodies])
        comments = [u.comment for u in bench_units if u.comment]
        self.comment_texts = {" ".join(c.lower().split()) for c in comments}
        self.comment_index = ShingleIndex([comment_shingles(c) for c in comments])

    def code_reason(self, code: str) -> str | None:
        if code_hash(code) in self.code_hashes:
            return "benchmark_code_exact"
        if self.code_index.max_jaccard(code_shingles(code))[0] >= CODE_JACCARD:
            return "benchmark_code_near"
        return None

    def comment_reason(self, comment: str) -> str | None:
        if " ".join(comment.lower().split()) in self.comment_texts:
            return "benchmark_comment_exact"
        if self.comment_index.max_jaccard(comment_shingles(comment))[0] >= COMMENT_JACCARD:
            return "benchmark_comment_near"
        return None


def check_disjoint_repos(training: tuple[PinnedRepo, ...], benchmark: tuple[PinnedRepo, ...] = PINNED_REPOS) -> None:
    """Raise if a training repository is one of the benchmark repositories."""
    bench_keys = {r.name.lower() for r in benchmark} | {r.url.lower().removesuffix(".git") for r in benchmark}
    for r in training:
        if r.name.lower() in bench_keys or r.url.lower().removesuffix(".git") in bench_keys:
            raise ValueError(f"training repository {r.name} is also a benchmark repository")


def find_leaks(pairs: list[TrainingPair], bench_units: list[FunctionUnit]) -> list[tuple[str, str]]:
    """Independent leakage audit: (unit_id, reason) for every pair that overlaps the benchmark."""
    guard = BenchmarkGuard(bench_units)
    leaks = []
    for p in pairs:
        reason = guard.code_reason(p.code) or guard.comment_reason(p.query)
        if reason is None and p.negative:
            reason = guard.code_reason(p.negative)
        if reason:
            leaks.append((p.unit_id, reason))
    return leaks


# ---------------------------------------------------------------- mining
def parse_training_repo(root: Path, repo: PinnedRepo | None = None) -> list[FunctionUnit]:
    """Parse a training repository, restricted to repo.paths when given; ids keep the full relative path."""
    if repo is None or not repo.paths:
        return parse_repository(root)
    units = []
    for sub in repo.paths:
        for u in parse_repository(root / sub):
            u.file = f"{sub}/{u.file}"
            u.id = f"{sub}/{u.id}"
            units.append(u)
    return units


def split_of(repo: str, file: str, val_fraction: float = VAL_FRACTION) -> str:
    """Deterministic split by source file, so no file contributes to both train and validation."""
    h = int(hashlib.sha1(f"{repo}/{file}".encode()).hexdigest()[:8], 16)
    return "val" if h / 0xFFFFFFFF < val_fraction else "train"


def mine_units(repo: str, units: list[FunctionUnit], guard: BenchmarkGuard | None,
               val_fraction: float = VAL_FRACTION) -> tuple[list[TrainingPair], list[FunctionUnit], MiningStats]:
    """Return the pairs of one repository, the leak-free negative pool and the drop counts."""
    st = MiningStats(repo, functions=len(units))
    pool: list[FunctionUnit] = []
    seen_code: set[str] = set()
    candidates: list[tuple[str, FunctionUnit, str]] = []
    for u in units:
        code = dense_text(u, False)
        h = code_hash(code)
        if h in seen_code:
            st.dropped["duplicate_code"] += 1
            continue
        seen_code.add(h)
        reason = guard.code_reason(code) if guard else None
        if reason:
            st.dropped[reason] += 1
            continue
        pool.append(u)
        q = query_text(u.comment)
        if q is None:
            continue
        st.with_query += 1
        if u.end_line - u.start_line + 1 < MIN_BODY_LINES:
            st.dropped["too_short"] += 1
            continue
        reason = guard.comment_reason(u.comment) if guard else None
        if reason:
            st.dropped[reason] += 1
            continue
        candidates.append((q, u, code))
    by_query = Counter(q for q, _, _ in candidates)
    pairs = []
    for q, u, code in candidates:
        if by_query[q] > 1:  # the same comment on several functions cannot name one target
            st.dropped["duplicate_comment"] += 1
            continue
        pairs.append(TrainingPair(repo, u.id, q, code, split_of(repo, u.file, val_fraction)))
    st.kept = len(pairs)
    return pairs, pool, st


def add_negatives(pairs: list[TrainingPair], pool: list[FunctionUnit], mode: str, seed: int = 0,
                  depth: int = HARD_NEGATIVE_DEPTH, val_fraction: float = VAL_FRACTION) -> list[TrainingPair]:
    """Copies of `pairs` with a negative from `pool` (same split): mode is "random" or "hard" (BM25)."""
    if mode not in ("random", "hard"):
        raise ValueError(f"unknown negative mode {mode!r}")
    rng = random.Random(seed)
    out = []
    for split in ("train", "val"):
        members = [p for p in pairs if p.split == split]
        cand = [u for u in pool if split_of(members[0].repo, u.file, val_fraction) == split] if members else []
        if len(cand) < 2:
            out.extend(members)
            continue
        cand_hash = [code_hash(dense_text(u, False)) for u in cand]
        bm25 = BM25([tokenize(lexical_text(u, False)) for u in cand]) if mode == "hard" else None
        for p in members:
            own = code_hash(p.code)
            ok = [i for i in range(len(cand)) if cand_hash[i] != own and cand[i].id != p.unit_id]
            if mode == "random":
                choice = rng.choice(ok)
            else:
                okset = set(ok)
                top = [i for i in rank_order(bm25.scores(tokenize(p.query))) if i in okset][:depth]
                choice = rng.choice(top)
            out.append(TrainingPair(**{**asdict(p), "negative": dense_text(cand[choice], False)}))
    return out


def mine_repositories(repos: dict[str, tuple[Path, PinnedRepo | None]], bench_units: list[FunctionUnit],
                      seed: int = 0, val_fraction: float = VAL_FRACTION) -> tuple[dict[str, list[TrainingPair]],
                                                                                 list[MiningStats]]:
    """Mine every repository and return {"random": pairs, "hard": pairs} plus per-repository stats."""
    guard = BenchmarkGuard(bench_units)
    out: dict[str, list[TrainingPair]] = {"random": [], "hard": []}
    stats = []
    for name, (root, pinned) in repos.items():
        pairs, pool, st = mine_units(name, parse_training_repo(root, pinned), guard, val_fraction)
        stats.append(st)
        for mode in out:
            out[mode].extend(add_negatives(pairs, pool, mode, seed, val_fraction=val_fraction))
    return out, stats


def save_pairs(pairs: list[TrainingPair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p)) + "\n")


def load_pairs(path: str | Path) -> list[TrainingPair]:
    with Path(path).open() as f:
        return [TrainingPair(**json.loads(line)) for line in f if line.strip()]


def stats_table(stats: list[MiningStats]) -> str:
    reasons = sorted({r for s in stats for r in s.dropped})
    head = "| Repository | Functions | With usable comment | " + " | ".join(reasons) + " | Pairs kept |"
    lines = [head, "|---|" + "---:|" * (len(reasons) + 3)]
    for s in stats:
        lines.append(f"| {s.repo} | {s.functions} | {s.with_query} | "
                     + " | ".join(str(s.dropped.get(r, 0)) for r in reasons) + f" | {s.kept} |")
    return "\n".join(lines)
