import copy

import pytest

from codesearch.index import dense_text
from codesearch.mining import (
    BenchmarkGuard,
    add_negatives,
    check_disjoint_repos,
    code_shingles,
    find_leaks,
    load_pairs,
    mine_units,
    parse_training_repo,
    save_pairs,
    split_of,
    stats_table,
)
from codesearch.parser import parse_source
from codesearch.repos import PINNED_REPOS, TRAINING_REPOS, PinnedRepo

TRAIN_SRC = b"""/* Grow the output buffer so that it can hold n more bytes. */
static int buf_grow(struct buf *b, size_t n)
{
    size_t want = b->len + n;
    if (want <= b->cap) return 0;
    while (b->cap < want) b->cap *= 2;
    b->data = realloc(b->data, b->cap);
    return b->data == NULL;
}

/* Too short a body. */
int tiny(void) { return 1; }

/* Append one byte to the end of the output buffer. */
int buf_putc(struct buf *b, int c)
{
    if (buf_grow(b, 1)) return -1;
    b->data[b->len++] = (char)c;
    return 0;
}

/* Same words describe two different functions here. */
int twin_a(int x)
{
    int y = x + 1;
    return y * 2;
}

/* Same words describe two different functions here. */
int twin_b(int x)
{
    int y = x - 1;
    return y * 3;
}

/* Copy of the putc function under another file name. */
int buf_putc(struct buf *b, int c)
{
    if (buf_grow(b, 1)) return -1;
    b->data[b->len++] = (char)c;
    return 0;
}

int no_comment(int a, int b)
{
    int s = a + b;
    return s;
}
"""


def _units(src: bytes = TRAIN_SRC, path: str = "lib/buf.c"):
    units = parse_source(src, path)
    for i, u in enumerate(units):
        u.id = f"{path}::{u.name}@{i}"
    return units


def test_mining_keeps_commented_functions_and_reports_drops():
    pairs, pool, st = mine_units("t", _units(), guard=None, val_fraction=0.0)
    assert [p.unit_id.split("::")[1].split("@")[0] for p in pairs] == ["buf_grow", "buf_putc"]
    grow = pairs[0]
    assert grow.query == "Grow the output buffer so that it can hold n more bytes."
    assert grow.query not in grow.code and "realloc" in grow.code
    assert st.functions == 7 and st.kept == 2
    assert st.dropped == {"duplicate_code": 1, "too_short": 1, "duplicate_comment": 2}
    # every distinct function (with or without a comment) can serve as a negative
    assert len(pool) == 6
    assert "| t | 7 |" in stats_table([st])


def test_split_is_by_file_and_deterministic():
    assert split_of("redis", "src/a.c") == split_of("redis", "src/a.c")
    splits = {split_of("r", f"src/f{i}.c") for i in range(200)}
    assert splits == {"train", "val"}
    pairs, _, _ = mine_units("t", _units(), guard=None)
    assert len({p.split for p in pairs}) == 1  # one file -> one split


@pytest.mark.parametrize("mode", ["random", "hard"])
def test_negatives_come_from_other_functions(mode):
    pairs, pool, _ = mine_units("t", _units(), guard=None, val_fraction=0.0)
    with_neg = add_negatives(pairs, pool, mode, seed=1, val_fraction=0.0)
    assert len(with_neg) == len(pairs)
    for p in with_neg:
        assert p.negative and p.negative != p.code
    assert add_negatives(pairs, pool, mode, seed=1, val_fraction=0.0) == with_neg
    with pytest.raises(ValueError):
        add_negatives(pairs, pool, "easy")


def test_hard_negative_is_a_top_bm25_hit():
    pairs, pool, _ = mine_units("t", _units(), guard=None, val_fraction=0.0)
    putc = next(p for p in add_negatives(pairs, pool, "hard", depth=1, val_fraction=0.0) if "putc" in p.unit_id)
    # the best lexical match for "append one byte to the output buffer" other than buf_putc itself
    assert putc.negative.startswith("buf grow")


def test_pairs_round_trip(tmp_path):
    pairs, pool, _ = mine_units("t", _units(), guard=None)
    pairs = add_negatives(pairs, pool, "random")
    save_pairs(pairs, tmp_path / "p.jsonl")
    assert load_pairs(tmp_path / "p.jsonl") == pairs


def test_parse_training_repo_reads_only_listed_directories(tmp_path, fixture_root):
    import shutil

    shutil.copytree(fixture_root, tmp_path / "r")
    units = parse_training_repo(tmp_path / "r", PinnedRepo("r", "u", "c", "t", "l", ("src",)))
    assert units and all(u.file.startswith("src/") and u.id.startswith("src/") for u in units)
    assert not any(u.file.startswith("include/") for u in units)


# ---------------------------------------------------------------- leakage
def test_training_repositories_are_not_benchmark_repositories():
    check_disjoint_repos(TRAINING_REPOS)
    clash = PinnedRepo("lua-copy", PINNED_REPOS[0].url.upper(), "x", "t", "MIT")
    with pytest.raises(ValueError):
        check_disjoint_repos((clash,))
    with pytest.raises(ValueError):
        check_disjoint_repos((PinnedRepo("ZLIB", "https://example.org/z", "x", "t", "zlib"),))


def test_guard_drops_exact_and_near_duplicates_of_benchmark_functions(units):
    bench = [u for u in units if u.file == "src/hash.c"]
    resize = next(u for u in bench if u.name == "table_resize")
    renamed = copy.deepcopy(resize)
    renamed.body = renamed.body.replace("on_resize", "resized")  # a lightly edited copy
    comment_copy = copy.deepcopy(units[0])
    comment_copy.comment = resize.comment
    exact = copy.deepcopy(resize)
    exact.comment = "An entirely different description of the same code."
    candidates = [exact, renamed, comment_copy, *[u for u in units if u.file == "src/strbuf.c"]]
    for i, u in enumerate(candidates):
        u.id = f"train.c::{u.name}@{i}"
        u.file = "train.c"

    guard = BenchmarkGuard(bench)
    assert guard.code_reason(dense_text(exact, False)) == "benchmark_code_exact"
    assert guard.code_reason(dense_text(renamed, False)) == "benchmark_code_near"
    assert guard.comment_reason(resize.comment) == "benchmark_comment_exact"
    assert guard.comment_reason(resize.comment.upper() + " Done.") == "benchmark_comment_near"

    pairs, pool, st = mine_units("t", candidates, guard, val_fraction=0.0)
    assert st.dropped["benchmark_code_exact"] == 1 and st.dropped["benchmark_code_near"] == 1
    assert st.dropped["benchmark_comment_exact"] == 1
    assert pairs and all(p.unit_id.split("::")[1].startswith("sb_") for p in pairs)
    assert all(u.name != "table_resize" for u in pool)
    assert find_leaks(add_negatives(pairs, pool, "hard", val_fraction=0.0), bench) == []


def test_find_leaks_flags_a_benchmark_function_smuggled_in(units):
    pairs, pool, _ = mine_units("t", units, guard=None)
    leaks = find_leaks(pairs, units)
    assert len(leaks) == len(pairs) and {r for _, r in leaks} == {"benchmark_code_exact"}


def test_unrelated_code_shares_few_shingles():
    a = code_shingles("int f(int x) { return x + 1; }")
    b = code_shingles("void g(char *s) { while (*s) putchar(*s++); }")
    assert len(a & b) / len(a | b) < 0.1
