import shutil

from codesearch.cli import main


def test_bench_on_fixture_writes_markdown(tmp_path, fixture_root):
    out = tmp_path / "bench.md"
    assert main(["bench", "--fixture", str(fixture_root), "--out", str(out)]) == 0
    text = out.read_text()
    assert text.startswith("# Retrieval benchmark results")
    assert "| Repository | Method | Recall@1 | Recall@5 | Recall@10 | MRR |" in text
    for label in ("BM25 (baseline)", "Dense", "Hybrid (RRF)", "Hybrid + call graph"):
        assert f"| mini_c | {label} |" in text
    assert "| mini_c | 16 | 13 |" in text


def test_index_then_query(tmp_path, fixture_root, capsys):
    repo = tmp_path / "repo"
    shutil.copytree(fixture_root, repo)
    assert main(["index", str(repo)]) == 0
    assert "indexed 16 functions, 19 call edges" in capsys.readouterr().out
    assert (repo / ".codesearch" / "units.json").exists()
    assert main(["query", str(repo), "double the bucket count and rehash", "-k", "3", "--show"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[2] == "table_resize"
    assert "src/hash.c:49-71" in out[0]


def test_bench_rejects_missing_directory(tmp_path):
    assert main(["bench", "--fixture", str(tmp_path / "nope"), "--out", str(tmp_path / "x.md")]) == 2
