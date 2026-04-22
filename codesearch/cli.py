"""Command-line interface: index, query, bench, fetch, serve, mine, train and ablation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .dense import DEFAULT_MODEL  # noqa: E402
from .index import METHODS, CodeIndex  # noqa: E402


def _cmd_index(args) -> int:
    idx = CodeIndex.build(args.repo, model_name=args.model)
    out = Path(args.index_dir) if args.index_dir else Path(args.repo) / ".codesearch"
    idx.save(out)
    print(f"indexed {len(idx.units)} functions, {len(idx.graph.edges())} call edges -> {out}")
    return 0


def _load(repo: str, index_dir: str | None, model: str) -> CodeIndex:
    if index_dir:
        return CodeIndex.load(index_dir)
    return CodeIndex.open(repo, model_name=model)


def _cmd_query(args) -> int:
    idx = _load(args.repo, args.index_dir, args.model)
    for rank, (u, score) in enumerate(idx.search(args.query, k=args.k, method=args.method), start=1):
        print(f"{rank:2d}. {score:.4f}  {u.name}  {u.file}:{u.start_line}-{u.end_line}")
        if args.show:
            print("    " + u.signature)
            if u.comment:
                print("    // " + u.comment[:160])
    return 0


def _cmd_bench(args) -> int:
    from .evaluate import evaluate_repo
    from .report import write_report

    targets: list[tuple[str, Path, str]] = []
    if args.fixture:
        targets.append((Path(args.fixture).name, Path(args.fixture), "local fixture"))
    for r in args.repo or []:
        targets.append((Path(r).name, Path(r), "local path"))
    if not targets:
        from .repos import PINNED_REPOS, fetch

        for pinned in PINNED_REPOS:
            path = fetch(pinned, Path(args.data_dir)) if args.data_dir else fetch(pinned)
            targets.append((pinned.name, path, f"{pinned.url} @ {pinned.commit[:12]} ({pinned.tag}, {pinned.license})"))
    results = []
    for name, path, source in targets:
        if not path.is_dir():
            print(f"error: {path} is not a directory", file=sys.stderr)
            return 2
        print(f"[bench] {name}: {path}", file=sys.stderr)
        res = evaluate_repo(path, name=name, model_name=args.model)
        res.source = source
        results.append(res)
    ablations = {}
    for extra in args.ablation_model or []:
        print(f"[bench] ablation with {extra}", file=sys.stderr)
        ablations[extra] = [evaluate_repo(path, name=name, model_name=extra) for name, path, _ in targets]
    text = write_report(results, args.model, ablations)
    out = Path(args.out)
    out.write_text(text)
    print(text)
    print(f"wrote {out}", file=sys.stderr)
    return 0


def _cmd_fetch(args) -> int:
    from .repos import PINNED_REPOS, fetch

    for pinned in PINNED_REPOS:
        path = fetch(pinned, Path(args.data_dir)) if args.data_dir else fetch(pinned)
        print(f"{pinned.name:6s} {pinned.tag:10s} {pinned.commit}  {path}")
    return 0


def _training_sources(data_dir: str | None) -> tuple[dict, list]:
    """Fetch the training and benchmark repositories; return the training sources and benchmark units."""
    from .mining import check_disjoint_repos
    from .parser import parse_repository
    from .repos import PINNED_REPOS, TRAINING_REPOS, fetch

    check_disjoint_repos(TRAINING_REPOS)
    kw = {"data_dir": Path(data_dir)} if data_dir else {}
    sources = {r.name: (fetch(r, **kw), r) for r in TRAINING_REPOS}
    bench_units = [u for r in PINNED_REPOS for u in parse_repository(fetch(r, **kw))]
    return sources, bench_units


def _cmd_mine(args) -> int:
    from .mining import find_leaks, mine_repositories, save_pairs, stats_table

    sources, bench_units = _training_sources(args.data_dir)
    print(f"[mine] guarding against {len(bench_units)} benchmark functions", file=sys.stderr)
    pairs, stats = mine_repositories(sources, bench_units, seed=args.seed)
    out = Path(args.out_dir)
    for mode, ps in pairs.items():
        leaks = find_leaks(ps, bench_units)
        if leaks:
            print(f"error: {len(leaks)} {mode} pairs overlap the benchmark, e.g. {leaks[0]}", file=sys.stderr)
            return 1
        save_pairs(ps, out / f"pairs_{mode}.jsonl")
    table = stats_table(stats)
    (out / "stats.md").write_text(table + "\n")
    print(table)
    n = pairs["hard"]
    print(f"[mine] {sum(p.split == 'train' for p in n)} train / {sum(p.split == 'val' for p in n)} val pairs, "
          f"no benchmark overlap -> {out}", file=sys.stderr)
    return 0


def _cmd_train(args) -> int:
    from .finetune import load_config, train
    from .mining import load_pairs

    cfg = load_config(args.config, args.run)
    path = Path(args.pairs_dir) / f"pairs_{'random' if cfg.negatives == 'random' else 'hard'}.jsonl"
    pairs = load_pairs(path)
    run_id = train(cfg, [p for p in pairs if p.split == "train"], [p for p in pairs if p.split == "val"],
                   tracking_dir=args.tracking_dir, run_name=args.run,
                   data_info={"pairs_file": str(path), "config": str(args.config)},
                   log=lambda m: print(m, file=sys.stderr))
    print(run_id)
    return 0


def _cmd_ablation(args) -> int:
    from .ablation import run_ablation

    text = run_ablation(args.models, data_dir=args.data_dir, tracking_dir=args.tracking_dir,
                        resamples=args.resamples, seed=args.seed, ranks_out=args.ranks_out, log=lambda m: print(m, file=sys.stderr))
    Path(args.out).write_text(text)
    print(text)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_serve(args) -> int:
    from .mcp_server import create_server

    idx = _load(args.repo, args.index_dir, args.model)
    print(f"serving {len(idx.units)} functions from {args.repo} with {args.model} over stdio", file=sys.stderr)
    create_server(idx).run("stdio")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codesearch", description="Structure-aware retrieval for C codebases.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="sentence-transformers model name or path, or a fine-tuned model from MLflow: "
                        "mlflow:best, mlflow:best-hard, mlflow:best-random or mlflow:RUN_ID (default: %(default)s)")
    p.add_argument("--tracking-dir", help="MLflow file store (default: ./mlruns or $MLFLOW_TRACKING_URI)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index", help="parse and embed a repository, saving the index")
    s.add_argument("repo")
    s.add_argument("--index-dir", help="where to save (default: REPO/.codesearch)")
    s.set_defaults(func=_cmd_index)

    s = sub.add_parser("query", help="search a repository")
    s.add_argument("repo")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--method", choices=METHODS, default="hybrid_graph")
    s.add_argument("--index-dir", help="load a saved index instead of REPO/.codesearch")
    s.add_argument("--show", action="store_true", help="print signatures and comments")
    s.set_defaults(func=_cmd_query)

    s = sub.add_parser("bench", help="run the comment-to-function benchmark and write a markdown table")
    s.add_argument("--fixture", help="benchmark a single local directory (e.g. tests/fixtures/mini_c)")
    s.add_argument("--repo", action="append", help="benchmark a local directory (repeatable)")
    s.add_argument("--data-dir", help="where pinned repositories are cloned (default: data/repos)")
    s.add_argument("--out", default="RESULTS.md")
    s.add_argument("--ablation-model", action="append", metavar="MODEL",
                   help="also evaluate with another embedding model and add a comparison table (repeatable)")
    s.set_defaults(func=_cmd_bench)

    s = sub.add_parser("fetch", help="clone the pinned evaluation repositories")
    s.add_argument("--data-dir")
    s.set_defaults(func=_cmd_fetch)

    s = sub.add_parser("serve", help="run the MCP server over stdio")
    s.add_argument("repo")
    s.add_argument("--index-dir")
    s.set_defaults(func=_cmd_serve)

    s = sub.add_parser("mine", help="mine leak-free (comment, code) training pairs from the training repositories")
    s.add_argument("--data-dir", help="where repositories are cloned (default: data/repos)")
    s.add_argument("--out-dir", default="data/training")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_cmd_mine)

    s = sub.add_parser("train", help="fine-tune the embedding model and log the run to MLflow")
    s.add_argument("--config", default="configs/finetune.toml")
    s.add_argument("--run", default="hard", help="run section of the config (default: %(default)s)")
    s.add_argument("--pairs-dir", default="data/training")
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("ablation", help="compare embedding models on the benchmark with bootstrap intervals")
    s.add_argument("models", nargs="+", metavar="MODEL", help="first model is the reference, e.g. "
                   "BAAI/bge-small-en-v1.5 mlflow:best-random mlflow:best-hard")
    s.add_argument("--data-dir")
    s.add_argument("--resamples", type=int, default=10000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="docs/ablation.md")
    s.add_argument("--ranks-out", help="also save every query's rank per model and method as JSON")
    s.set_defaults(func=_cmd_ablation)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model.startswith("mlflow:") and args.cmd in ("index", "query", "bench", "serve"):
        from .finetune import resolve_model

        try:
            args.model = resolve_model(args.model, args.tracking_dir)
        except LookupError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
