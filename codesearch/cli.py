"""Command-line interface: index, query, bench, fetch and serve."""

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


def _cmd_serve(args) -> int:
    from .mcp_server import create_server

    idx = _load(args.repo, args.index_dir, args.model)
    print(f"serving {len(idx.units)} functions from {args.repo} over stdio", file=sys.stderr)
    create_server(idx).run("stdio")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codesearch", description="Structure-aware retrieval for C codebases.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model (default: %(default)s)")
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
