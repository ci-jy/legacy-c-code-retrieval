# codesearch: structure-aware code retrieval for undocumented C

`codesearch` finds where a behaviour lives in a C codebase. It splits the code at function
boundaries with tree-sitter, builds a static call graph, indexes each function both lexically
(BM25 over split identifiers) and semantically (a local sentence-transformers model on the CPU),
fuses the two rankings, and optionally promotes callers and callees of the best hits. The same
index is served to LLM agents through a Model Context Protocol (MCP) server with four tools:
`search_code`, `get_function`, `callers_of` and `callees_of`.

A benchmark harness measures retrieval quality on real open-source C projects pinned to exact
commits; the latest numbers are in [RESULTS.md](RESULTS.md).

| Method (macro average over Lua, zlib, jq) | Recall@1 | Recall@10 | MRR |
|---|---:|---:|---:|
| BM25 baseline | 0.407 | 0.758 | 0.532 |
| Dense (all-MiniLM-L6-v2) | 0.364 | 0.744 | 0.492 |
| Hybrid (RRF) | 0.457 | 0.802 | 0.579 |
| Hybrid + call graph | 0.457 | 0.813 | 0.582 |

## Installation

Python 3.10+ is required (developed on 3.14). Everything runs locally on the CPU; no hosted
model or embedding API is called.

```bash
python3 -m pip install -r requirements.txt     # CPU-only PyTorch wheels + the rest
# or: python3 -m pip install -e '.[test]'
```

The first run downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB)
into the Hugging Face cache; later runs load it from the cache without network access. Use
`--model NAME` (or the `CODESEARCH_MODEL` environment variable) to choose another
sentence-transformers model.

## Usage

All commands are run from the repository root (or use the `codesearch` entry point after
`pip install -e .`).

```bash
# Index a repository (saved under <repo>/.codesearch, or --index-dir DIR)
python3 -m codesearch.cli index path/to/c-project

# Search it (methods: bm25, dense, hybrid, hybrid_graph [default])
python3 -m codesearch.cli query path/to/c-project "where are config lines parsed" -k 5 --show

# Benchmark the small fixture shipped with the tests
python3 -m codesearch.cli bench --fixture tests/fixtures/mini_c --out /tmp/bench.md

# Clone the pinned evaluation repositories into data/repos/ and regenerate RESULTS.md
python3 -m codesearch.cli fetch          # or: python3 scripts/fetch_repos.py
python3 -m codesearch.cli bench --out RESULTS.md

# Benchmark any local directory (repeatable)
python3 -m codesearch.cli bench --repo path/to/c-project --out my-results.md

# Serve a repository to an MCP client over stdio
python3 -m codesearch.cli serve path/to/c-project
```

`query` and `serve` reuse the saved index when the parsed functions and model are unchanged
and rebuild it otherwise.

Example (on the test fixture):

```
$ python3 -m codesearch.cli query tests/fixtures/mini_c "double the bucket count and rehash" -k 3
 1. 0.0460  table_resize  src/hash.c:49-71
 2. 0.0457  table_put  src/hash.c:31-45
 3. 0.0445  table_free  src/hash.c:83-88
```

## Connecting the MCP server to an agent

The server uses the official MCP Python SDK and the stdio transport. Any MCP client that can
launch a local command can use it. A typical client configuration (the `mcpServers` format
used by Claude Desktop, Claude Code's `.mcp.json`, Cursor and others):

```json
{
  "mcpServers": {
    "codesearch": {
      "command": "python3",
      "args": ["-m", "codesearch.cli", "serve", "/abs/path/to/c-project"],
      "cwd": "/abs/path/to/legacy-c-code-retrieval"
    }
  }
}
```

Index large repositories once with `codesearch index` beforehand so the server starts quickly.

| Tool | Arguments | Returns |
|---|---|---|
| `search_code` | `query`, `k` (≤ 50, default 10), `method` | ranked hits: id, name, file, line span, signature, leading comment, score |
| `get_function` | `name` (bare name or `file.c::name` id) | every match with full source (comment included), callers and callees ids |
| `callers_of` | `name` | for each match, the repository functions that call it |
| `callees_of` | `name` | for each match, the repository functions it calls |

All tools return structured content with an output schema (pydantic models in
`codesearch/mcp_server.py`) plus the same JSON as text. Unknown functions or methods produce an
MCP tool error with a readable message. Function ids have the form `src/file.c::name`
(`@line` is appended when one file defines the same name twice, e.g. in `#ifdef` branches).

## Design

```
codesearch/
  parser.py      tree-sitter C -> FunctionUnit (name, signature, body, leading comment, file, lines, calls)
  callgraph.py   caller/callee edges between units
  tokenize.py    identifier splitting (snake_case, camelCase, acronyms) for BM25
  bm25.py        Okapi BM25 (k1=1.2, b=0.75), numpy, deterministic
  dense.py       sentence-transformers embedder (PyTorch, CPU, L2-normalised)
  ranking.py     rank ordering, reciprocal-rank fusion, call-graph expansion
  index.py       CodeIndex: build/save/load/search over all of the above
  evaluate.py    comment-to-function benchmark and metrics
  report.py      markdown report
  repos.py       pinned evaluation repositories
  mcp_server.py  MCP tools
  cli.py         index / query / bench / fetch / serve
```

**Function units.** Every `function_definition` node in `.c` and `.h` files becomes a unit,
including those inside `#if`/`#ifdef` blocks. Declarators are unwrapped through pointers and
parentheses, so `int **f(void)` and `int (*f(int))(int)` are both named `f`; macro prefixes such
as `LUA_API int f(...)` parse with a local error node and are still recovered. Prototypes are not
units. The leading comment is the run of comment nodes directly above the definition (at most
one blank line between them); it is cleaned of `/* */`, `//`, `**`, rules made of `=`/`-` and
similar decoration.

**Call graph.** Calls are `call_expression`s whose callee is a plain identifier. A call is resolved
roughly as the C linker would, without running the preprocessor: a definition in the same file
wins, otherwise every non-static definition of that name (plus `static` functions defined in
headers, which are textually included). Calls through function pointers, macros that expand to
calls, and calls to functions outside the repository produce no edge.

**Lexical index.** Each unit's text (path + code, plus the leading comment in normal use) is
tokenised into identifier pieces: `luaH_getShortStr` becomes `luah_getshortstr lua h get short str`.
C keywords, English stop words, numbers and one-character tokens are dropped. Queries are
tokenised the same way, which is what lets "get short string" reach `luaH_getShortStr`.

**Dense index.** The embedding input is the split function name and path, followed by the code,
so that the most informative part survives the model's 256-token truncation. Vectors are
normalised, so scoring is a single matrix-vector product. Embeddings are stored next to the
saved index together with a fingerprint of the model and texts.

**Hybrid ranking.** Reciprocal-rank fusion, `score(d) = Σ 1 / (60 + rank_i(d))`, over the top 100
of each ranking. RRF needs no score calibration between BM25 and cosine similarity, which live on
unrelated scales.

**Call-graph expansion.** After fusion the top 3 hits ("seeds") keep their positions; every other
function that calls or is called by a seed gains `0.5 · score(seed) / degree(seed)`. Dividing by
degree stops hubs such as allocation helpers from flooding the list. An earlier variant that let
neighbours overtake the seeds cut Recall@1 on zlib from 0.40 to 0.16–0.38 depending on the
weight, which is why the seeds are protected. Seeds = 3 and weight = 0.5 were chosen on zlib only;
Lua and jq were not used for tuning.

## Evaluation method

There is no labelled query set for these projects, so the benchmark builds one from the code:

1. Parse the repository into function units.
2. For every function whose cleaned leading comment has at least 4 words (excluding licence
   banners, and excluding comment texts shared by several functions), the comment is the
   query and that function is the single correct answer. Queries are cut to 64 words.
3. Build the index **with every leading comment removed**, so the query text is not in the
   index and a method has to connect the description to the code itself.
4. Rank all functions in the repository for each query and report Recall@1/5/10 and MRR
   (mean reciprocal rank over the full ranking) for BM25, dense, hybrid and hybrid + graph.

The repositories are fetched by `codesearch fetch` / `scripts/fetch_repos.py` with a shallow fetch
of an exact commit, and the checked-out hash is verified:

| Repository | Tag | Commit | Licence |
|---|---|---|---|
| [Lua](https://github.com/lua/lua) | v5.4.9 | `312b9efaa1061c2c4cad08554dbc1351c3270eef` | MIT |
| [zlib](https://github.com/madler/zlib) | v1.3.2 | `da607da739fa6047df13e66a2af6b8bec7c2a498` | zlib |
| [jq](https://github.com/jqlang/jq) | jq-1.8.2 | `34f7186b86743a083a589741b6cea95293524108` | MIT |

The repositories are only downloaded for evaluation (into the git-ignored `data/repos/`); none of
their code is included here. A full run over all three takes about 1.5 minutes on 4 CPU cores.

## Tests

```bash
python3 -m pytest -q
```

The suite uses the hand-written fixture in `tests/fixtures/mini_c` (three `.c` files and two
headers: a string buffer, a hash table and a config loader, with two same-named `static`
functions, a recursive function, a call through a function pointer, an `#ifdef` block and a
`static inline` header function). It checks:

- the extracted function set, line spans, `static` flags, signatures and leading comments
  match the expected values exactly, and the call-graph edge set matches exactly (19 edges);
- BM25 scores against hand-computed values, and that BM25 and dense rankings are deterministic;
- reciprocal-rank fusion and graph expansion against hand-computed cases;
- the benchmark query construction (comments removed from the index, short and duplicate
  comments skipped) and the metrics;
- the four MCP tools through an in-process MCP client and through a stdio subprocess,
  including schemas, structured results and error cases;
- the CLI `index`, `query` and `bench` commands.

The dense tests use the real embedding model, so it must be downloadable or already cached.

## Limitations

- **No preprocessor.** Macros are not expanded, so functions generated by macros are invisible and
  calls hidden inside macros produce no edges. Heavily macro-based code can confuse tree-sitter;
  the parser recovers most definitions but not all.
- **Approximate call resolution.** Name-based resolution can add edges between unrelated
  functions with the same external name in different programs of one repository (e.g. examples
  or tests), and misses all indirect calls through function pointers.
- **The benchmark is a proxy.** Leading comments are written by the authors of the code and
  often reuse its vocabulary, which favours lexical matching; real user questions may be phrased
  differently. Each query has exactly one correct answer, so the call-graph step, which mainly
  surfaces *related* functions, can only gain a little on this metric.
- **General-purpose embedding model.** `all-MiniLM-L6-v2` was not trained on code and truncates
  inputs to 256 tokens, so long functions are embedded by their first part only. The model is not
  fine-tuned.
- **Graph expansion parameters** were tuned on one repository (zlib) with a small grid.
- **C only**, and indexing is not incremental: any change re-embeds the whole repository
  (about 30 seconds per thousand functions on 4 cores).

## Third-party software

Built on [tree-sitter](https://tree-sitter.github.io/) and
[tree-sitter-c](https://github.com/tree-sitter/tree-sitter-c) (MIT),
[sentence-transformers](https://www.sbert.net/) (Apache-2.0), [PyTorch](https://pytorch.org/)
(BSD-3-Clause), the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (MIT),
numpy, pydantic and pytest. The default embedding model is
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0).
No third-party source code is copied into this repository.

## License

MIT
