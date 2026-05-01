# Fine-tuning the embedding model on C

The hybrid retriever's dense side uses `BAAI/bge-small-en-v1.5`, a general English retrieval
model that never saw C comments and identifiers during training. This step tests whether
contrastive fine-tuning on (comment, function) pairs from *other* C projects improves the
benchmark in [RESULTS.md](../RESULTS.md). Short answer: a little. Fine-tuning with BM25 hard
negatives raises hybrid MRR by +0.019 (95% CI +0.004 to +0.035). The other improvements are
of the same size, but their intervals include zero, and on zlib the fine-tuned models are worse
than the base model.

## Training data

`codesearch mine` (`codesearch/mining.py`) reads three C projects pinned to exact commits
(`TRAINING_REPOS` in `codesearch/repos.py`):

| Repository | Tag | Commit | Directories read | Licence |
|---|---|---|---|---|
| [SQLite](https://github.com/sqlite/sqlite) | version-3.47.2 | `262de1bebb0647eb6fa6a2b0434111c7d831a14d` | `src/` | public domain |
| [Redis](https://github.com/redis/redis) | 7.4.2 | `a0a6f23d997b024689ba157916837f493a593a34` | `src/` | BSD-3-Clause |
| [curl](https://github.com/curl/curl) | curl-8_11_1 | `75a2079d5c28debb2eaa848ca9430f1fe0d7844c` | `lib/`, `src/` | curl |

Redis vendors Lua (a benchmark repository) under `deps/`, which is why only its own `src/` is
read. A pair is the benchmark-style query built from a function's leading comment (same
rules as the benchmark: at least 4 words, no licence banners, cut to 64 words) and the text the
dense index embeds for that function with its comment removed (split name, path, code).

**Leakage control.** Training data must not contain anything the benchmark asks for:

1. `check_disjoint_repos` refuses a training repository whose name or URL is a benchmark
   repository.
2. Every training function is compared with all 2,620 functions of Lua, zlib and jq. It is
   dropped when its body (without the name/path header, so copies under another path count) has
   the same normalised token sequence as a benchmark function, or a token 5-shingle Jaccard
   similarity ≥ 0.5 with one. Its comment is dropped when it equals a benchmark comment or has a
   word 3-shingle Jaccard ≥ 0.6 with one. Dropped functions are not used as negatives either.
3. After mining, `find_leaks` audits the saved pairs (positives, queries and negatives) again
   from scratch, and `codesearch mine` fails if anything overlaps.

Duplicate bodies within a repository, comments shared by several functions and bodies shorter
than 3 lines are also dropped. The split is by source file (a hash of the path, 10% to
validation), so no file contributes to both training and validation.

| Repository | Functions | With usable comment | Benchmark near-duplicate | Duplicate code | Duplicate comment | Too short | Pairs kept |
|---|---:|---:|---:|---:|---:|---:|---:|
| sqlite | 4401 | 3428 | 0 | 4 | 221 | 26 | 3181 |
| redis | 4521 | 2838 | 0 | 12 | 82 | 7 | 2749 |
| curl | 3341 | 1087 | 2 | 6 | 122 | 0 | 965 |

That gives 6,217 training and 678 validation pairs, and the audit found no overlap. Only two
functions came close to benchmark code (both in curl), which shows that the projects really are
separate.

**Negatives.** Each pair gets one extra negative from the same repository and split: either a
uniformly random function (`pairs_random.jsonl`) or one drawn at random from the 10 best BM25
hits for the query, excluding the function itself (`pairs_hard.jsonl`). The BM25 index is the
project's own (`codesearch/bm25.py`) over the same identifier-split tokens used for retrieval.

## Training

`codesearch train --run hard|random` (`codesearch/finetune.py`, config in
`configs/finetune.toml`) fine-tunes the whole model with sentence-transformers'
`MultipleNegativesRankingLoss` (scale 20) in a plain PyTorch loop. Each batch of 32 holds
(query, positive, negative) triples, and every query is scored against all 64 code texts in
the batch. Other settings:

| Setting | Value |
|---|---|
| Base model | `BAAI/bge-small-en-v1.5` (33M parameters) |
| Training pairs | 2,880 sampled from the 6,217 (seed 0), 1 epoch, 90 steps |
| Optimiser | AdamW, lr 3e-5, weight decay 0.01, linear warm-up 10% then linear decay, grad-norm clip 1.0 |
| Sequence length | 128 tokens in training (512 at inference, as for the base model) |
| Memory | gradient checkpointing; peak RSS about 2.7 GB |
| Seeds | Python, NumPy and PyTorch seeded with 0; batch order from a seeded RNG |
| Validation | MRR of 300 held-out queries against the 300 held-out positives, every 15 steps |
| Hardware | 4 CPU cores, about 29 minutes per run |

The training subset and sequence length were chosen to fit two runs plus evaluation into the
time and memory available. They were not tuned, and no other hyper-parameters were tried.

**MLflow.** Each run is logged to a local file store (`./mlruns`, experiment
`codesearch-finetune`): all config values, training and validation pair counts and the query
prefix as params; `train_loss` and `learning_rate` per step; `val_mrr` (step 0 is the base
model) and `final_val_mrr`; the pairs file as `data_info.json`; and the fine-tuned model as the
`model` artefact. A `codesearch.json` file is stored next to the model with the base model and
its query instruction prefix, which the embedder reads so queries are encoded exactly as in
training. `--model mlflow:best` selects the finished run with the highest final validation MRR;
`mlflow:best-hard`, `mlflow:best-random` and `mlflow:<run_id>` are also accepted by every
command, including `serve` (the MCP server).

| Run | Negatives | Run id | Val MRR step 0 → 15 → 30 → 45 → 60 → 75 → 90 |
|---|---|---|---|
| hard | BM25 top-10 | `04a293d7` | 0.805 → 0.834 → 0.853 → 0.859 → 0.859 → 0.859 → **0.862** |
| random | uniform | `4209cb2c` | 0.805 → 0.831 → 0.850 → 0.857 → 0.860 → 0.858 → **0.859** |

Mean training loss over each block of 15 steps went from 1.06 to about 0.5 (hard) and from 0.95
to about 0.38 (random). The hard-negative loss stays higher because its negatives are harder to
separate. Validation MRR levels off after about 45 steps, so more steps of the same data would
probably not help much.

## Ablation on the benchmark

`codesearch ablation BAAI/bge-small-en-v1.5 mlflow:best-random mlflow:best-hard` evaluates each
model on the unchanged benchmark (Lua, zlib and jq at their pinned commits, the same 1,014
queries, comments removed from the index, ranks over every function). Each model is run dense-only
and inside the existing fusion (same RRF weights, chosen earlier for the base model). Differences
are paired per query. The 95% intervals come from 10,000 bootstrap resamples drawn *within each
repository* and then macro-averaged, so they match the macro averages in the table. The
per-query ranks are in [ablation_ranks.json](ablation_ranks.json), and the generated table is
[ablation.md](ablation.md).

| Model | Method | Recall@1 | Recall@10 | MRR | ΔMRR [95% CI] | ΔRecall@10 [95% CI] | P(ΔMRR≤0) |
|---|---|---:|---:|---:|---:|---:|---:|
| base | Dense | 0.537 | 0.866 | 0.661 | — | — | — |
| base | Hybrid (RRF) | 0.546 | 0.878 | 0.667 | — | — | — |
| base | Hybrid + call graph | 0.546 | 0.895 | 0.669 | — | — | — |
| random negatives | Dense | 0.554 | 0.880 | 0.671 | +0.011 [-0.004, +0.026] | +0.014 [-0.001, +0.028] | 0.084 |
| random negatives | Hybrid (RRF) | 0.562 | 0.885 | 0.678 | +0.012 [-0.002, +0.026] | +0.007 [-0.006, +0.021] | 0.051 |
| random negatives | Hybrid + call graph | 0.562 | 0.891 | 0.680 | +0.011 [-0.003, +0.026] | -0.004 [-0.017, +0.009] | 0.055 |
| hard negatives | Dense | 0.559 | 0.876 | 0.675 | +0.014 [-0.002, +0.030] | +0.010 [-0.005, +0.025] | 0.040 |
| hard negatives | Hybrid (RRF) | 0.570 | 0.891 | 0.686 | **+0.019 [+0.004, +0.035]** | +0.014 [+0.000, +0.027] | 0.005 |
| hard negatives | Hybrid + call graph | 0.570 | 0.899 | 0.687 | **+0.019 [+0.004, +0.034]** | +0.004 [-0.008, +0.016] | 0.007 |

MRR per repository:

| Model | Method | Lua (497 q) | zlib (260 q) | jq (257 q) |
|---|---|---:|---:|---:|
| base | Dense | 0.594 | 0.620 | 0.768 |
| base | Hybrid + call graph | 0.594 | 0.636 | 0.776 |
| random negatives | Dense | 0.625 | 0.597 | 0.791 |
| random negatives | Hybrid + call graph | 0.618 | 0.612 | 0.810 |
| hard negatives | Dense | 0.622 | 0.604 | 0.798 |
| hard negatives | Hybrid + call graph | 0.624 | 0.624 | 0.813 |

## What this shows

- **The gain is real but small.** Every fine-tuned configuration improves macro MRR by 0.011 to
  0.019, which is 2–3% relative. Only the hard-negative model inside the fusion has an interval
  that excludes zero. Six comparisons are made against one baseline with no correction for
  multiple testing, so treat the dense-only and random-negative intervals, which include zero,
  as "no clear effect".
- **Hard negatives help a little more than random ones** (+0.019 vs +0.012 hybrid MRR). Their
  intervals overlap a lot, and with one seed per configuration this ranking is not established.
- **Fusion and fine-tuning add up.** The hard-negative model gains more inside the hybrid
  (+0.019) than alone (+0.014), so BM25 still adds something the fine-tuned model lacks.
- **The gain is uneven.** Lua (+0.02 to +0.03) and jq (+0.02 to +0.04) improve, while zlib loses 0.012 to
  0.024 MRR with every fine-tuned model. zlib's terse, compression-specific comments are nothing
  like the database, network and server code of the training projects. A fine-tuned model is not
  a safe default for every codebase, which is why the base model remains the default and the
  fine-tuned one is opt-in.
- **Recall@10 barely moves** once the call-graph step is applied (+0.004, interval includes zero).
  Most of the gain is at the top of the ranking (Recall@1 0.546 → 0.570).

## Limitations

- One seed and one hyper-parameter setting per configuration. The run-to-run variance of
  fine-tuning was not measured, and it may be as large as the gap between random and hard
  negatives.
- Only 2,880 of the 6,217 training pairs were used, for one epoch at 128 tokens, to fit the CPU
  time budget. Longer training at full length may change the picture, in either direction.
- The RRF dense weight (5) and the call-graph parameters were tuned on zlib with the base model
  and were not re-tuned for the fine-tuned models.
- Validation MRR uses 300 candidates and is much easier than the benchmark (ranks over all
  functions of a repository), so it is useful for monitoring training but does not predict the
  size of the benchmark gain.
- The near-duplicate thresholds (Jaccard 0.5 on code, 0.6 on comments) are heuristics. A
  benchmark function rewritten more heavily could still pass, although with three unrelated
  projects there is little reason to expect copies.

## Reproducing

```bash
python3 -m codesearch.cli mine                 # ~20 s after the clones; writes data/training/
python3 -m codesearch.cli train --run hard     # ~30 min on 4 cores
python3 -m codesearch.cli train --run random   # ~30 min
python3 -m codesearch.cli ablation BAAI/bge-small-en-v1.5 mlflow:best-random mlflow:best-hard \
    --out docs/ablation.md --ranks-out docs/ablation_ranks.json   # ~13 min
```

Neither the mined pairs nor the MLflow store (about 130 MB per model) are committed. CPU
floating-point results can vary slightly between machines and library versions, so retrained
models may differ in the last digits.
