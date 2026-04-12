# Retrieval benchmark results

Task: each function's leading comment is used as a natural-language query; the function it documents is the only correct answer. All leading comments are removed from the index first, so a method has to connect the description to code. Comments with fewer than 4 words, licence banners and comments shared by several functions are skipped; queries are cut to 64 words. Ranks are over every function in the repository.

- Embedding model: `BAAI/bge-small-en-v1.5` (CPU, not fine-tuned)
- Hybrid: weighted reciprocal-rank fusion (k=60; BM25 weight 1, dense weight 5) of the top 100 BM25 and dense results
- Hybrid + call graph: hybrid, then callers/callees of the top 3 hits are promoted (top 3 kept in place)
- Python 3.14.4

## Corpora

| Repository | Source |
|---|---|
| lua | https://github.com/lua/lua.git @ 312b9efaa106 (v5.4.9, MIT) |
| zlib | https://github.com/madler/zlib.git @ da607da739fa (v1.3.2, zlib) |
| jq | https://github.com/jqlang/jq.git @ 34f7186b8674 (jq-1.8.2, MIT) |

| Repository | Functions indexed | Queries | Time (s) |
|---|---:|---:|---:|
| lua | 1212 | 497 | 98.8 |
| zlib | 530 | 260 | 56.9 |
| jq | 878 | 257 | 77.3 |

## Results

| Repository | Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| lua | BM25 (baseline) | 0.378 | 0.590 | 0.684 | 0.484 |
| lua | Dense | 0.465 | 0.751 | 0.837 | 0.594 |
| lua | Hybrid (RRF) | 0.455 | 0.757 | 0.845 | 0.590 |
| lua | Hybrid + call graph | 0.455 | 0.779 | 0.865 | 0.594 |
| zlib | BM25 (baseline) | 0.346 | 0.619 | 0.704 | 0.471 |
| zlib | Dense | 0.485 | 0.788 | 0.838 | 0.620 |
| zlib | Hybrid (RRF) | 0.512 | 0.785 | 0.858 | 0.635 |
| zlib | Hybrid + call graph | 0.512 | 0.796 | 0.881 | 0.636 |
| jq | BM25 (baseline) | 0.498 | 0.821 | 0.887 | 0.642 |
| jq | Dense | 0.661 | 0.911 | 0.922 | 0.768 |
| jq | Hybrid (RRF) | 0.673 | 0.914 | 0.930 | 0.776 |
| jq | Hybrid + call graph | 0.673 | 0.914 | 0.938 | 0.776 |

## Macro average over repositories

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 (baseline) | 0.407 | 0.677 | 0.758 | 0.532 |
| Dense | 0.537 | 0.816 | 0.866 | 0.661 |
| Hybrid (RRF) | 0.546 | 0.819 | 0.878 | 0.667 |
| Hybrid + call graph | 0.546 | 0.830 | 0.895 | 0.669 |

## Hybrid vs. BM25 (MRR)

- lua: BM25 0.484, hybrid 0.590 (+0.106), hybrid + graph 0.594 (+0.109)
- zlib: BM25 0.471, hybrid 0.635 (+0.164), hybrid + graph 0.636 (+0.165)
- jq: BM25 0.642, hybrid 0.776 (+0.134), hybrid + graph 0.776 (+0.134)

## Embedding model ablation: `sentence-transformers/all-MiniLM-L6-v2`

Same queries and index settings; only the embedding model differs (BM25 is unchanged).

| Repository | Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| lua | BM25 (baseline) | 0.378 | 0.590 | 0.684 | 0.484 |
| lua | Dense | 0.362 | 0.628 | 0.734 | 0.488 |
| lua | Hybrid (RRF) | 0.370 | 0.652 | 0.763 | 0.505 |
| lua | Hybrid + call graph | 0.370 | 0.678 | 0.781 | 0.509 |
| zlib | BM25 (baseline) | 0.346 | 0.619 | 0.704 | 0.471 |
| zlib | Dense | 0.350 | 0.627 | 0.735 | 0.471 |
| zlib | Hybrid (RRF) | 0.377 | 0.681 | 0.765 | 0.512 |
| zlib | Hybrid + call graph | 0.377 | 0.712 | 0.815 | 0.519 |
| jq | BM25 (baseline) | 0.498 | 0.821 | 0.887 | 0.642 |
| jq | Dense | 0.381 | 0.658 | 0.763 | 0.517 |
| jq | Hybrid (RRF) | 0.482 | 0.724 | 0.837 | 0.596 |
| jq | Hybrid + call graph | 0.482 | 0.759 | 0.856 | 0.601 |

Macro average:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 (baseline) | 0.407 | 0.677 | 0.758 | 0.532 |
| Dense | 0.364 | 0.637 | 0.744 | 0.492 |
| Hybrid (RRF) | 0.410 | 0.685 | 0.788 | 0.538 |
| Hybrid + call graph | 0.410 | 0.716 | 0.817 | 0.543 |
