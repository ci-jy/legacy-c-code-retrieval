Macro average over lua, zlib, jq (1014 queries). Intervals are 95% paired bootstrap intervals (10000 resamples, stratified by repository) of the difference to `BAAI/bge-small-en-v1.5`; P(Δ≤0) is the share of resamples in which the model did not improve.

| Model | Method | Recall@1 | Recall@10 | MRR | ΔMRR [95% CI] | ΔRecall@10 [95% CI] | P(ΔMRR≤0) |
|---|---|---:|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | Dense | 0.537 | 0.866 | 0.661 | — | — | — |
| BAAI/bge-small-en-v1.5 | Hybrid (RRF) | 0.546 | 0.878 | 0.667 | — | — | — |
| BAAI/bge-small-en-v1.5 | Hybrid + call graph | 0.546 | 0.895 | 0.669 | — | — | — |
| mlflow:best-random (4209cb2c) | Dense | 0.554 | 0.880 | 0.671 | +0.011 [-0.004, +0.026] | +0.014 [-0.001, +0.028] | 0.084 |
| mlflow:best-random (4209cb2c) | Hybrid (RRF) | 0.562 | 0.885 | 0.678 | +0.012 [-0.002, +0.026] | +0.007 [-0.006, +0.021] | 0.051 |
| mlflow:best-random (4209cb2c) | Hybrid + call graph | 0.562 | 0.891 | 0.680 | +0.011 [-0.003, +0.026] | -0.004 [-0.017, +0.009] | 0.055 |
| mlflow:best-hard (04a293d7) | Dense | 0.559 | 0.876 | 0.675 | +0.014 [-0.002, +0.030] | +0.010 [-0.005, +0.025] | 0.040 |
| mlflow:best-hard (04a293d7) | Hybrid (RRF) | 0.570 | 0.891 | 0.686 | +0.019 [+0.004, +0.035] | +0.014 [+0.000, +0.027] | 0.005 |
| mlflow:best-hard (04a293d7) | Hybrid + call graph | 0.570 | 0.899 | 0.687 | +0.019 [+0.004, +0.034] | +0.004 [-0.008, +0.016] | 0.007 |

Per repository (MRR):

| Model | Method | lua | zlib | jq |
|---|---|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | Dense | 0.594 | 0.620 | 0.768 |
| BAAI/bge-small-en-v1.5 | Hybrid (RRF) | 0.590 | 0.635 | 0.776 |
| BAAI/bge-small-en-v1.5 | Hybrid + call graph | 0.594 | 0.636 | 0.776 |
| mlflow:best-random (4209cb2c) | Dense | 0.625 | 0.597 | 0.791 |
| mlflow:best-random (4209cb2c) | Hybrid (RRF) | 0.615 | 0.611 | 0.809 |
| mlflow:best-random (4209cb2c) | Hybrid + call graph | 0.618 | 0.612 | 0.810 |
| mlflow:best-hard (04a293d7) | Dense | 0.622 | 0.604 | 0.798 |
| mlflow:best-hard (04a293d7) | Hybrid (RRF) | 0.622 | 0.623 | 0.813 |
| mlflow:best-hard (04a293d7) | Hybrid + call graph | 0.624 | 0.624 | 0.813 |