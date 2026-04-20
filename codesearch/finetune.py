"""Contrastive fine-tuning of the embedding model on mined C pairs, tracked with MLflow.

The training loop is plain PyTorch around a SentenceTransformer and its
MultipleNegativesRankingLoss: every batch holds (query, positive code, negative code) triples,
and each query is scored against all positives and negatives in the batch. Seeds are fixed,
and validation MRR on the held-out split is logged before training and every `eval_every` steps.
Runs go to a local MLflow file store; the trained model is logged as the run's artefact and
can be selected later with the model spec ``mlflow:best`` or ``mlflow:<run_id>``.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from .dense import DEFAULT_MODEL, META_FILE, query_prefix
from .mining import TrainingPair

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

DEFAULT_TRACKING_DIR = Path(__file__).resolve().parent.parent / "mlruns"
EXPERIMENT = "codesearch-finetune"
MLFLOW_PREFIX = "mlflow:"


def _mlflow():
    """Import mlflow after transformers: mlflow pulls in the importlib_metadata backport, which makes
    transformers' import-time package scan fail on distributions with incomplete metadata."""
    import transformers  # noqa: F401
    import mlflow

    return mlflow


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_MODEL
    negatives: str = "hard"  # "hard", "random" or "none" (in-batch negatives only)
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 256
    scale: float = 20.0
    seed: int = 0
    max_train_pairs: int = 0  # 0: use all
    max_val_pairs: int = 500
    eval_every: int = 50
    threads: int = 0  # 0: PyTorch default
    gradient_checkpointing: bool = True  # recompute activations in backward to stay within a few GB of RAM

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown training options: {', '.join(sorted(unknown))}")
        return cls(**d)


def load_config(path: str | Path, run: str) -> TrainConfig:
    """Read [defaults] and [runs.<run>] from a TOML file; the run section overrides the defaults."""
    data = tomllib.loads(Path(path).read_text())
    runs = data.get("runs", {})
    if run not in runs:
        raise ValueError(f"no run {run!r} in {path}; available: {', '.join(runs)}")
    return TrainConfig.from_dict({**data.get("defaults", {}), **runs[run]})


def tracking_uri(directory: str | Path | None = None) -> str:
    if directory is None and os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    return Path(directory or DEFAULT_TRACKING_DIR).resolve().as_uri()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validation_mrr(model, pairs: list[TrainingPair], prefix: str, batch_size: int = 64) -> float:
    """MRR of each validation query against the positives of all validation pairs."""
    import torch

    if not pairs:
        return 0.0
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        q = model.encode([prefix + p.query for p in pairs], batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True)
        d = model.encode([p.code for p in pairs], batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True)
    model.train(was_training)
    sims = q @ d.T
    target = np.diag(sims)
    ranks = 1 + (sims > target[:, None]).sum(axis=1)
    return float((1.0 / ranks).mean())


def _mnrl(model, scale: float):
    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:  # sentence-transformers < 6
        from sentence_transformers.losses import MultipleNegativesRankingLoss
    return MultipleNegativesRankingLoss(model, scale=scale)


def _features(model, texts: list[str]) -> dict:
    tok = getattr(model, "preprocess", None) or model.tokenize  # tokenize is deprecated in sentence-transformers 6
    return dict(tok(texts).items())


def train(cfg: TrainConfig, train_pairs: list[TrainingPair], val_pairs: list[TrainingPair],
          tracking_dir: str | Path | None = None, run_name: str | None = None,
          data_info: dict | None = None, log=print) -> str:
    """Fine-tune, log everything to MLflow and return the run id."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import get_linear_schedule_with_warmup

    mlflow = _mlflow()

    if cfg.negatives not in ("hard", "random", "none"):
        raise ValueError(f"negatives must be hard, random or none, not {cfg.negatives!r}")
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    train_pairs = list(train_pairs)
    if cfg.max_train_pairs and len(train_pairs) > cfg.max_train_pairs:
        train_pairs = rng.sample(train_pairs, cfg.max_train_pairs)
    val_pairs = list(val_pairs)
    if cfg.max_val_pairs and len(val_pairs) > cfg.max_val_pairs:
        val_pairs = random.Random(cfg.seed + 1).sample(val_pairs, cfg.max_val_pairs)
    if cfg.negatives != "none" and any(not p.negative for p in train_pairs):
        raise ValueError("negatives requested but some training pairs have none")

    try:
        model = SentenceTransformer(cfg.base_model, device="cpu", local_files_only=True)
    except OSError:
        model = SentenceTransformer(cfg.base_model, device="cpu")
    inference_len = model.max_seq_length
    if cfg.gradient_checkpointing:
        model[0].auto_model.gradient_checkpointing_enable()
    model.max_seq_length = cfg.max_seq_length
    prefix = query_prefix(cfg.base_model)
    loss_fn = _mnrl(model, cfg.scale)

    steps_per_epoch = max(1, len(train_pairs) // cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.learning_rate)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)

    mlflow.set_tracking_uri(tracking_uri(tracking_dir))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**asdict(cfg), "n_train": len(train_pairs), "n_val": len(val_pairs),
                           "steps": total, "query_prefix": prefix})
        mlflow.set_tags({"negatives": cfg.negatives, "base_model": cfg.base_model})
        if data_info:
            mlflow.log_dict(data_info, "data_info.json")

        best = validation_mrr(model, val_pairs, prefix)
        mlflow.log_metric("val_mrr", best, step=0)
        mlflow.log_metric("base_val_mrr", best)
        log(f"[train] {run.info.run_id}: {len(train_pairs)} pairs, {total} steps; base val MRR {best:.4f}")
        model.train()
        step, t0 = 0, time.time()
        for epoch in range(cfg.epochs):
            order = list(range(len(train_pairs)))
            rng.shuffle(order)
            for b in range(steps_per_epoch):
                batch = [train_pairs[i] for i in order[b * cfg.batch_size : (b + 1) * cfg.batch_size]]
                columns = [[prefix + p.query for p in batch], [p.code for p in batch]]
                if cfg.negatives != "none":
                    columns.append([p.negative for p in batch])
                loss = loss_fn([_features(model, c) for c in columns], torch.empty(0))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step += 1
                mlflow.log_metric("train_loss", float(loss.item()), step=step)
                mlflow.log_metric("learning_rate", sched.get_last_lr()[0], step=step)
                if step % cfg.eval_every == 0 or step == total:
                    mrr = validation_mrr(model, val_pairs, prefix)
                    mlflow.log_metric("val_mrr", mrr, step=step)
                    log(f"[train] step {step}/{total} epoch {epoch} loss {loss.item():.4f} val MRR {mrr:.4f} "
                        f"({time.time() - t0:.0f}s)")
        final = validation_mrr(model, val_pairs, prefix)
        mlflow.log_metric("final_val_mrr", final)
        mlflow.log_metric("train_seconds", time.time() - t0)

        model.eval()
        model.max_seq_length = inference_len
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model"
            model.save(str(out))
            (out / META_FILE).write_text(json.dumps({"base_model": cfg.base_model, "query_prefix": prefix,
                                                     "negatives": cfg.negatives, "run_id": run.info.run_id}))
            mlflow.log_artifacts(str(out), artifact_path="model")
        return run.info.run_id


# ---------------------------------------------------------------- model selection
def _artifact_dir(run) -> Path:
    uri = run.info.artifact_uri
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return Path(unquote(parsed.path)) / "model"
    return Path(_mlflow().artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="model"))


def find_runs(tracking_dir: str | Path | None = None, negatives: str | None = None) -> list:
    """Finished fine-tuning runs with a model artefact, best final validation MRR first."""
    mlflow = _mlflow()

    mlflow.set_tracking_uri(tracking_uri(tracking_dir))
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        return []
    flt = "attributes.status = 'FINISHED'"
    if negatives:
        flt += f" and tags.negatives = '{negatives}'"
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], filter_string=flt, output_format="list",
                              order_by=["metrics.final_val_mrr DESC", "attributes.start_time DESC"])
    return [r for r in runs if "final_val_mrr" in r.data.metrics and (_artifact_dir(r) / META_FILE).exists()]


def resolve_model(spec: str, tracking_dir: str | Path | None = None) -> str:
    """Turn a model spec into something SentenceTransformer can load.

    ``mlflow:best`` is the finished run with the highest final validation MRR,
    ``mlflow:best-hard`` / ``mlflow:best-random`` the best run with that negative mode,
    ``mlflow:<run_id>`` a specific run. Anything else (hub name or path) is returned unchanged.
    """
    if not spec.startswith(MLFLOW_PREFIX):
        return spec
    key = spec[len(MLFLOW_PREFIX):]
    if key == "best" or key.startswith("best-"):
        runs = find_runs(tracking_dir, key[5:] or None)
        if not runs:
            raise LookupError(f"no finished fine-tuning run for {spec!r} in {tracking_uri(tracking_dir)}")
        return str(_artifact_dir(runs[0]))
    mlflow = _mlflow()

    mlflow.set_tracking_uri(tracking_uri(tracking_dir))
    try:
        run = mlflow.get_run(key)
    except Exception as e:  # mlflow raises its own exception types for unknown ids
        raise LookupError(f"no MLflow run {key!r}") from e
    path = _artifact_dir(run)
    if not path.is_dir():
        raise LookupError(f"run {key} has no model artefact")
    return str(path)
