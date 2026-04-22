import json
import shutil

import pytest

from codesearch.dense import QUERY_PREFIXES, Embedder, query_prefix
from codesearch.finetune import TrainConfig, find_runs, load_config, resolve_model, train
from codesearch.mining import add_negatives, mine_units
from codesearch.parser import parse_repository

BASE = "BAAI/bge-small-en-v1.5"


def test_config_file_runs_override_defaults(tmp_path):
    cfg = load_config("configs/finetune.toml", "random")
    assert cfg.negatives == "random" and cfg.base_model == BASE and cfg.seed == 0
    assert load_config("configs/finetune.toml", "hard").negatives == "hard"
    path = tmp_path / "c.toml"
    path.write_text('[defaults]\nbatch_size = 8\n[runs.a]\nbatch_size = 4\n[runs.b]\nbogus = 1\n')
    assert load_config(path, "a").batch_size == 4
    with pytest.raises(ValueError, match="bogus"):
        load_config(path, "b")
    with pytest.raises(ValueError, match="no run"):
        load_config(path, "c")


@pytest.fixture(scope="module")
def smoke(tmp_path_factory, fixture_root):
    """Two tiny training runs with the same seed on pairs mined from the fixture."""
    pairs, pool, _ = mine_units("mini_c", parse_repository(fixture_root), guard=None, val_fraction=0.0)
    pairs = add_negatives(pairs, pool, "hard", val_fraction=0.0)
    cfg = TrainConfig(base_model=BASE, negatives="hard", batch_size=4, max_train_pairs=8, max_seq_length=64,
                      eval_every=1, gradient_checkpointing=False, learning_rate=1e-4)
    store = tmp_path_factory.mktemp("mlruns")
    ids = [train(cfg, pairs, pairs[:6], tracking_dir=store, run_name=f"smoke{i}", log=lambda m: None)
           for i in range(2)]
    return store, ids


def _run(store, run_id):
    import mlflow

    from codesearch.finetune import tracking_uri

    mlflow.set_tracking_uri(tracking_uri(store))
    return mlflow.get_run(run_id), mlflow.MlflowClient()


def test_training_logs_params_curves_and_model(smoke):
    store, (run_id, _) = smoke
    run, client = _run(store, run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["negatives"] == "hard" and run.data.params["n_train"] == "8"
    assert run.data.params["steps"] == "2"
    assert [m.step for m in client.get_metric_history(run_id, "train_loss")] == [1, 2]
    assert [m.step for m in client.get_metric_history(run_id, "val_mrr")] == [0, 1, 2]
    assert 0 < run.data.metrics["final_val_mrr"] <= 1
    model_dir = resolve_model(f"mlflow:{run_id}", store)
    meta = json.loads(open(f"{model_dir}/codesearch.json").read())
    assert meta == {"base_model": BASE, "query_prefix": QUERY_PREFIXES[BASE], "negatives": "hard", "run_id": run_id}
    assert query_prefix(model_dir) == QUERY_PREFIXES[BASE]


def test_fixed_seed_makes_training_repeatable(smoke):
    store, ids = smoke
    curves = []
    for run_id in ids:
        _, client = _run(store, run_id)
        curves.append([m.value for m in client.get_metric_history(run_id, "train_loss")])
    assert curves[0] == pytest.approx(curves[1], rel=1e-4)


def test_best_run_is_selected_and_loads(smoke):
    store, ids = smoke
    best = find_runs(store)[0]
    assert best.info.run_id in ids
    path = resolve_model("mlflow:best", store)
    assert path == resolve_model(f"mlflow:{best.info.run_id}", store)
    assert resolve_model("mlflow:best-hard", store) == path
    with pytest.raises(LookupError):
        resolve_model("mlflow:best-random", store)
    with pytest.raises(LookupError):
        resolve_model("mlflow:0123456789abcdef", store)
    assert resolve_model(BASE, store) == BASE
    vec = Embedder(path).encode_query("grow the buffer")
    assert vec.shape == (384,)


def test_cli_query_and_serve_accept_a_fine_tuned_model(smoke, tmp_path, fixture_root, capsys):
    from codesearch.cli import main

    store, _ = smoke
    repo = tmp_path / "repo"
    shutil.copytree(fixture_root, repo)
    args = ["--tracking-dir", str(store), "--model", "mlflow:best"]
    assert main([*args, "query", str(repo), "double the bucket count and rehash", "-k", "3"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 3
    meta = json.loads((repo / ".codesearch" / "units.json").read_text())
    assert meta["model"] == resolve_model("mlflow:best", store)
    assert main(["--tracking-dir", str(tmp_path / "empty"), "--model", "mlflow:best", "serve", str(repo)]) == 2
