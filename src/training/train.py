"""Stage 6 - model training with MLflow tracking.

Each model gets its own MLflow run (parameters, model file, tuning table). The run id is saved to
models/<name>.run.json so the evaluation stage can add metrics to the same run.
"""
from __future__ import annotations

import hashlib
import json

import joblib
import mlflow
import pandas as pd

from src.common.config import ROOT, get_logger, path
from src.common.context import Context, load_context
from src.evaluation.ranking_eval import evaluate_ranking, relevant_sets
from src.training.models import (ContentBasedRecommender, HybridRecommender, ItemKNNRecommender,
                                 PopularityRecommender, SVDRecommender)

log = get_logger("training")
MODEL_ORDER = ["popularity", "svd_cf", "item_knn_cf", "content_based", "hybrid"]


def setup_mlflow(cfg: dict) -> None:
    root = cfg["_root"]
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(root / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)


def _data_version(cfg: dict) -> str:
    return hashlib.sha256((path(cfg, "processed") / "train.parquet").read_bytes()).hexdigest()[:12]


def _validation_setup(ctx: Context, train: pd.DataFrame, fraction: float):
    """Older part of train becomes the fit data, the most recent slice the validation set."""
    cut = train["timestamp"].quantile(1 - fraction)
    sub, val = train[train["timestamp"] < cut], train[train["timestamp"] >= cut]
    sub_ctx = ctx.with_matrix(ctx.matrix_from(sub))
    val = val[val["user_id"].isin(sub["user_id"]) & val["item_id"].isin(sub["item_id"])]
    return sub_ctx, relevant_sets(val, ctx), ctx.store.item_genre_matrix(ctx.item_ids)


def tune_svd(ctx: Context, train: pd.DataFrame, cfg: dict):
    """Pick the number of factors on the validation slice (the test set is never touched)."""
    t = cfg["training"]
    sub_ctx, rel, vectors = _validation_setup(ctx, train, t["validation_fraction"])
    rows = []
    for f in t["svd"]["n_factors_grid"]:
        model = SVDRecommender(f, t["svd"]["explicit_factors"], t["svd"]["bias_damping"], cfg["seed"]).fit(sub_ctx)
        res = evaluate_ranking(model, sub_ctx, rel, [10], vectors)
        rows.append({"n_factors": f, "val_ndcg_at_10": res["ndcg_at_10"], "val_recall_at_10": res["recall_at_10"]})
        log.info("tuning svd n_factors=%d -> val NDCG@10 %.4f", f, res["ndcg_at_10"])
    table = pd.DataFrame(rows)
    return int(table.sort_values("val_ndcg_at_10", ascending=False).iloc[0]["n_factors"]), table


def tune_knn(ctx: Context, train: pd.DataFrame, cfg: dict):
    t = cfg["training"]
    sub_ctx, rel, vectors = _validation_setup(ctx, train, t["validation_fraction"])
    rows = []
    for sh in t["item_knn"]["shrinkage_grid"]:
        res = evaluate_ranking(ItemKNNRecommender(sh).fit(sub_ctx), sub_ctx, rel, [10], vectors)
        rows.append({"shrinkage": sh, "val_ndcg_at_10": res["ndcg_at_10"], "val_recall_at_10": res["recall_at_10"]})
        log.info("tuning item_knn shrinkage=%s -> val NDCG@10 %.4f", sh, res["ndcg_at_10"])
    table = pd.DataFrame(rows)
    return float(table.sort_values("val_ndcg_at_10", ascending=False).iloc[0]["shrinkage"]), table


def train_model(name: str, cfg: dict) -> dict:
    models_dir, reports = path(cfg, "models"), path(cfg, "reports")
    ctx = load_context(cfg)
    extra_artifacts = []
    tuning = None

    if name == "popularity":
        model = PopularityRecommender().fit(ctx)
    elif name == "svd_cf":
        train = pd.read_parquet(path(cfg, "processed") / "train.parquet")
        best, tuning = tune_svd(ctx, train, cfg)
        t = cfg["training"]["svd"]
        model = SVDRecommender(best, t["explicit_factors"], t["bias_damping"], cfg["seed"]).fit(ctx)
        tuning.to_csv(reports / "svd_tuning.csv", index=False)
        extra_artifacts.append(reports / "svd_tuning.csv")
    elif name == "item_knn_cf":
        train = pd.read_parquet(path(cfg, "processed") / "train.parquet")
        best, tuning = tune_knn(ctx, train, cfg)
        model = ItemKNNRecommender(best).fit(ctx)
        tuning.to_csv(reports / "item_knn_tuning.csv", index=False)
        extra_artifacts.append(reports / "item_knn_tuning.csv")
    elif name == "content_based":
        model = ContentBasedRecommender().fit(ctx)
    elif name == "hybrid":
        weights = cfg["training"]["hybrid"]["weights"]
        parts = {n: joblib.load(models_dir / f"{n}.joblib") for n in weights}   # needs the other models first
        model = HybridRecommender(parts, weights)
    else:
        raise ValueError(name)

    model_path = models_dir / f"{name}.joblib"
    joblib.dump(model, model_path)

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"stage": "training", "model": name, "dataset": cfg["dataset"]["name"]})
        mlflow.log_params({"seed": cfg["seed"], "relevance_threshold": ctx.threshold, "n_users": ctx.n_users,
                           "n_items": ctx.n_items, "train_interactions": ctx.train_matrix.nnz,
                           "data_version": _data_version(cfg), **model.params()})
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(cfg["_root"] / "configs" / "config.yaml"), artifact_path="config")
        for a in extra_artifacts:
            mlflow.log_artifact(str(a))
        run_id = run.info.run_id
    info = {"model": name, "run_id": run_id, "params": model.params()}
    (models_dir / f"{name}.run.json").write_text(json.dumps(info, indent=2))
    log.info("trained %s (mlflow run %s)", name, run_id[:8])
    return info


def train_all(cfg: dict) -> list[dict]:
    return [train_model(n, cfg) for n in MODEL_ORDER]
