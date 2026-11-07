"""Stage 7 - model evaluation on the time-based test set; results are written to reports/ and to MLflow."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.common.config import get_logger, path
from src.common.context import load_context
from src.evaluation import metrics as M
from src.evaluation.ranking_eval import evaluate_ranking, relevant_sets
from src.training.train import MODEL_ORDER, setup_mlflow

log = get_logger("evaluation")


def evaluate(cfg: dict) -> pd.DataFrame:
    models_dir, reports, processed = path(cfg, "models"), path(cfg, "reports"), path(cfg, "processed")
    ks, hk = cfg["evaluation"]["ks"], cfg["evaluation"]["headline_k"]
    ctx = load_context(cfg)
    test = pd.read_parquet(processed / "test.parquet")
    rel = relevant_sets(test, ctx)
    vectors = ctx.store.item_genre_matrix(ctx.item_ids)
    log.info("evaluating on %d test interactions, %d users with at least one liked item", len(test), len(rel))

    t_u = test["user_id"].map(ctx.u_idx).to_numpy()
    t_i = test["item_id"].map(ctx.i_idx).to_numpy()
    rows, models = [], {}
    for name in MODEL_ORDER:
        model = joblib.load(models_dir / f"{name}.joblib")
        models[name] = model
        res = evaluate_ranking(model, ctx, rel, ks, vectors)
        if model.supports_rating:
            pred = model.predict_rating(t_u, t_i)
            res["rmse"], res["mae"] = M.rmse(test["rating"], pred), M.mae(test["rating"], pred)
        rows.append({"model": name, **res})
        log.info("%s: P@%d %.4f R@%d %.4f NDCG@%d %.4f", name, hk, res[f"precision_at_{hk}"], hk,
                 res[f"recall_at_{hk}"], hk, res[f"ndcg_at_{hk}"])
    table = pd.DataFrame(rows)
    table.to_csv(reports / "model_comparison.csv", index=False)
    _write_markdown(table, ks, hk, reports / "model_comparison.md", cfg)
    _plot(table, hk, reports / "model_comparison.png")
    _sample_recommendations(models, table, ctx, rel, cfg, reports)

    setup_mlflow(cfg)
    for _, r in table.iterrows():
        run_id = json.loads((models_dir / f"{r['model']}.run.json").read_text())["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("stage", "evaluated")
            mlflow.log_metrics({k: float(v) for k, v in r.items() if k != "model" and pd.notna(v)})
            mlflow.log_artifact(str(reports / "model_comparison.csv"), artifact_path="evaluation")
            mlflow.log_artifact(str(reports / "model_comparison.png"), artifact_path="evaluation")
    return table


def _write_markdown(table, ks, hk, out, cfg):
    def fmt(v):
        return "n/a" if pd.isna(v) else f"{v:.4f}"
    lines = ["# Model comparison", "",
             f"Dataset `{cfg['dataset']['name']}`, time-based split, a rating >= {cfg['ds']['relevance_threshold']} counts as relevant. "
             f"Evaluated on {int(table['n_eval_users'].iloc[0])} test users. Items a user already rated in train are never recommended.", ""]
    for k in sorted(ks, reverse=True):
        lines += [f"## K = {k}", "", "| model | Precision@K | Recall@K | NDCG@K | Coverage | Diversity | RMSE | MAE |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in table.iterrows():
            lines.append(f"| {r['model']} | {fmt(r[f'precision_at_{k}'])} | {fmt(r[f'recall_at_{k}'])} | {fmt(r[f'ndcg_at_{k}'])} | "
                         f"{fmt(r[f'coverage_at_{k}'])} | {fmt(r[f'diversity_at_{k}'])} | {fmt(r.get('rmse'))} | {fmt(r.get('mae'))} |")
        lines.append("")
    lines += ["Coverage = share of the catalog that appears in at least one list. Diversity = 1 - mean pairwise genre "
              "cosine similarity inside a list. RMSE and MAE are computed on all test ratings by models that predict ratings.", ""]
    out.write_text("\n".join(lines))


def _plot(table, hk, out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (col, title) in zip(ax, [(f"precision_at_{hk}", f"Precision@{hk}"), (f"ndcg_at_{hk}", f"NDCG@{hk}"),
                                    (f"coverage_at_{hk}", f"Coverage@{hk}")]):
        a.bar(table["model"], table[col], color="#3a6ea5")
        a.set_title(title)
        a.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _sample_recommendations(models, table, ctx, rel, cfg, reports, n_users=3, k=10):
    best = table.sort_values(f"ndcg_at_{cfg['evaluation']['headline_k']}", ascending=False).iloc[0]["model"]
    items = pd.read_parquet(path(cfg, "processed") / "items.parquet").set_index("item_id")
    train = pd.read_parquet(path(cfg, "processed") / "train.parquet")
    users = np.array(sorted(rel))[:: max(1, len(rel) // n_users)][:n_users]
    scores = models[best].score(users)
    rec = M.top_k(scores, ctx.seen[users].toarray() > 0, k)
    rows = []
    for u, r in zip(users, rec):
        uid = ctx.user_ids[u]
        liked = train[(train["user_id"] == uid) & (train["rating"] >= ctx.threshold)].sort_values("timestamp").tail(3)
        for rank, i in enumerate(r, 1):
            iid = ctx.item_ids[i]
            rows.append({"model": best, "user_id": uid, "rank": rank, "item_id": iid, "title": items.loc[iid, "title"],
                         "genres": items.loc[iid, "genres"], "hit_in_test": bool(i in rel[u]),
                         "recently_liked": "; ".join(items.loc[liked["item_id"], "title"]) if rank == 1 else ""})
    pd.DataFrame(rows).to_csv(reports / "sample_recommendations.csv", index=False)
