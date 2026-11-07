"""Stage 4 - transformation: time-based split and derived datasets.

Outputs (data/processed):
    train.parquet, test.parquet        time-based split
    user_activity.parquet              activity summary per user (train only)
    item_popularity.parquet            popularity summary per item (train only)
    interaction_matrix.npz             sparse user x item matrix of train ratings
    user_index.csv, item_index.csv     row / column positions of that matrix
"""
from __future__ import annotations

import json

import pandas as pd
import scipy.sparse as sp

from src.common.config import get_logger, path

log = get_logger("transformation")


def time_split(inter: pd.DataFrame, test_fraction: float):
    cutoff = int(inter["timestamp"].quantile(1 - test_fraction))
    train = inter[inter["timestamp"] < cutoff].copy()
    test = inter[inter["timestamp"] >= cutoff].copy()
    return train, test, cutoff


def transform(cfg: dict) -> dict:
    processed, reports = path(cfg, "processed"), path(cfg, "reports")
    inter = pd.read_parquet(processed / "interactions.parquet")
    thr = cfg["ds"]["relevance_threshold"]

    train, test, cutoff = time_split(inter, cfg["split"]["test_fraction"])
    n_test_raw = len(test)
    # users and items that never appear in train cannot be scored by the models
    test = test[test["user_id"].isin(train["user_id"]) & test["item_id"].isin(train["item_id"])]
    log.info("split at %s: train %d, test %d (%d dropped as cold start)",
             pd.to_datetime(cutoff, unit="s"), len(train), len(test), n_test_raw - len(test))

    users = sorted(train["user_id"].unique(), key=int)
    items = sorted(train["item_id"].unique(), key=int)
    u_idx = pd.Series(range(len(users)), index=users)
    i_idx = pd.Series(range(len(items)), index=items)
    shift = 1 - cfg["ds"]["rating_min"]    # stored as rating + shift so that a rating of 0 is not lost as a sparse zero
    matrix = sp.csr_matrix((train["rating"].to_numpy() + shift,
                            (train["user_id"].map(u_idx).to_numpy(), train["item_id"].map(i_idx).to_numpy())),
                           shape=(len(users), len(items)))

    last = train["event_time"].max()
    g = train.groupby("user_id")
    user_activity = pd.DataFrame({
        "n_interactions": g.size(),
        "n_positive": g["rating"].apply(lambda s: int((s >= thr).sum())),
        "avg_rating": g["rating"].mean().round(4),
        "first_interaction": g["event_time"].min(),
        "last_interaction": g["event_time"].max(),
    })
    user_activity["active_days"] = (user_activity["last_interaction"] - user_activity["first_interaction"]).dt.days
    user_activity = user_activity.reset_index()
    gi = train.groupby("item_id")
    item_pop = pd.DataFrame({"n_interactions": gi.size(), "n_positive": gi["rating"].apply(lambda s: int((s >= thr).sum())),
                             "avg_rating": gi["rating"].mean().round(4)})
    item_pop["popularity_rank"] = item_pop["n_interactions"].rank(ascending=False, method="min").astype(int)
    item_pop = item_pop.reset_index().sort_values("popularity_rank")

    train.to_parquet(processed / "train.parquet", index=False)
    test.to_parquet(processed / "test.parquet", index=False)
    user_activity.to_parquet(processed / "user_activity.parquet", index=False)
    item_pop.to_parquet(processed / "item_popularity.parquet", index=False)
    sp.save_npz(processed / "interaction_matrix.npz", matrix)
    pd.DataFrame({"user_id": users, "idx": range(len(users))}).to_csv(processed / "user_index.csv", index=False)
    pd.DataFrame({"item_id": items, "idx": range(len(items))}).to_csv(processed / "item_index.csv", index=False)

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    summary = {
        "split_cutoff_unix": cutoff,
        "split_cutoff_utc": str(pd.to_datetime(cutoff, unit="s")),
        "train_rows": len(train), "test_rows": len(test), "test_rows_dropped_cold_start": n_test_raw - len(test),
        "train_period": [str(train["event_time"].min()), str(last)],
        "test_period": [str(test["event_time"].min()), str(test["event_time"].max())],
        "matrix_shape": list(matrix.shape), "matrix_density": round(density, 5),
        "test_users": int(test["user_id"].nunique()),
    }
    (reports / "split_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("matrix %s, density %.4f", matrix.shape, density)
    return summary
