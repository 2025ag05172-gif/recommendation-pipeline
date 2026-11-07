"""Stage 5 - feature management: compute user, item and interaction features from the train split."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import get_logger, path
from src.features.feature_store import FeatureStore
from src.features.registry import expected_columns, sanitize, write_catalog

log = get_logger("features")


def build_features(cfg: dict) -> dict:
    processed, feats, reports = path(cfg, "processed"), path(cfg, "features"), path(cfg, "reports")
    train = pd.read_parquet(processed / "train.parquet")
    items = pd.read_parquet(processed / "items.parquet")
    split = json.loads((reports / "split_summary.json").read_text())
    cutoff = pd.Timestamp(split["split_cutoff_unix"], unit="s", tz="UTC")
    thr = cfg["ds"]["relevance_threshold"]
    m = cfg["features"]["bayes_prior_strength"]

    raw_genres = sorted({g for s in items["genres"] for g in s.split("|")})
    genres = [sanitize(g) for g in raw_genres]
    items = items[items["item_id"].isin(train["item_id"])].reset_index(drop=True)
    G = np.array([[1 if g in s.split("|") else 0 for g in raw_genres] for s in items["genres"]], dtype=float)
    item_pos = pd.Series(range(len(items)), index=items["item_id"])

    train = train.copy()
    train["liked"] = train["rating"] >= thr

    # genre profile of every user: liked items first, all items if the user liked nothing
    users = sorted(train["user_id"].unique(), key=int)
    user_pos = pd.Series(range(len(users)), index=users)
    r, c = train["user_id"].map(user_pos).to_numpy(), train["item_id"].map(item_pos).to_numpy()
    A_all = sp.csr_matrix((np.ones(len(train)), (r, c)), shape=(len(users), len(items)))
    lk = train["liked"].to_numpy()
    A_pos = sp.csr_matrix((np.ones(lk.sum()), (r[lk], c[lk])), shape=(len(users), len(items)))
    has_pos = np.asarray(A_pos.sum(1)).ravel() > 0
    counts = np.where(has_pos[:, None], A_pos @ G, A_all @ G)
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    profile = pd.DataFrame(share, columns=[f"user_genre_share_{g}" for g in genres])
    profile.insert(0, "user_id", users)

    # user features
    g = train.groupby("user_id")
    uf = pd.DataFrame({
        "user_activity_count": g.size(),
        "user_avg_rating": g["rating"].mean(),
        "user_rating_std": g["rating"].std(ddof=0).fillna(0.0),
        "user_positive_ratio": g["liked"].mean(),
        "user_days_since_last_interaction": (cutoff - g["event_time"].max()).dt.total_seconds() / 86400,
        "user_active_span_days": (g["event_time"].max() - g["event_time"].min()).dt.days,
    }).reindex(users)
    uf["user_preferred_genre"] = np.array(genres)[share.argmax(1)]
    uf["user_distinct_genres"] = (share > 0).sum(1)
    uf = uf.reset_index(names="user_id")

    # item features
    gi = train.groupby("item_id")
    n = gi.size()
    mean = gi["rating"].mean()
    itf = pd.DataFrame({
        "item_popularity": n,
        "item_avg_rating": mean,
        "item_bayes_rating": (n * mean + m * train["rating"].mean()) / (n + m),
        "item_positive_ratio": gi["liked"].mean(),
        "item_days_since_last_interaction": (cutoff - gi["event_time"].max()).dt.total_seconds() / 86400,
    }).reindex(items["item_id"])
    itf["item_release_year"] = items["release_year"].to_numpy()
    itf["item_age_years"] = (cutoff.year - itf["item_release_year"]).clip(lower=0)
    itf["item_genre_count"] = G.sum(1).astype(int)
    for k, gname in enumerate(genres):
        itf[f"genre_{gname}"] = G[:, k].astype("int8")
    itf = itf.reset_index(names="item_id")

    uf.to_parquet(feats / "user_features.parquet", index=False)
    itf.to_parquet(feats / "item_features.parquet", index=False)
    profile.to_parquet(feats / "user_genre_profile.parquet", index=False)
    (feats / "feature_meta.json").write_text(json.dumps(
        {"cutoff_utc": str(cutoff.tz_localize(None)), "genres": genres, "relevance_threshold": thr}, indent=2))

    # interaction features go through the same FeatureStore code that is used at prediction time
    store = FeatureStore(cfg)
    inter = store.interaction_features(train[["user_id", "item_id", "rating", "event_time"]])
    inter.to_parquet(feats / "interaction_features.parquet", index=False)

    docs = cfg["_root"] / "docs"
    docs.mkdir(exist_ok=True)
    write_catalog(docs / "feature_catalog.csv", docs / "feature_catalog.md")

    for table, df in (("user_features", uf), ("item_features", itf), ("user_genre_profile", profile),
                      ("interaction_features", inter)):
        missing = expected_columns(table, genres) - set(df.columns)
        extra = set(df.columns) - expected_columns(table, genres) - {"user_id", "item_id"}
        if missing or extra:
            raise RuntimeError(f"{table} does not match the registry: missing {missing}, extra {extra}")
    summary = {"users": len(uf), "items": len(itf), "interaction_rows": len(inter), "genres": len(genres),
               "feature_columns": {"user": uf.shape[1] - 1, "item": itf.shape[1] - 1,
                                   "user_genre_profile": profile.shape[1] - 1, "interaction": inter.shape[1] - 2}}
    (reports / "feature_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("features built: %s", summary)
    return summary
