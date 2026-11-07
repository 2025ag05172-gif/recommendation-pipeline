"""Stage 3 - data preparation: clean, standardise and filter the validated tables."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import get_logger, path

log = get_logger("preparation")
YEAR_RE = r"\((\d{4})\)\s*$"


def _kcore(inter: pd.DataFrame, min_u: int, min_i: int) -> pd.DataFrame:
    while True:
        uc = inter["user_id"].map(inter["user_id"].value_counts())
        ic = inter["item_id"].map(inter["item_id"].value_counts())
        keep = (uc >= min_u) & (ic >= min_i)
        if keep.all():
            return inter
        inter = inter[keep]


def prepare(cfg: dict) -> dict:
    interim, processed, reports = path(cfg, "interim"), path(cfg, "processed"), path(cfg, "reports")
    users = pd.read_parquet(interim / "users.parquet")
    items = pd.read_parquet(interim / "items.parquet")
    inter = pd.read_parquet(interim / "interactions.parquet")
    stats = {"interactions_in": len(inter)}

    # duplicates: keep the most recent rating of a user for an item
    inter = inter.sort_values("timestamp", kind="stable")
    before = len(inter)
    inter = inter.drop_duplicates(["user_id", "item_id"], keep="last")
    stats["duplicate_interactions_removed"] = before - len(inter)
    users = users.drop_duplicates("user_id", keep="first")
    items = items.drop_duplicates("item_id", keep="first")

    # items: standard format for title, genres and release year
    items = items.copy()
    items["title"] = items["title"].str.replace(r"\s+", " ", regex=True).str.strip()
    stats["missing_genres_filled"] = int(items["genres"].isna().sum())
    items["genres"] = items["genres"].fillna("Unknown").map(
        lambda s: "|".join(sorted({g.strip() for g in s.split("|") if g.strip()})) or "Unknown")
    year = items["title"].str.extract(YEAR_RE)[0].astype(float)
    if "release_date" in items:
        year = year.fillna(pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce").dt.year)
    stats["release_year_imputed"] = int(year.isna().sum())
    items["release_year"] = year.fillna(year.median()).astype(int)
    items = items[["item_id", "title", "genres", "release_year"]]

    inter["event_time"] = pd.to_datetime(inter["timestamp"], unit="s", utc=True)

    # drop sparse users and items so that every kept user has some history
    inter = _kcore(inter, cfg["preparation"]["min_user_interactions"], cfg["preparation"]["min_item_interactions"])
    inter = inter[inter["user_id"].isin(users["user_id"]) & inter["item_id"].isin(items["item_id"])]
    users = users[users["user_id"].isin(inter["user_id"])].reset_index(drop=True)
    items = items[items["item_id"].isin(inter["item_id"])].reset_index(drop=True)
    inter = inter.sort_values("timestamp", kind="stable").reset_index(drop=True)

    stats.update(interactions_out=len(inter), users_out=len(users), items_out=len(items),
                 share_kept=round(len(inter) / stats["interactions_in"], 4),
                 min_user_interactions=cfg["preparation"]["min_user_interactions"],
                 min_item_interactions=cfg["preparation"]["min_item_interactions"])
    users.to_parquet(processed / "users.parquet", index=False)
    items.to_parquet(processed / "items.parquet", index=False)
    inter.to_parquet(processed / "interactions.parquet", index=False)
    (reports / "preparation_summary.json").write_text(json.dumps(stats, indent=2))
    log.info("prepared: %s", stats)
    return stats
