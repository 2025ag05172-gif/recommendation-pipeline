"""A small file-based feature layer.

Training, evaluation and any later prediction code read features only through this
class, so a feature has one definition (registry.py) and one implementation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import path


class FeatureStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        d = path(cfg, "features")
        meta = json.loads((d / "feature_meta.json").read_text())
        self.cutoff = pd.Timestamp(meta["cutoff_utc"], tz="UTC")
        self.genres: list[str] = meta["genres"]
        self.genre_cols = [f"genre_{g}" for g in self.genres]
        self.share_cols = [f"user_genre_share_{g}" for g in self.genres]
        self.users = pd.read_parquet(d / "user_features.parquet").set_index("user_id")
        self.items = pd.read_parquet(d / "item_features.parquet").set_index("item_id")
        self.profile = pd.read_parquet(d / "user_genre_profile.parquet").set_index("user_id")
        self.half_life = cfg["features"]["recency_half_life_days"]

    def get_user_features(self, user_ids) -> pd.DataFrame:
        return self.users.reindex(user_ids)

    def get_item_features(self, item_ids) -> pd.DataFrame:
        return self.items.reindex(item_ids)

    def item_genre_matrix(self, item_ids) -> np.ndarray:
        return self.items.reindex(item_ids)[self.genre_cols].fillna(0).to_numpy(dtype=float)

    def user_profile(self, user_ids) -> np.ndarray:
        return self.profile.reindex(user_ids)[self.share_cols].fillna(0).to_numpy(dtype=float)

    def interaction_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """pairs needs user_id and item_id. rating and event_time are optional (training rows have them)."""
        out = pairs[["user_id", "item_id"]].reset_index(drop=True).copy()
        uf = self.get_user_features(out["user_id"])
        itf = self.get_item_features(out["item_id"])
        prof = self.user_profile(out["user_id"])
        genres = self.item_genre_matrix(out["item_id"])
        out["user_genre_affinity"] = (prof * genres).sum(1) / np.maximum(genres.sum(1), 1)
        if "rating" in pairs:
            rating = pairs["rating"].reset_index(drop=True)
            out["rating_user_deviation"] = rating - uf["user_avg_rating"].to_numpy()
            out["rating_item_deviation"] = rating - itf["item_avg_rating"].to_numpy()
        else:
            out["rating_user_deviation"] = np.nan
            out["rating_item_deviation"] = np.nan
        if "event_time" in pairs:
            days = (self.cutoff - pairs["event_time"].reset_index(drop=True)).dt.total_seconds() / 86400
            out["days_since_interaction"] = days
            out["recency_weight"] = 0.5 ** (days / self.half_life)
        else:
            out["days_since_interaction"] = np.nan
            out["recency_weight"] = np.nan
        return out
