"""Shared view of the training data: id <-> matrix position maps, the train matrix and the feature store."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common.config import path
from src.features.feature_store import FeatureStore


@dataclass
class Context:
    user_ids: list
    item_ids: list
    u_idx: pd.Series
    i_idx: pd.Series
    train_matrix: sp.csr_matrix
    store: FeatureStore | None
    cfg: dict

    def __post_init__(self):
        m = self.train_matrix
        # structure only: which user rated which item, even when the rating value is 0
        self.seen = sp.csr_matrix((np.ones(m.nnz), m.indices, m.indptr), shape=m.shape)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def threshold(self) -> float:
        return self.cfg["ds"]["relevance_threshold"]

    def with_matrix(self, m: sp.csr_matrix) -> "Context":
        return replace(self, train_matrix=m)

    def matrix_from(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Rating matrix for any slice of the train data, in this context's row / column positions."""
        shift = 1 - self.cfg["ds"]["rating_min"]
        m = sp.csr_matrix(((df["rating"].to_numpy() + shift),
                           (df["user_id"].map(self.u_idx).to_numpy(), df["item_id"].map(self.i_idx).to_numpy())),
                          shape=(self.n_users, self.n_items))
        m.data = m.data - shift
        return m


def load_context(cfg: dict, with_store: bool = True) -> Context:
    processed = path(cfg, "processed")
    users = pd.read_csv(processed / "user_index.csv", dtype={"user_id": str})
    items = pd.read_csv(processed / "item_index.csv", dtype={"item_id": str})
    matrix = sp.load_npz(processed / "interaction_matrix.npz").tocsr()
    matrix.data = matrix.data - (1 - cfg["ds"]["rating_min"])
    return Context(users["user_id"].tolist(), items["item_id"].tolist(),
                   pd.Series(users["idx"].to_numpy(), index=users["user_id"]),
                   pd.Series(items["idx"].to_numpy(), index=items["item_id"]),
                   matrix, FeatureStore(cfg) if with_store else None, cfg)
