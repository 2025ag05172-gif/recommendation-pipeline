"""Recommenders. Every model exposes score(user_positions) -> users x items matrix of scores."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from src.common.context import Context


class BaseModel:
    name = "base"
    supports_rating = False

    def score(self, uidx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_rating(self, uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}


class PopularityRecommender(BaseModel):
    """Baseline: everyone gets the most rated items, ties broken by the Bayesian average rating."""
    name = "popularity"
    supports_rating = True

    def fit(self, ctx: Context):
        f = ctx.store.get_item_features(ctx.item_ids)
        self.popularity = f["item_popularity"].to_numpy(dtype=float)
        self.bayes = f["item_bayes_rating"].to_numpy(dtype=float)
        self.base_score = self.popularity + 1e-3 * self.bayes
        return self

    def score(self, uidx):
        return np.tile(self.base_score, (len(uidx), 1))

    def predict_rating(self, uidx, iidx):
        return self.bayes[iidx]

    def params(self):
        return {"ranking": "train rating count", "rating_prediction": "bayesian item mean"}


class SVDRecommender(BaseModel):
    """Matrix factorisation with truncated SVD.

    Ranking: SVD of the binary matrix of liked items (PureSVD), score = reconstructed value.
    Rating prediction: global mean + damped user and item biases + SVD of the residual matrix.
    """
    name = "svd_cf"
    supports_rating = True

    def __init__(self, n_factors=32, explicit_factors=16, damping=10.0, seed=42):
        self.n_factors, self.explicit_factors, self.damping, self.seed = n_factors, explicit_factors, damping, seed

    def fit(self, ctx: Context):
        R = ctx.train_matrix.tocsr()
        self.rating_min = float(ctx.cfg["ds"]["rating_min"])
        self.rating_max = float(ctx.cfg["ds"]["rating_max"])
        liked = R.copy()
        liked.data = (liked.data >= ctx.threshold).astype(float)
        liked.eliminate_zeros()
        f = max(1, min(self.n_factors, min(liked.shape) - 1))
        svd = TruncatedSVD(n_components=f, random_state=self.seed)
        self.U = svd.fit_transform(liked)
        self.Vt = svd.components_

        coo = R.tocoo()
        self.mu = float(coo.data.mean())
        n_u = np.diff(R.indptr)
        n_i = np.bincount(R.indices, minlength=R.shape[1])
        item_sum = np.asarray(R.sum(0)).ravel()
        self.b_i = (item_sum - self.mu * n_i) / (n_i + self.damping)
        user_sum = np.asarray(R.sum(1)).ravel()
        ib = sp.csr_matrix((self.b_i[coo.col], (coo.row, coo.col)), shape=R.shape)
        item_bias_sum = np.asarray(ib.sum(1)).ravel()
        self.b_u = (user_sum - self.mu * n_u - item_bias_sum) / (n_u + self.damping)
        resid = coo.data - self.mu - self.b_u[coo.row] - self.b_i[coo.col]
        res_m = sp.csr_matrix((resid, (coo.row, coo.col)), shape=R.shape)
        fe = max(1, min(self.explicit_factors, min(R.shape) - 1))
        svd_e = TruncatedSVD(n_components=fe, random_state=self.seed)
        self.Ue = svd_e.fit_transform(res_m)
        self.Vte = svd_e.components_
        return self

    def score(self, uidx):
        return self.U[uidx] @ self.Vt

    def predict_rating(self, uidx, iidx):
        pred = self.mu + self.b_u[uidx] + self.b_i[iidx] + (self.Ue[uidx] * self.Vte[:, iidx].T).sum(1)
        return np.clip(pred, self.rating_min, self.rating_max)

    def params(self):
        return {"n_factors": self.n_factors, "explicit_factors": self.explicit_factors,
                "bias_damping": self.damping, "seed": self.seed}


class ItemKNNRecommender(BaseModel):
    """Item-based collaborative filtering: score = sum of cosine similarities to the items the user liked.

    The similarity is shrunk towards zero for item pairs with little overlap (shrinkage in the denominator).
    """
    name = "item_knn_cf"

    def __init__(self, shrink=20.0):
        self.shrink = shrink

    def fit(self, ctx: Context):
        X = ctx.train_matrix.copy()
        X.data = (X.data >= ctx.threshold).astype(float)
        X.eliminate_zeros()
        self.X = X.tocsr()
        co = (self.X.T @ self.X).toarray()
        np.fill_diagonal(co, 0.0)
        n = np.asarray(self.X.sum(0)).ravel()
        self.S = co / (np.sqrt(np.outer(n, n)) + self.shrink)
        return self

    def score(self, uidx):
        return np.asarray(self.X[uidx] @ self.S)

    def params(self):
        return {"similarity": "cosine on liked items", "shrinkage": self.shrink}


class ContentBasedRecommender(BaseModel):
    """Cosine similarity between the user's liked-genre profile and each item's genre vector (idf weighted)."""
    name = "content_based"

    def fit(self, ctx: Context):
        G = ctx.store.item_genre_matrix(ctx.item_ids)
        df = G.sum(0)
        self.idf = np.log((1 + G.shape[0]) / (1 + df)) + 1.0
        W = G * self.idf
        self.item_vectors = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
        P = ctx.store.user_profile(ctx.user_ids) * self.idf
        self.user_vectors = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        pop = ctx.store.get_item_features(ctx.item_ids)["item_popularity"].to_numpy(dtype=float)
        self.tie_break = 1e-4 * pop / pop.max()
        return self

    def score(self, uidx):
        return self.user_vectors[uidx] @ self.item_vectors.T + self.tie_break

    def params(self):
        return {"item_representation": "idf weighted genre vector", "user_representation": "liked genre shares"}


class HybridRecommender(BaseModel):
    """Weighted sum of per-user z-scored scores from other models."""
    name = "hybrid"
    supports_rating = False     # ranking model; rating metrics are reported for the SVD component only

    def __init__(self, components: dict, weights: dict):
        self.components, self.weights = components, weights

    @staticmethod
    def _z(s):
        return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)

    def score(self, uidx):
        return sum(w * self._z(self.components[n].score(uidx)) for n, w in self.weights.items())

    def params(self):
        return {f"weight_{n}": w for n, w in self.weights.items()}
