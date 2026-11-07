"""Ranking and rating metrics. All functions work on plain numpy arrays so they are easy to test."""
from __future__ import annotations

import numpy as np


def top_k(scores: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best unseen items per row, best first."""
    s = np.where(seen, -np.inf, scores)
    k = min(k, s.shape[1])
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def precision_recall_ndcg(rec: np.ndarray, relevant: list[set], k: int):
    n = len(rec)
    p, r, nd = np.zeros(n), np.zeros(n), np.zeros(n)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for row in range(n):
        rel = relevant[row]
        hits = np.array([item in rel for item in rec[row, :k]], dtype=float)
        p[row] = hits.sum() / k
        r[row] = hits.sum() / len(rel)
        idcg = discounts[:min(len(rel), k)].sum()
        nd[row] = (hits * discounts[:len(hits)]).sum() / idcg
    return p, r, nd


def coverage(rec: np.ndarray, n_items: int) -> float:
    return len(np.unique(rec)) / n_items


def diversity(rec: np.ndarray, item_vectors: np.ndarray) -> float:
    """Intra-list diversity: 1 - mean pairwise cosine similarity of the genre vectors in each list."""
    norm = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    unit = item_vectors / np.maximum(norm, 1e-12)
    k = rec.shape[1]
    if k < 2:
        return 0.0
    vecs = unit[rec]                                   # users x k x genres
    sim = np.einsum("ukg,ulg->ukl", vecs, vecs)
    off_diag = (sim.sum(axis=(1, 2)) - np.trace(sim, axis1=1, axis2=2)) / (k * (k - 1))
    return float((1.0 - off_diag).mean())


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
