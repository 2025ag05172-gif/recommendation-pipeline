import numpy as np
import pytest

from src.evaluation import metrics as M


def test_top_k_skips_seen_items_and_orders_by_score():
    scores = np.array([[0.9, 0.8, 0.7, 0.1]])
    seen = np.array([[True, False, False, False]])
    assert M.top_k(scores, seen, 2).tolist() == [[1, 2]]


def test_precision_recall_ndcg_hand_computed():
    rec = np.array([[3, 1, 0, 2]])
    rel = [{1, 2}]
    p, r, n = M.precision_recall_ndcg(rec, rel, 4)
    assert p[0] == pytest.approx(0.5)
    assert r[0] == pytest.approx(1.0)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)       # hits at rank 2 and rank 4
    idcg = 1 + 1 / np.log2(3)
    assert n[0] == pytest.approx(dcg / idcg)


def test_perfect_ranking_has_ndcg_one():
    p, r, n = M.precision_recall_ndcg(np.array([[5, 6, 7]]), [{5, 6}], 3)
    assert n[0] == pytest.approx(1.0)
    assert r[0] == 1.0


def test_coverage():
    rec = np.array([[0, 1], [1, 2]])
    assert M.coverage(rec, 10) == pytest.approx(0.3)


def test_diversity_identical_items_is_zero_and_orthogonal_is_one():
    same = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert M.diversity(np.array([[0, 1]]), same) == pytest.approx(0.0)
    orth = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert M.diversity(np.array([[0, 1]]), orth) == pytest.approx(1.0)


def test_rmse_mae():
    assert M.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert M.mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)
