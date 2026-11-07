"""Ranking evaluation shared by the tuning step (validation slice) and the final evaluation (test set)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.context import Context
from src.evaluation import metrics as M


def relevant_sets(df: pd.DataFrame, ctx: Context) -> dict[int, set]:
    """User position -> set of item positions the user liked in df."""
    liked = df[df["rating"] >= ctx.threshold]
    u = liked["user_id"].map(ctx.u_idx)
    i = liked["item_id"].map(ctx.i_idx)
    ok = u.notna() & i.notna()
    out: dict[int, set] = {}
    for uu, ii in zip(u[ok].astype(int), i[ok].astype(int)):
        out.setdefault(uu, set()).add(ii)
    return out


def evaluate_ranking(model, ctx: Context, relevant: dict[int, set], ks, item_vectors: np.ndarray, batch: int = 500) -> dict:
    users = np.array(sorted(relevant))
    kmax = max(ks)
    recs = []
    for start in range(0, len(users), batch):
        u = users[start:start + batch]
        scores = model.score(u)
        seen = ctx.seen[u].toarray() > 0
        recs.append(M.top_k(scores, seen, kmax))
    rec = np.vstack(recs)
    rel_list = [relevant[u] for u in users]
    out = {"n_eval_users": int(len(users))}
    for k in ks:
        p, r, nd = M.precision_recall_ndcg(rec[:, :k], rel_list, k)
        out[f"precision_at_{k}"] = float(p.mean())
        out[f"recall_at_{k}"] = float(r.mean())
        out[f"ndcg_at_{k}"] = float(nd.mean())
        out[f"coverage_at_{k}"] = M.coverage(rec[:, :k], ctx.n_items)
        out[f"diversity_at_{k}"] = M.diversity(rec[:, :k], item_vectors)
    return out
