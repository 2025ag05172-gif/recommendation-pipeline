"""Stage 1 - ingestion.

Downloads a public dataset (or builds a synthetic one) and writes three raw CSV
files with a common layout:

    users.csv         user_id, [attributes]
    items.csv         item_id, title, genres  (genres separated by '|')
    interactions.csv  user_id, item_id, rating, timestamp (unix seconds)

Everything is written as text, nothing is cleaned here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import get_logger, path

log = get_logger("ingestion")

ML_GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
             "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
             "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def _download(url: str, dest: Path, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("already downloaded: %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s", url)
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            return
        except Exception as exc:  # network errors are retried, final one is raised
            log.warning("download failed (%s), attempt %d/%d", exc, attempt, retries)
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _load_movietweetings(cfg: dict, src: Path):
    base = cfg["ds"]["base_url"].rstrip("/")
    files = ["users.dat", "movies.dat", "ratings.dat"]
    for fn in files:
        _download(f"{base}/{fn}", src / fn)
    kw = dict(sep="::", engine="python", dtype=str, keep_default_na=False, na_values=[""],
              quoting=csv.QUOTE_NONE, encoding="utf-8")
    users = pd.read_csv(src / "users.dat", names=["user_id", "twitter_id"], **kw)
    items = pd.read_csv(src / "movies.dat", names=["item_id", "title", "genres"], **kw)
    inter = pd.read_csv(src / "ratings.dat", names=["user_id", "item_id", "rating", "timestamp"], **kw)
    return users, items, inter, [src / f for f in files]


def _load_movielens100k(cfg: dict, src: Path):
    zpath = src / "ml-100k.zip"
    _download(cfg["ds"]["url"], zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(src)
    d = src / "ml-100k"
    inter = pd.read_csv(d / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"], dtype=str)
    users = pd.read_csv(d / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"], dtype=str)
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + ML_GENRES
    raw_items = pd.read_csv(d / "u.item", sep="|", names=cols, dtype=str, encoding="latin-1")
    flags = raw_items[ML_GENRES].astype(int).to_numpy()
    names = np.array(ML_GENRES)
    genres = ["|".join(names[row == 1]) or np.nan for row in flags]
    items = pd.DataFrame({"item_id": raw_items["item_id"], "title": raw_items["title"],
                          "genres": genres, "release_date": raw_items["release_date"]})
    return users, items, inter, [zpath]


def _make_synthetic(cfg: dict):
    ds, rng = cfg["ds"], np.random.default_rng(cfg["seed"])
    genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Documentary"]
    nu, ni, nr = ds["n_users"], ds["n_items"], ds["n_ratings"]
    item_g = rng.random((ni, len(genres))) < 0.2
    item_g[np.arange(ni), rng.integers(0, len(genres), ni)] = True
    taste = rng.dirichlet(np.ones(len(genres)) * 0.4, nu)
    pop = rng.pareto(1.2, ni) + 1
    pop = pop / pop.sum()
    users_i = rng.integers(0, nu, nr * 2)
    items_i = rng.choice(ni, nr * 2, p=pop)
    pairs = pd.DataFrame({"u": users_i, "i": items_i}).drop_duplicates().head(nr)
    affinity = (taste[pairs.u] * item_g[pairs.i]).sum(1)
    score = 2.0 + 6 * affinity + rng.normal(0, 0.6, len(pairs))
    rating = np.clip(np.round(score), ds["rating_min"], ds["rating_max"]).astype(int)
    t0 = int(pd.Timestamp("2018-01-01").timestamp())
    ts = t0 + rng.integers(0, 365 * 24 * 3600, len(pairs))
    inter = pd.DataFrame({"user_id": (pairs.u + 1).astype(str), "item_id": (pairs.i + 1).astype(str),
                          "rating": rating.astype(str), "timestamp": ts.astype(str)})
    users = pd.DataFrame({"user_id": (np.arange(nu) + 1).astype(str),
                          "age": rng.integers(15, 70, nu).astype(str)})
    items = pd.DataFrame({"item_id": (np.arange(ni) + 1).astype(str),
                          "title": [f"Movie {k + 1} ({rng.integers(1980, 2017)})" for k in range(ni)],
                          "genres": ["|".join(np.array(genres)[row]) for row in item_g]})
    return users, items, inter, []


def inject_faults(users, items, inter, cfg):
    """Adds a small, logged set of defects to the raw tables (appended rows or changed cells)."""
    rng = np.random.default_rng(cfg["seed"] + 1)
    n = max(5, int(len(inter) * cfg["ingestion"]["fault_rate"]))
    users, items, inter = (d.reset_index(drop=True) for d in (users, items, inter))
    pick = rng.choice(len(inter), size=n * 6, replace=False)
    kinds = ["missing_rating", "rating_out_of_range", "rating_not_numeric",
             "timestamp_in_future", "orphan_item", "orphan_user"]
    log_rows = []
    for k, kind in enumerate(kinds):
        for row in pick[k * n:(k + 1) * n]:
            if kind == "missing_rating":
                inter.loc[row, "rating"] = np.nan
            elif kind == "rating_out_of_range":
                inter.loc[row, "rating"] = str(rng.choice([-3, 15, 99]))
            elif kind == "rating_not_numeric":
                inter.loc[row, "rating"] = "abc"
            elif kind == "timestamp_in_future":
                inter.loc[row, "timestamp"] = "4102444800"
            elif kind == "orphan_item":
                inter.loc[row, "item_id"] = "9" + str(9000000 + int(row))
            elif kind == "orphan_user":
                inter.loc[row, "user_id"] = "9" + str(9000000 + int(row))
            log_rows.append(("interactions", int(row), kind))

    clean_rows = np.setdiff1d(np.arange(len(inter)), pick)
    dup_src = rng.choice(clean_rows, size=n, replace=False)
    dups = inter.iloc[dup_src].copy()
    dups["timestamp"] = (dups["timestamp"].astype("int64") - 1).astype(str)   # older copy, original stays latest
    start = len(inter)
    inter = pd.concat([inter, dups], ignore_index=True)
    log_rows += [("interactions", start + j, "duplicate_user_item") for j in range(len(dups))]

    start = len(items)
    bad_items = pd.DataFrame({"item_id": ["8000001", "8000002", "8000003"],
                              "title": [np.nan, np.nan, "Untitled (2012)"], "genres": ["Drama", np.nan, np.nan]})
    dup_item = items.iloc[[0]].copy()
    dup_item["title"] = dup_item["title"] + " (dup)"
    items = pd.concat([items, bad_items.reindex(columns=items.columns), dup_item], ignore_index=True)
    log_rows += [("items", start, "missing_title"), ("items", start + 1, "missing_title"),
                 ("items", start + 2, "missing_genres"), ("items", start + 3, "duplicate_item_id")]

    start = len(users)
    dup_user = users.iloc[[0, 1]].copy()
    users = pd.concat([users, dup_user], ignore_index=True)
    log_rows += [("users", start, "duplicate_user_id"), ("users", start + 1, "duplicate_user_id")]

    return users, items, inter, pd.DataFrame(log_rows, columns=["table", "raw_row", "kind"])


def ingest(cfg: dict) -> dict:
    name = cfg["dataset"]["name"]
    raw, src = path(cfg, "raw"), path(cfg, "raw") / "source" / name
    src.mkdir(parents=True, exist_ok=True)
    log.info("ingesting dataset '%s'", name)

    if name == "movietweetings":
        users, items, inter, src_files = _load_movietweetings(cfg, src)
    elif name == "movielens100k":
        users, items, inter, src_files = _load_movielens100k(cfg, src)
    elif name == "synthetic":
        users, items, inter, src_files = _make_synthetic(cfg)
    else:
        raise ValueError(f"unknown dataset {name}")

    injected = 0
    fault_log = pd.DataFrame(columns=["table", "raw_row", "kind"])
    if cfg["ingestion"]["inject_faults"]:
        users, items, inter, fault_log = inject_faults(users, items, inter, cfg)
        injected = len(fault_log)
        log.info("injected %d defects for the validation stage (see data/raw/injected_faults.csv)", injected)

    for fname, df in (("users", users), ("items", items), ("interactions", inter)):
        df.to_csv(raw / f"{fname}.csv", index=False)
    fault_log.to_csv(raw / "injected_faults.csv", index=False)

    summary = {
        "dataset": name,
        "rows": {"users": len(users), "items": len(items), "interactions": len(inter)},
        "source_sha256": {p.name: _sha256(p) for p in src_files},
        "injected_defects": injected,
    }
    (raw / "ingestion_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("raw tables written: %s", summary["rows"])
    return summary
