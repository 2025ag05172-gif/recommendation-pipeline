"""Feature registry: the single place where every feature is defined and documented.

The feature catalog (docs/feature_catalog.md and .csv) is generated from this list,
and a test checks that the feature tables contain exactly these columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Feature:
    name: str
    entity: str          # user | item | interaction
    dtype: str
    description: str
    formula: str
    table: str


FEATURES = [
    Feature("user_activity_count", "user", "int", "Number of ratings the user gave before the cutoff", "count(train rows of user)", "user_features"),
    Feature("user_avg_rating", "user", "float", "Mean rating given by the user", "mean(rating)", "user_features"),
    Feature("user_rating_std", "user", "float", "Spread of the user's ratings, 0 for a single rating", "std(rating)", "user_features"),
    Feature("user_positive_ratio", "user", "float", "Share of the user's ratings that count as liked", "mean(rating >= relevance_threshold)", "user_features"),
    Feature("user_days_since_last_interaction", "user", "float", "Days from the user's last rating to the split cutoff", "(cutoff - max(event_time)) in days", "user_features"),
    Feature("user_active_span_days", "user", "int", "Days between first and last rating", "max(event_time) - min(event_time)", "user_features"),
    Feature("user_preferred_genre", "user", "string", "Genre with the largest share of the user's liked items", "argmax(user_genre_share_*)", "user_features"),
    Feature("user_distinct_genres", "user", "int", "Number of genres the user has liked at least once", "count(user_genre_share_* > 0)", "user_features"),
    Feature("user_genre_share_<genre>", "user", "float", "Share of the user's liked items that carry the genre (rows sum to 1)", "sum(liked items with genre) / sum(liked genre tags)", "user_genre_profile"),
    Feature("item_popularity", "item", "int", "Number of train ratings the item received", "count(train rows of item)", "item_features"),
    Feature("item_avg_rating", "item", "float", "Mean rating of the item", "mean(rating)", "item_features"),
    Feature("item_bayes_rating", "item", "float", "Mean rating pulled towards the global mean for items with few ratings", "(n * item_mean + m * global_mean) / (n + m), m = prior strength", "item_features"),
    Feature("item_positive_ratio", "item", "float", "Share of the item's ratings that count as liked", "mean(rating >= relevance_threshold)", "item_features"),
    Feature("item_release_year", "item", "int", "Release year taken from the title", "regex (YYYY) in title, median if missing", "item_features"),
    Feature("item_age_years", "item", "int", "Age of the item at the split cutoff", "max(cutoff_year - release_year, 0)", "item_features"),
    Feature("item_days_since_last_interaction", "item", "float", "Days from the last rating of the item to the cutoff", "(cutoff - max(event_time)) in days", "item_features"),
    Feature("item_genre_count", "item", "int", "Number of genres on the item", "sum(genre_*)", "item_features"),
    Feature("genre_<genre>", "item", "int8", "1 when the item carries the genre", "genre in item.genres", "item_features"),
    Feature("rating_user_deviation", "interaction", "float", "How far the rating is from the user's own mean", "rating - user_avg_rating", "interaction_features"),
    Feature("rating_item_deviation", "interaction", "float", "How far the rating is from the item's mean", "rating - item_avg_rating", "interaction_features"),
    Feature("days_since_interaction", "interaction", "float", "Age of the interaction at the cutoff", "(cutoff - event_time) in days", "interaction_features"),
    Feature("recency_weight", "interaction", "float", "Exponential decay of the interaction age", "0.5 ** (days_since_interaction / half_life)", "interaction_features"),
    Feature("user_genre_affinity", "interaction", "float", "How well the item's genres match the user's liked genres (available at prediction time)", "mean over item genres of user_genre_share_<genre>", "interaction_features"),
]


def sanitize(genre: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in genre)
    return "_".join(p for p in out.split("_") if p)


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in FEATURES])


def expected_columns(table: str, genre_names: list[str]) -> set[str]:
    cols = set()
    for f in FEATURES:
        if f.table != table:
            continue
        if "<genre>" in f.name:
            cols |= {f.name.replace("<genre>", g) for g in genre_names}
        else:
            cols.add(f.name)
    return cols


def write_catalog(csv_path, md_path) -> None:
    df = catalog_frame()
    df.to_csv(csv_path, index=False)
    lines = ["# Feature catalog", "",
             "Generated from `src/features/registry.py`. Features are computed from the train split only, "
             "as of the split cutoff, and read through `FeatureStore` for both training and prediction.", "",
             "| feature | entity | type | table | description | formula |", "|---|---|---|---|---|---|"]
    for f in FEATURES:
        lines.append(f"| `{f.name}` | {f.entity} | {f.dtype} | {f.table} | {f.description} | `{f.formula}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
