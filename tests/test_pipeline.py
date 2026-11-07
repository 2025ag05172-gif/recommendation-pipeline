import json
import zipfile

import numpy as np
import pandas as pd

from src.common.config import path
from src.features.feature_store import FeatureStore
from src.features.registry import FEATURES, expected_columns
from src.ingestion.ingest import ingest
from src.pipeline import run
from tests.conftest import make_cfg


def test_validation_catches_every_injected_defect(pipeline_run):
    report = json.loads((path(pipeline_run, "reports") / "validation_report.json").read_text())
    c = report["injected_defect_check"]
    assert c["injected"] > 0 and c["caught"] == c["injected"]
    errors = pd.read_csv(path(pipeline_run, "errors") / "interactions_errors.csv")
    assert {"rating_out_of_range", "orphan_item", "orphan_user", "invalid_timestamp"} <= set(
        ";".join(errors["error_reason"]).split(";"))


def test_valid_data_has_no_referential_or_range_problems(pipeline_run):
    interim = path(pipeline_run, "interim")
    inter = pd.read_parquet(interim / "interactions.parquet")
    users, items = pd.read_parquet(interim / "users.parquet"), pd.read_parquet(interim / "items.parquet")
    assert inter["user_id"].isin(users["user_id"]).all() and inter["item_id"].isin(items["item_id"]).all()
    assert inter["rating"].between(pipeline_run["ds"]["rating_min"], pipeline_run["ds"]["rating_max"]).all()


def test_prepared_data_has_no_duplicates_or_missing_values(pipeline_run):
    processed = path(pipeline_run, "processed")
    inter = pd.read_parquet(processed / "interactions.parquet")
    items = pd.read_parquet(processed / "items.parquet")
    assert not inter.duplicated(["user_id", "item_id"]).any()
    assert items[["title", "genres", "release_year"]].notna().all().all()


def test_time_split_has_no_leakage(pipeline_run):
    processed = path(pipeline_run, "processed")
    train, test = pd.read_parquet(processed / "train.parquet"), pd.read_parquet(processed / "test.parquet")
    assert train["timestamp"].max() < test["timestamp"].min()
    assert test["user_id"].isin(train["user_id"]).all() and test["item_id"].isin(train["item_id"]).all()


def test_feature_tables_match_registry_and_catalog(pipeline_run):
    feats = path(pipeline_run, "features")
    meta = json.loads((feats / "feature_meta.json").read_text())
    for table in ("user_features", "item_features", "user_genre_profile", "interaction_features"):
        cols = set(pd.read_parquet(feats / f"{table}.parquet").columns) - {"user_id", "item_id"}
        assert cols == expected_columns(table, meta["genres"]), table
    catalog = pd.read_csv(pipeline_run["_root"] / "docs" / "feature_catalog.csv")
    assert len(catalog) == len(FEATURES)


def test_features_use_train_data_only(pipeline_run):
    processed, feats = path(pipeline_run, "processed"), path(pipeline_run, "features")
    train = pd.read_parquet(processed / "train.parquet")
    items = pd.read_parquet(feats / "item_features.parquet").set_index("item_id")
    expected = train.groupby("item_id").size()
    assert (items["item_popularity"] == expected.reindex(items.index)).all()


def test_same_feature_code_for_training_and_prediction(pipeline_run):
    store = FeatureStore(pipeline_run)
    train = pd.read_parquet(path(pipeline_run, "processed") / "train.parquet").head(50)
    full = store.interaction_features(train[["user_id", "item_id", "rating", "event_time"]])
    pair_only = store.interaction_features(train[["user_id", "item_id"]])      # prediction time: no rating, no time
    np.testing.assert_allclose(full["user_genre_affinity"], pair_only["user_genre_affinity"])
    assert pair_only["rating_user_deviation"].isna().all()


def test_results_table_has_all_models_and_metrics(pipeline_run):
    t = pd.read_csv(path(pipeline_run, "reports") / "model_comparison.csv")
    assert set(t["model"]) == {"popularity", "svd_cf", "item_knn_cf", "content_based", "hybrid"}
    for col in ("precision_at_10", "recall_at_10", "ndcg_at_10", "coverage_at_10", "diversity_at_10", "precision_at_5"):
        assert t[col].between(0, 1).all()
    assert t.loc[t.model == "svd_cf", "rmse"].notna().all()
    ndcg = t.set_index("model")["ndcg_at_10"]
    assert max(ndcg["item_knn_cf"], ndcg["svd_cf"]) > ndcg["popularity"]


def test_mlflow_runs_have_params_and_metrics(pipeline_run):
    import mlflow
    from src.training.train import setup_mlflow
    setup_mlflow(pipeline_run)
    runs = mlflow.search_runs(experiment_names=[pipeline_run["mlflow"]["experiment"]])
    assert len(runs) == 5
    assert runs["metrics.ndcg_at_10"].notna().all() and runs["params.seed"].notna().all()


def test_pipeline_is_reproducible(pipeline_run, tmp_path):
    cfg2 = make_cfg(tmp_path)
    run(cfg2)
    a = pd.read_csv(path(pipeline_run, "reports") / "model_comparison.csv")
    b = pd.read_csv(path(cfg2, "reports") / "model_comparison.csv")
    pd.testing.assert_frame_equal(a, b)


def test_movielens_100k_adapter_parses_original_file_layout(tmp_path):
    d = tmp_path / "ml-100k"
    d.mkdir()
    (d / "u.data").write_text("1\t10\t5\t881250949\n2\t10\t3\t881250950\n")
    (d / "u.user").write_text("1|24|M|technician|85711\n2|53|F|other|94043\n")
    flags = lambda *idx: "|".join("1" if i in idx else "0" for i in range(19))  # noqa: E731
    (d / "u.item").write_bytes(
        (f"10|Café Film (1995)|01-Jan-1995||http://x|{flags(1, 5)}\n").encode("latin-1"))
    zpath = tmp_path / "ml-100k.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in d.iterdir():
            z.write(f, f"ml-100k/{f.name}")
    cfg = make_cfg(tmp_path / "root", "movielens100k")
    cfg["ds"]["url"] = zpath.as_uri()
    cfg["ingestion"]["inject_faults"] = False
    ingest(cfg)
    raw = path(cfg, "raw")
    items = pd.read_csv(raw / "items.csv", dtype=str)
    users = pd.read_csv(raw / "users.csv", dtype=str)
    assert items.loc[0, "genres"] == "Action|Comedy" and items.loc[0, "title"] == "Café Film (1995)"
    assert list(users.columns) == ["user_id", "age", "gender", "occupation", "zip_code"]
    assert len(pd.read_csv(raw / "interactions.csv")) == 2
