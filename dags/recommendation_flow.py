"""Prefect flow that orchestrates the whole pipeline.

Dependencies (an arrow means "must finish before"):

    ingest -> validate -> prepare -> transform -> features -> train: popularity, svd_cf, item_knn_cf, content_based
    (those four run in parallel)  ->  train: hybrid (needs the other four)  ->  evaluate -> summary

Run once:        python dags/recommendation_flow.py
Run on schedule: python dags/recommendation_flow.py --serve      (daily at 02:00)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, get_run_logger, task  # noqa: E402

from src.common.config import load_config, path  # noqa: E402
from src.evaluation.evaluate import evaluate  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.ingestion.ingest import ingest  # noqa: E402
from src.preparation.prepare import prepare  # noqa: E402
from src.training.train import train_model  # noqa: E402
from src.transformation.transform import transform  # noqa: E402
from src.validation.validate import validate  # noqa: E402


@task(name="1-ingest", retries=2, retry_delay_seconds=10)
def t_ingest(config_path):
    return ingest(load_config(config_path))


@task(name="2-validate")
def t_validate(config_path):
    report = validate(load_config(config_path))
    return {k: v["rows_rejected"] for k, v in report["tables"].items()}


@task(name="3-prepare")
def t_prepare(config_path):
    return prepare(load_config(config_path))


@task(name="4-transform")
def t_transform(config_path):
    return transform(load_config(config_path))


@task(name="5-features")
def t_features(config_path):
    return build_features(load_config(config_path))


@task(name="6-train")
def t_train(model_name, config_path):
    return train_model(model_name, load_config(config_path))


@task(name="7-evaluate")
def t_evaluate(config_path):
    table = evaluate(load_config(config_path))
    return table.round(4).to_dict(orient="records")


@task(name="8-summary")
def t_summary(config_path, results):
    cfg = load_config(config_path)
    out = path(cfg, "reports") / "flow_summary.json"
    out.write_text(json.dumps({"dataset": cfg["dataset"]["name"], "results": results}, indent=2))
    get_run_logger().info("flow summary written to %s", out)
    return str(out)


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None):
    ing = t_ingest(config_path)
    val = t_validate(config_path, wait_for=[ing])
    prep = t_prepare(config_path, wait_for=[val])
    trans = t_transform(config_path, wait_for=[prep])
    feats = t_features(config_path, wait_for=[trans])

    base = [t_train.submit(m, config_path, wait_for=[feats])
            for m in ("popularity", "svd_cf", "item_knn_cf", "content_based")]
    hybrid = t_train.submit("hybrid", config_path, wait_for=base)
    ev = t_evaluate(config_path, wait_for=[hybrid])
    return t_summary(config_path, ev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--serve", action="store_true", help="keep running and trigger the flow daily at 02:00")
    args = ap.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="recommendation-daily", cron="0 2 * * *",
                                      parameters={"config_path": args.config})
    else:
        recommendation_pipeline(args.config)
