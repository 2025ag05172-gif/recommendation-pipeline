"""Plain Python runner: runs every stage in order. The Prefect flow in dags/ runs the same functions.

    python -m src.pipeline                 # full run
    python -m src.pipeline --stage train   # one stage (needs the earlier ones to have run)
"""
from __future__ import annotations

import argparse
import json
import time

from src.common.config import get_logger, load_config, path
from src.evaluation.evaluate import evaluate
from src.features.build_features import build_features
from src.ingestion.ingest import ingest
from src.preparation.prepare import prepare
from src.training.train import MODEL_ORDER, train_all
from src.transformation.transform import transform
from src.validation.validate import validate

log = get_logger("pipeline")

STAGES = {
    "ingest": ingest,
    "validate": validate,
    "prepare": prepare,
    "transform": transform,
    "features": build_features,
    "train": train_all,
    "evaluate": evaluate,
}


def run(cfg: dict, only: str | None = None) -> dict:
    timings = {}
    for name, fn in STAGES.items():
        if only and name != only:
            continue
        t0 = time.time()
        log.info("=== stage: %s ===", name)
        fn(cfg)
        timings[name] = round(time.time() - t0, 1)
    if not only:
        (path(cfg, "reports") / "run_summary.json").write_text(json.dumps(
            {"dataset": cfg["dataset"]["name"], "seed": cfg["seed"], "stage_seconds": timings,
             "models": MODEL_ORDER}, indent=2))
        log.info("pipeline finished: %s", timings)
    return timings


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", choices=list(STAGES), default=None)
    args = ap.parse_args()
    run(load_config(args.config), args.stage)
