"""Config loading, path helpers and logging shared by every stage."""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
_LOG_READY = False


def load_config(path: str | os.PathLike | None = None, root: Path | None = None) -> dict:
    cfg_path = Path(path or os.environ.get("REC_CONFIG", ROOT / "configs" / "config.yaml"))
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = Path(root) if root else ROOT
    cfg["ds"] = deepcopy(cfg["dataset"][cfg["dataset"]["name"]])
    return cfg


def path(cfg: dict, key: str) -> Path:
    p = Path(cfg["_root"]) / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    global _LOG_READY
    if not _LOG_READY:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if not root.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            root.addHandler(sh)
        log_dir = Path((cfg or {}).get("_root", ROOT)) / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "pipeline.log")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            pass
        _LOG_READY = True
    return logging.getLogger(name)
