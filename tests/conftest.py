import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROOT, load_config  # noqa: E402


def make_cfg(root: Path, dataset: str = "synthetic") -> dict:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "configs" / "config.yaml", root / "configs" / "config.yaml")
    cfg = load_config(root / "configs" / "config.yaml", root=root)
    cfg["dataset"]["name"] = dataset
    cfg["ds"] = dict(cfg["dataset"][dataset])
    return cfg


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    """One full pipeline run on synthetic data, shared by several tests."""
    from src.pipeline import run
    root = tmp_path_factory.mktemp("run1")
    cfg = make_cfg(root)
    run(cfg)
    return cfg
