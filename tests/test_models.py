"""I test that my modeling pipeline runs end-to-end and returns sensible metrics."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.models import make_dataset, run_all_models


def test_make_dataset_shapes() -> None:
    df, X, y = make_dataset()
    assert len(df) == X.shape[0] == y.shape[0]
    assert y.ndim == 1
    assert not pd.isna(y).any()


def test_run_all_models_returns_metrics_table() -> None:
    metrics = run_all_models()
    expected_cols = {
        "model",
        "train_MAE",
        "train_RMSE",
        "train_R2",
        "test_MAE",
        "test_RMSE",
        "test_R2",
    }
    assert expected_cols.issubset(metrics.columns)
    assert len(metrics) >= 2
    assert (metrics["model"] != "").all()
