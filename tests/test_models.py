"""I test that my modeling pipeline runs end-to-end and returns sensible metrics."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import make_dataset, run_all_models


class TestModels(unittest.TestCase):
    def test_make_dataset_shapes(self) -> None:
        df, X, y = make_dataset(include_oil=False)
        self.assertEqual(len(df), X.shape[0])
        self.assertEqual(len(df), y.shape[0])
        self.assertEqual(y.ndim, 1)
        self.assertFalse(pd.isna(y).any())

    def test_run_all_models_returns_metrics_table(self) -> None:
        metrics = run_all_models(include_oil=False, test_ratio=0.2, tune_rf=False)
        expected_cols = {
            "model",
            "train_MAE",
            "train_RMSE",
            "train_R2",
            "test_MAE",
            "test_RMSE",
            "test_R2",
        }
        self.assertTrue(expected_cols.issubset(set(metrics.columns)))
        self.assertGreaterEqual(len(metrics), 2)
        self.assertTrue((metrics["model"] != "").all())


if __name__ == "__main__":
    unittest.main()
