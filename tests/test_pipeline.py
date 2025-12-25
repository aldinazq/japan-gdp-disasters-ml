"""I test that the forecasting dataset is aligned, time split preserves chronology, and features follow a forecasting setup."""

import sys
import unittest
from pathlib import Path
import inspect

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import make_dataset, time_train_test_split


class TestPipeline(unittest.TestCase):
    def test_make_dataset_alignment(self) -> None:
        df, X, y = make_dataset(include_oil=False, mode="forecast")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertEqual(len(df), len(X))
        self.assertEqual(len(df), len(y))

        # Most important: same index (prevents df.loc[X.index] KeyErrors)
        self.assertTrue(df.index.equals(X.index))

        # Basic sanity
        self.assertIn("year", X.columns)

    def test_time_split_no_shuffle(self) -> None:
        df, X, y = make_dataset(include_oil=False, mode="forecast")

        # Compatible with BOTH versions:
        # - old: time_train_test_split(df, X, y, test_ratio=0.2)
        # - new: time_train_test_split(X, y, test_ratio=0.2)
        sig = inspect.signature(time_train_test_split)
        param_names = list(sig.parameters.keys())

        if len(param_names) >= 3 and param_names[0] in {"df", "dataframe"}:
            X_train, X_test, y_train, y_test = time_train_test_split(df, X, y, test_ratio=0.2)
        else:
            X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_ratio=0.2)

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)

        # Robust check: use year from X
        self.assertLess(int(X_train["year"].max()), int(X_test["year"].min()))

        # y lengths match
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))

    def test_forecast_setup_has_lag1_target_feature(self) -> None:
        _, X, _ = make_dataset(include_oil=False, mode="forecast")
        self.assertIn("gdp_growth_lag1", X.columns)

    def test_optional_macro_signature_if_present(self) -> None:
        sig = inspect.signature(make_dataset)

        kwargs = {"include_oil": False, "mode": "forecast"}
        if "include_macro" in sig.parameters:
            kwargs["include_macro"] = True
        if "start_year" in sig.parameters:
            kwargs["start_year"] = 1976

        df, X, y = make_dataset(**kwargs)
        self.assertGreater(len(df), 5)
        self.assertIn("year", X.columns)
        self.assertEqual(len(df), len(X))
        self.assertEqual(len(df), len(y))


if __name__ == "__main__":
    unittest.main()
