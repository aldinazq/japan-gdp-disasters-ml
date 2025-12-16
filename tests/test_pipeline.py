"""I test that my time-series split is ordered and my feature set avoids obvious leakage."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import make_dataset, time_train_test_split


class TestPipeline(unittest.TestCase):
    def test_time_split_no_shuffle(self) -> None:
        _, X, y = make_dataset(include_oil=False)
        X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_ratio=0.2)

        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertLess(X_train.index.max(), X_test.index.min())
        self.assertEqual(X_train.index.max() + 1, X_test.index.min())
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_no_obvious_macro_leakage(self) -> None:
        _, X, _ = make_dataset(include_oil=True)

        forbidden_exact = {
            "inflation_cpi",
            "exports_pct_gdp",
            "unemployment_rate",
            "investment_pct_gdp",
            "fx_jpy_per_usd",
            "oil_price_usd",
        }
        for c in X.columns:
            self.assertNotIn(c, forbidden_exact)

        for c in X.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(X[c]), msg=f"{c} is not numeric")


if __name__ == "__main__":
    unittest.main()
