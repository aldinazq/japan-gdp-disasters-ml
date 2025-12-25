"""I verify that the feature builder returns a clean yearly master table with core columns and ordering."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features import build_master_table


class TestFeatures(unittest.TestCase):
    def test_master_table_is_dataframe_and_not_empty(self) -> None:
        df = build_master_table()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 10)

    def test_master_table_has_core_columns(self) -> None:
        df = build_master_table()
        required = {
            "year",
            "gdp",
            "gdp_growth",
            "n_events",
            "total_deaths",
            "total_damage",
        }
        self.assertTrue(required.issubset(set(df.columns)))

    def test_year_is_unique_and_sorted(self) -> None:
        df = build_master_table().copy()
        years = df["year"].astype(int).to_numpy()

        # Unique yearly rows
        self.assertEqual(len(years), len(set(years)))

        # Strictly increasing (time series)
        diffs = np.diff(years)
        self.assertTrue((diffs > 0).all())

    def test_year_has_no_missing_values(self) -> None:
        df = build_master_table()
        self.assertFalse(df["year"].isna().any())


if __name__ == "__main__":
    unittest.main()
