"""I test that my feature-building code produces a clean yearly dataset with expected columns and ordering."""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features import build_master_table


class TestFeatures(unittest.TestCase):
    def test_master_table_not_empty(self) -> None:
        df = build_master_table()
        self.assertFalse(df.empty)
        self.assertGreater(df.shape[0], 10)

    def test_master_table_has_expected_columns(self) -> None:
        df = build_master_table()
        expected = {
            "year",
            "gdp",
            "gdp_growth",
            "n_events",
            "total_deaths",
            "total_damage",
            "avg_magnitude",
            "log_total_damage",
            "damage_share_gdp",
            "has_disaster",
            "inflation_cpi",
            "exports_pct_gdp",
            "unemployment_rate",
            "investment_pct_gdp",
            "fx_jpy_per_usd",
            "oil_price_usd",
        }
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_year_sorted(self) -> None:
        df = build_master_table()
        self.assertTrue(df["year"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
