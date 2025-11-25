"""I test that my feature-building code produces a clean yearly dataset with the expected structure."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features import build_master_table


def test_master_table_not_empty() -> None:
    df = build_master_table()
    assert not df.empty
    assert df.shape[0] > 10


def test_master_table_has_expected_columns() -> None:
    df = build_master_table()
    expected = {
        "year",
        "gdp",
        "gdp_growth",
        "n_events",
        "total_deaths",
        "total_damage",
        "avg_magnitude",
    }
    assert expected.issubset(df.columns)
