"""I test that my feature-building code produces a clean yearly dataset with the expected structure."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

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
        "log_total_damage",
        "damage_share_gdp",
        "has_disaster",
    }
    assert expected.issubset(df.columns)


def test_year_is_sorted_and_unique() -> None:
    df = build_master_table()
    assert df["year"].is_monotonic_increasing
    assert df["year"].is_unique


def test_no_missing_values_in_core_features() -> None:
    df = build_master_table()
    core_cols = [
        "gdp_growth",
        "n_events",
        "total_damage",
        "log_total_damage",
        "damage_share_gdp",
        "has_disaster",
    ]
    assert df[core_cols].notna().all().all()


def test_has_disaster_flag_consistent_with_events() -> None:
    df = build_master_table()
    zero_mask = df["n_events"] == 0
    positive_mask = df["n_events"] > 0
    assert (df.loc[zero_mask, "has_disaster"] == 0).all()
    assert (df.loc[positive_mask, "has_disaster"] == 1).all()
