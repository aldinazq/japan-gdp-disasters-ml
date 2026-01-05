"""
I test that src.features.build_master_table produces a clean yearly master table with the required columns and structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_master_table


def test_master_table_is_dataframe_and_not_empty() -> None:
    """I ensure the master table builds and contains a reasonable number of yearly rows."""
    df = build_master_table()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 10, "Master table seems too small; check data loading and merges."


def test_master_table_has_core_columns() -> None:
    """I ensure the master table contains the minimum set of columns required by the modeling pipeline."""
    df = build_master_table()

    required = {
        "year",
        "gdp",
        "gdp_growth",
        "n_events",
        "total_deaths",
        "total_damage",
        "avg_magnitude",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns in master table: {sorted(missing)}"


def test_year_is_unique_sorted_and_non_missing() -> None:
    """I ensure year is well-formed: no NaN, unique, and strictly increasing."""
    df = build_master_table()

    assert "year" in df.columns
    assert df["year"].notna().all(), "year contains missing values."
    assert df["year"].is_unique, "year contains duplicates; merges may have created multiple rows per year."

    years = df["year"].astype(int).to_numpy()
    assert np.all(np.diff(years) > 0), "year is not strictly increasing; sort/merge logic may be broken."


def test_disaster_columns_are_non_negative() -> None:
    """I ensure disaster aggregates are non-negative (counts, deaths, damages)."""
    df = build_master_table()

    for col in ["n_events", "total_deaths", "total_damage"]:
        assert col in df.columns
        x = pd.to_numeric(df[col], errors="coerce")
        assert x.notna().all(), f"{col} contains non-numeric values or NaNs."
        assert (x >= 0).all(), f"{col} contains negative values; check aggregation/cleaning."

    # avg_magnitude can be zero if missing, but should not be negative.
    if "avg_magnitude" in df.columns:
        m = pd.to_numeric(df["avg_magnitude"], errors="coerce").fillna(0)
        assert (m >= 0).all(), "avg_magnitude contains negative values; check data cleaning."


def test_build_master_table_accepts_mode_argument() -> None:
    """I ensure API compatibility: build_master_table can be called with mode=... even if unused."""
    df_forecast = build_master_table(mode="forecast")
    df_nowcast = build_master_table(mode="nowcast")

    assert isinstance(df_forecast, pd.DataFrame)
    assert isinstance(df_nowcast, pd.DataFrame)
    assert df_forecast.shape[0] > 10
    assert df_nowcast.shape[0] > 10
