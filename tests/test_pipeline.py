"""
I test that the dataset construction and time-based splitting behave consistently and without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import make_dataset, time_train_test_split


def test_make_dataset_alignment_and_types() -> None:
    """I ensure df, X, y align perfectly and are numeric-friendly for sklearn."""
    df, X, y = make_dataset(
        include_oil=False,
        include_macro=False,
        include_covid=False,
        start_year=None,
        mode="forecast",
    )

    assert isinstance(df, pd.DataFrame)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)

    assert len(df) == len(X) == len(y), "df, X, y must have the same number of rows."
    assert X.shape[0] > 10, "Dataset seems too small; check feature building or start_year restriction."

    # The modeling pipeline expects to use year as a feature (and tests rely on it).
    assert "year" in X.columns, "X must include 'year' for time ordering checks."

    # y should be finite for training/testing; df has been cleaned with dropna on key columns.
    assert np.isfinite(y).all(), "y contains non-finite values; check target construction."


def test_make_dataset_contains_key_forecast_features() -> None:
    """I ensure the strict forecast setup includes at least the key lagged GDP feature."""
    df, X, y = make_dataset(mode="forecast")

    assert "gdp_growth_lag1" in X.columns, "Forecast dataset must contain gdp_growth_lag1."
    assert "gdp_growth_lag2" in X.columns, "Forecast dataset should contain gdp_growth_lag2."
    assert "n_events_lag1" in X.columns, "Forecast dataset should contain lagged disaster count."

    # At minimum, lags should exist (even if some values are NaN early in sample).
    assert X["gdp_growth_lag1"].notna().sum() > 5, "Too few non-missing lag1 values; check lag construction."


def test_time_split_is_chronological_no_leakage() -> None:
    """I ensure the train/test split respects time ordering (train years < test years)."""
    df, X, y = make_dataset(mode="forecast", start_year=1992)

    X_train, X_test, y_train, y_test = time_train_test_split(df, X, y, test_ratio=0.2)

    assert len(X_train) > 0 and len(X_test) > 0
    assert len(X_train) + len(X_test) == len(X)

    # Chronological sanity check: last train year < first test year
    max_train_year = int(np.nanmax(pd.to_numeric(X_train["year"], errors="coerce")))
    min_test_year = int(np.nanmin(pd.to_numeric(X_test["year"], errors="coerce")))
    assert max_train_year < min_test_year, "Time split violated chronology; potential leakage."


def test_macro_and_oil_flags_do_not_break_dataset() -> None:
    """I ensure include_macro/include_oil runs build successfully and keep baseline features."""
    df_base, X_base, y_base = make_dataset(
        include_macro=False,
        include_oil=False,
        include_covid=False,
        mode="forecast",
        start_year=1992,
    )

    df_macro, X_macro, y_macro = make_dataset(
        include_macro=True,
        include_oil=False,
        include_covid=False,
        mode="forecast",
        start_year=1992,
    )

    df_oil, X_oil, y_oil = make_dataset(
        include_macro=False,
        include_oil=True,
        include_covid=False,
        mode="forecast",
        start_year=1992,
    )

    # Basic validity
    assert len(X_macro) == len(y_macro) == len(df_macro)
    assert len(X_oil) == len(y_oil) == len(df_oil)

    # Baseline columns should still exist
    for c in ["year", "gdp_growth_lag1", "n_events_lag1"]:
        assert c in X_macro.columns, f"Missing baseline column '{c}' in macro dataset."
        assert c in X_oil.columns, f"Missing baseline column '{c}' in oil dataset."


def test_nowcast_adds_contemporaneous_disaster_info_if_available() -> None:
    """I ensure nowcast mode can include contemporaneous disaster columns (when present in df)."""
    df_now, X_now, y_now = make_dataset(mode="nowcast", start_year=1992)

    # In nowcast mode we attempt to include contemporaneous disaster columns if present.
    # We do not require them strictly (depends on master table), but if they exist they should be in X.
    for c in ["n_events", "total_deaths", "log_total_damage", "damage_share_gdp", "avg_magnitude"]:
        if c in df_now.columns:
            assert c in X_now.columns, f"Nowcast expected '{c}' in X when available in df."
