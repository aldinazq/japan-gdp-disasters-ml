"""
I build the cleaned yearly master table by merging GDP, GDP growth, disaster aggregates, and optional macro/oil controls.
"""

import numpy as np
import pandas as pd

from typing import Optional

from src.data_loading import (
    load_gdp_japan_raw,
    load_gdp_growth_japan_raw,
    load_japan_disasters_raw,
    load_wdi_macros_japan,
    load_oil_price,
)


def build_disaster_features(disasters: pd.DataFrame) -> pd.DataFrame:
    """
    I aggregate event-level EM-DAT data into yearly disaster features.

    This paragraph is for making the forecasting dataset usable.
    Models work with one row per year, so I convert many events into yearly totals and averages.

    Convention:
    - total_damage is in USD (not thousands)
    """
    df = disasters.copy()

    # This paragraph is for backward compatibility.
    # If the input only has damages in thousands, I convert here so the rest of the project is consistent.
    if "damage_usd" not in df.columns and "damage_usd_thousands" in df.columns:
        df["damage_usd"] = pd.to_numeric(df["damage_usd_thousands"], errors="coerce") * 1000.0

    # This paragraph is for data integrity.
    # I fail early if the expected columns are missing so bugs are caught close to the source.
    required = ["year", "deaths", "damage_usd", "magnitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in disasters DataFrame: {missing}. "
            "Check load_japan_disasters_raw() column names."
        )

    # This paragraph is for stable aggregation.
    # I convert numeric types explicitly so groupby sums/means behave as expected.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0)
    df["damage_usd"] = pd.to_numeric(df["damage_usd"], errors="coerce").fillna(0)
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce").fillna(0)

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # This paragraph is for interpretable yearly signals.
    # Counts + totals give scale; average magnitude gives “typical intensity” when magnitudes exist.
    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            n_events=("year", "count"),
            total_deaths=("deaths", "sum"),
            total_damage=("damage_usd", "sum"),
            avg_magnitude=("magnitude", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    # This paragraph is for conservative defaults.
    # If magnitude is missing for a year, I keep 0 so the column stays numeric and merge-friendly.
    yearly["avg_magnitude"] = yearly["avg_magnitude"].fillna(0)

    return yearly


def build_master_table(mode: Optional[str] = None) -> pd.DataFrame:
    """
    I build the master annual dataset by merging:
    - GDP level
    - GDP growth (target)
    - disaster aggregates
    - optional macro controls
    - optional oil price

    This paragraph is for API compatibility.
    The `mode` argument is accepted so models.py can call build_master_table(mode=...) consistently,
    but this function currently builds the same master table for both forecast and nowcast.

    This paragraph is for having a single “source of truth” table.
    All model runs start from the same master table so results are comparable and reproducible.
    """
    # This paragraph is for loading core targets and levels.
    # GDP and GDP growth are required; disasters/macros/oil can be partially missing.
    gdp = load_gdp_japan_raw()
    growth = load_gdp_growth_japan_raw()

    # This paragraph is for building yearly disaster aggregates from event-level EM-DAT.
    disasters = load_japan_disasters_raw()
    disaster_features = build_disaster_features(disasters)

    # This paragraph is for optional controls.
    # I keep these as left-merged blocks so missing macro/oil values do not delete years.
    macros = load_wdi_macros_japan()
    oil = load_oil_price()

    # This paragraph is for defining the modeling sample cleanly.
    # I use an inner merge for GDP and GDP growth because the target must exist.
    master = gdp.merge(growth, on="year", how="inner")

    # This paragraph is for adding disaster features.
    # If some years have zero disasters, they might be missing in the aggregated table, so I left-merge and fill.
    master = master.merge(disaster_features, on="year", how="left")
    master["n_events"] = master["n_events"].fillna(0).astype(int)
    master["total_deaths"] = master["total_deaths"].fillna(0.0)
    master["total_damage"] = master["total_damage"].fillna(0.0)
    master["avg_magnitude"] = master["avg_magnitude"].fillna(0.0)

    # This paragraph is for optional macro and oil merges.
    # I keep them as left merges because missing macro/oil values should not delete years.
    master = master.merge(macros, on="year", how="left")
    master = master.merge(oil, on="year", how="left")

    # This paragraph is for numeric stability.
    # I enforce numeric types for key variables so downstream sklearn code behaves consistently.
    master["gdp"] = pd.to_numeric(master["gdp"], errors="coerce")
    master["gdp_growth"] = pd.to_numeric(master["gdp_growth"], errors="coerce")

    # This paragraph is for making damages easier to learn from.
    # Raw damages can be extremely skewed, so log(1+damage) is a standard stabilizing transform.
    master["log_total_damage"] = np.log1p(pd.to_numeric(master["total_damage"], errors="coerce").fillna(0))

    # This paragraph is for comparability across time.
    # Damages in dollars mean different things when GDP changes, so I scale by GDP as well.
    master["damage_share_gdp"] = np.where(
        pd.to_numeric(master["gdp"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(master["total_damage"], errors="coerce").fillna(0) / pd.to_numeric(master["gdp"], errors="coerce"),
        0.0,
    )

    # This paragraph is for a simple binary signal that some models can use well.
    master["has_disaster"] = (master["n_events"] > 0).astype(int)

    return master.sort_values("year").reset_index(drop=True)


if __name__ == "__main__":
    # This paragraph is for quick manual checks during development.
    # I print head + columns to spot merge issues before running the full pipeline.
    df = build_master_table()
    print("=== Master dataset ===")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    print(f"\n{df.shape[0]} rows, {df.shape[1]} columns")
