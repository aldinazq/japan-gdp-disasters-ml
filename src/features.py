"""I build the cleaned yearly dataset combining GDP, GDP growth, and enriched disaster + macro features for my project."""

import numpy as np
import pandas as pd

from src.data_loading import (
    load_gdp_japan_raw,
    load_gdp_growth_japan_raw,
    load_japan_disasters_raw,
    load_wdi_macros_japan,
    load_oil_price,
)


def build_disaster_features(disasters: pd.DataFrame) -> pd.DataFrame:
    """I aggregate raw disaster events into yearly features (counts, deaths, damages, magnitude)."""
    df = disasters.copy()

    # Ensure required columns exist (and are numeric)
    required = ["year", "deaths", "damage_usd_thousands", "magnitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in disasters DataFrame: {missing}. "
            "Check load_japan_disasters_raw() column names."
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0)
    df["damage_usd_thousands"] = pd.to_numeric(df["damage_usd_thousands"], errors="coerce").fillna(0)
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce").fillna(0)

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            n_events=("year", "count"),
            total_deaths=("deaths", "sum"),
            total_damage=("damage_usd_thousands", "sum"),
            avg_magnitude=("magnitude", "mean"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    yearly["avg_magnitude"] = yearly["avg_magnitude"].fillna(0)
    return yearly


def build_master_table() -> pd.DataFrame:
    """I build the master annual dataset merging GDP, GDP growth, disaster features, and macro indicators."""
    gdp = load_gdp_japan_raw()
    growth = load_gdp_growth_japan_raw()

    disasters = load_japan_disasters_raw()
    disaster_features = build_disaster_features(disasters)

    macros = load_wdi_macros_japan()
    oil = load_oil_price()

    master = gdp.merge(growth, on="year", how="inner")
    master = master.merge(disaster_features, on="year", how="left")

    # Fill ONLY disaster columns with 0 (not macro variables)
    for c in ["n_events", "total_deaths", "total_damage", "avg_magnitude"]:
        if c in master.columns:
            master[c] = master[c].fillna(0)

    master = master.merge(macros, on="year", how="left")
    master = master.merge(oil, on="year", how="left")

    # Derived features
    master["log_total_damage"] = np.log1p(master["total_damage"])
    master["damage_share_gdp"] = np.where(
        master["gdp"] > 0,
        (master["total_damage"] * 1000.0) / master["gdp"],
        0.0,
    )
    master["has_disaster"] = (master["n_events"] > 0).astype(int)

    return master.sort_values("year").reset_index(drop=True)


if __name__ == "__main__":
    df = build_master_table()
    print("=== Master dataset ===")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    print(f"\n{df.shape[0]} rows, {df.shape[1]} columns")
