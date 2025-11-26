"""I build the cleaned yearly dataset combining GDP, GDP growth, and enriched disaster features for my project."""

import numpy as np
import pandas as pd
from src.data_loading import (
    load_gdp_japan_raw,
    load_gdp_growth_japan_raw,
    load_japan_disasters_raw,
)


def build_disaster_features(disasters: pd.DataFrame) -> pd.DataFrame:
    disasters = disasters.copy()
    disasters.columns = disasters.columns.str.strip()
    disasters["year"] = pd.to_numeric(disasters["year"], errors="coerce")
    disasters = disasters.dropna(subset=["year"])
    disasters["year"] = disasters["year"].astype(int)
    disasters["deaths"] = pd.to_numeric(disasters["deaths"], errors="coerce").fillna(0)
    disasters["damage_usd_thousands"] = pd.to_numeric(
        disasters["damage_usd_thousands"], errors="coerce"
    ).fillna(0)
    disasters["magnitude"] = pd.to_numeric(disasters["magnitude"], errors="coerce")

    yearly = (
        disasters.groupby("year")
        .agg(
            n_events=("event_name", "count"),
            total_deaths=("deaths", "sum"),
            total_damage=("damage_usd_thousands", "sum"),
            avg_magnitude=("magnitude", "mean"),
        )
        .reset_index()
    )
    return yearly


def build_master_table() -> pd.DataFrame:
    gdp = load_gdp_japan_raw()
    growth = load_gdp_growth_japan_raw()
    disasters = load_japan_disasters_raw()
    disaster_features = build_disaster_features(disasters)

    master = (
        gdp.merge(growth, on="year", how="inner")
        .merge(disaster_features, on="year", how="left")
        .fillna(0)
    )

    master["log_total_damage"] = np.log1p(master["total_damage"])
    master["damage_share_gdp"] = np.where(
        master["gdp"] > 0,
        (master["total_damage"] * 1000.0) / master["gdp"],
        0.0,
    )
    master["has_disaster"] = (master["n_events"] > 0).astype(int)

    return master


if __name__ == "__main__":
    df = build_master_table()
    print("=== Master dataset ===")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    print(f"\n{df.shape[0]} rows, {df.shape[1]} columns")
