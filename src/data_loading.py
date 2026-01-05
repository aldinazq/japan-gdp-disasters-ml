"""
I load and standardize the raw Excel inputs (WDI + EM-DAT + oil) into clean yearly DataFrames for my project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# This paragraph is for stable paths when TAs run the code from any directory.
# I compute paths from the file location so "python main.py" works everywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_gdp_japan_raw() -> pd.DataFrame:
    """
    I load Japan GDP level (current US$) from a WDI-style Excel extract.

    This paragraph is for keeping the master table consistent across sources.
    I return only [year, gdp] so merges are simple and easy to audit.
    """
    path = DATA_DIR / "GDP_Japan.xlsx"

    # This paragraph is for robustness: WDI exports often include header rows and many empty columns.
    df = pd.read_excel(path, sheet_name="Data", skiprows=3)
    df = df[df["Country Code"] == "JPN"].dropna(axis=1, how="all")

    # This paragraph is for converting the wide WDI format into a tidy time series.
    df_long = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="year",
        value_name="gdp",
    )

    # This paragraph is for type safety.
    # I force numeric types because Excel imports can create mixed types and break merges later.
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long["gdp"] = pd.to_numeric(df_long["gdp"], errors="coerce")

    # This paragraph is for keeping only usable observations for modeling.
    df_long = df_long.dropna(subset=["gdp", "year"])

    return df_long[["year", "gdp"]].sort_values("year").reset_index(drop=True)


def load_gdp_growth_japan_raw() -> pd.DataFrame:
    """
    I load Japan GDP growth (annual %) from a WDI-style Excel extract.

    This paragraph is for separating "level GDP" and "growth GDP" clearly,
    because mixing them causes confusion in feature engineering and evaluation.
    """
    path = DATA_DIR / "GDP_Growth_japan.xlsx"
    df = pd.read_excel(path, sheet_name="Data", skiprows=3)
    df = df[df["Country Code"] == "JPN"].dropna(axis=1, how="all")

    df_long = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="year",
        value_name="gdp_growth",
    )

    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long["gdp_growth"] = pd.to_numeric(df_long["gdp_growth"], errors="coerce")
    df_long = df_long.dropna(subset=["gdp_growth", "year"])

    return df_long[["year", "gdp_growth"]].sort_values("year").reset_index(drop=True)


def load_japan_disasters_raw() -> pd.DataFrame:
    """
    I load EM-DAT events for Japan and return an event-level cleaned table.

    This paragraph is for unit consistency.
    EM-DAT provides "Total Damage ('000 US$)" in thousands, so I create a USD column (damage_usd)
    that the rest of the project can safely use.
    """
    path = DATA_DIR / "Natural_disasters_Japan.xlsx"
    df = pd.read_excel(path, sheet_name="EM-DAT Data")

    # This paragraph is for keeping only fields that matter for yearly aggregation.
    # I avoid carrying many unused columns because it makes debugging harder later.
    keep_cols = [
        "Disaster Type",
        "Disaster Subtype",
        "Event Name",
        "Magnitude",
        "Magnitude Scale",
        "Total Deaths",
        "Total Damage ('000 US$)",
        "Start Year",
    ]
    df = df[keep_cols].rename(
        columns={
            "Start Year": "year",
            "Total Deaths": "deaths",
            "Total Damage ('000 US$)": "damage_usd_thousands",
            "Disaster Type": "disaster_type",
            "Disaster Subtype": "disaster_subtype",
            "Event Name": "event_name",
            "Magnitude": "magnitude",
            "Magnitude Scale": "magnitude_scale",
        }
    )

    # This paragraph is for clean year keys.
    # If year is missing, the event cannot be used in a yearly forecast dataset.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # This paragraph is for preserving meaning.
    # I keep NaN as "unknown" at the event level; I only fill zeros after yearly aggregation and merge.
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df["damage_usd_thousands"] = pd.to_numeric(df["damage_usd_thousands"], errors="coerce")
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")

    # This paragraph is for coherence across the whole project.
    # Downstream code should never guess whether damages are in thousands or USD.
    df["damage_usd"] = df["damage_usd_thousands"] * 1000.0

    # This paragraph is for sanity: negative damages do not make sense.
    # I set them to missing rather than forcing them to 0 (0 would mean "no damage").
    df.loc[df["damage_usd"] < 0, "damage_usd"] = pd.NA
    df.loc[df["damage_usd_thousands"] < 0, "damage_usd_thousands"] = pd.NA

    return df.reset_index(drop=True)


def _load_wdi_extract(file_name: str, value_name: str) -> pd.DataFrame:
    """
    I read a World Bank DataBank WDI extract and return a tidy DataFrame [year, value_name].

    This paragraph is for handling the common WDI export format:
    - wide columns like [YR1961], [YR1962], ...
    - sometimes extra header rows
    - sometimes ".." or non-numeric markers
    """
    path = DATA_DIR / file_name

    # This paragraph is for compatibility with different export styles.
    # Some files already have the right header; some need skiprows.
    df = pd.read_excel(path, sheet_name="Data")
    if "Series Code" not in df.columns or "Country Code" not in df.columns:
        df = pd.read_excel(path, sheet_name="Data", skiprows=3)

    # This paragraph is for filtering the exact series + country before reshaping.
    # It keeps the melt small and avoids accidentally mixing countries.
    df = df[df["Series Code"].notna()].copy()
    df = df[df["Country Code"] == "JPN"].copy()

    year_cols = [c for c in df.columns if isinstance(c, str) and "[YR" in c]

    # This paragraph is for turning WDI’s wide structure into a standard year/value series.
    long = df.melt(
        id_vars=["Series Code", "Country Code"],
        value_vars=year_cols,
        var_name="year_raw",
        value_name=value_name,
    )

    # This paragraph is for safe numeric conversion.
    # I treat non-numeric markers as missing because forcing them to 0 would be misleading.
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long["year"] = long["year_raw"].str.extract(r"\[YR(\d{4})\]")[0].astype(int)
    long = long.dropna(subset=[value_name])

    return long[["year", value_name]].sort_values("year").reset_index(drop=True)


def load_wdi_macros_japan() -> pd.DataFrame:
    """
    I load a small set of macro controls from WDI and merge them by year.

    This paragraph is for modularity.
    I keep macro loading in one function so the master table stays clean and predictable.
    """
    inflation = _load_wdi_extract("wdi_inflation.xlsx", "inflation_cpi")
    exports = _load_wdi_extract("wdi_exports_pct_gdp.xlsx", "exports_pct_gdp")
    unemp = _load_wdi_extract("wdi_unemployment.xlsx", "unemployment_rate")
    invest = _load_wdi_extract("wdi_investment_pct_gdp.xlsx", "investment_pct_gdp")
    fx = _load_wdi_extract("wdi_fx_jpy_per_usd.xlsx", "fx_jpy_per_usd")

    # This paragraph is for keeping years whenever at least one macro series exists.
    # Outer merges avoid dropping years just because one indicator is missing.
    df = inflation.merge(exports, on="year", how="outer")
    df = df.merge(unemp, on="year", how="outer")
    df = df.merge(invest, on="year", how="outer")
    df = df.merge(fx, on="year", how="outer")

    return df.sort_values("year").reset_index(drop=True)


def load_oil_price() -> pd.DataFrame:
    """
    I load annual oil prices and return [year, oil_price_usd].

    This paragraph is for robustness to different column names.
    Some sources name the date/value columns differently, so I detect them safely.
    """
    path = DATA_DIR / "oil_price.xlsx"
    df = pd.read_excel(path)

    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    value_col = "POILAPSPUSDA" if "POILAPSPUSDA" in df.columns else df.columns[1]

    # This paragraph is for converting to a yearly index because my dataset is annual.
    df["year"] = pd.to_datetime(df[date_col]).dt.year
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.rename(columns={value_col: "oil_price_usd"}).dropna(subset=["oil_price_usd"])

    return df[["year", "oil_price_usd"]].sort_values("year").reset_index(drop=True)


if __name__ == "__main__":
    # This paragraph is for quick manual checks during development.
    # I keep it lightweight so it does not interfere with grading or main.py runs.
    print(load_wdi_macros_japan().head())
    print(load_oil_price().head())
