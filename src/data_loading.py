"""I load and reshape GDP, GDP growth, disasters, and extra macro indicators into tidy pandas DataFrames for my Japan project."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_gdp_japan_raw() -> pd.DataFrame:
    """I load Japan GDP level (current US$) from a WDI-style Excel extract and return [year, gdp]."""
    path = DATA_DIR / "GDP_Japan.xlsx"
    df = pd.read_excel(path, sheet_name="Data", skiprows=3)
    df = df[df["Country Code"] == "JPN"].dropna(axis=1, how="all")

    df_long = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="year",
        value_name="gdp",
    )
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long["gdp"] = pd.to_numeric(df_long["gdp"], errors="coerce")
    df_long = df_long.dropna(subset=["gdp", "year"])

    return df_long[["year", "gdp"]].sort_values("year").reset_index(drop=True)


def load_gdp_growth_japan_raw() -> pd.DataFrame:
    """I load Japan GDP growth (annual %) from a WDI-style Excel extract and return [year, gdp_growth]."""
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
    """I load raw EM-DAT disasters for Japan and return a cleaned event-level DataFrame."""
    path = DATA_DIR / "Natural_disasters_Japan.xlsx"
    df = pd.read_excel(path, sheet_name="EM-DAT Data")

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

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # Coerce numeric columns
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df["damage_usd_thousands"] = pd.to_numeric(df["damage_usd_thousands"], errors="coerce")
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")

    return df.reset_index(drop=True)


# --- Extra macro data loaders (WDI + oil) ---

def _load_wdi_extract(file_name: str, value_name: str) -> pd.DataFrame:
    """
    I read a World Bank DataBank WDI extract (wide year columns) and return a tidy DataFrame [year, value_name].
    Handles WDI missing marker '..' by converting to NaN.
    """
    path = DATA_DIR / file_name

    df = pd.read_excel(path, sheet_name="Data")
    if "Series Code" not in df.columns or "Country Code" not in df.columns:
        # Some WDI extracts have metadata rows at the top
        df = pd.read_excel(path, sheet_name="Data", skiprows=3)

    df = df[df["Series Code"].notna()].copy()
    df = df[df["Country Code"] == "JPN"].copy()

    year_cols = [c for c in df.columns if isinstance(c, str) and "[YR" in c]

    long = df.melt(
        id_vars=["Series Code", "Country Code"],
        value_vars=year_cols,
        var_name="year_raw",
        value_name=value_name,
    )

    # Convert WDI missing marker ".." (and any non-numeric) to NaN
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")

    long["year"] = long["year_raw"].str.extract(r"\[YR(\d{4})\]")[0].astype(int)
    long = long.dropna(subset=[value_name])

    return long[["year", value_name]].sort_values("year").reset_index(drop=True)


def load_wdi_macros_japan() -> pd.DataFrame:
    """I load and merge WDI macro indicators for Japan into a single DataFrame keyed by year."""
    inflation = _load_wdi_extract("wdi_inflation.xlsx", "inflation_cpi")
    exports = _load_wdi_extract("wdi_exports_pct_gdp.xlsx", "exports_pct_gdp")
    unemp = _load_wdi_extract("wdi_unemployment.xlsx", "unemployment_rate")
    invest = _load_wdi_extract("wdi_investment_pct_gdp.xlsx", "investment_pct_gdp")
    fx = _load_wdi_extract("wdi_fx_jpy_per_usd.xlsx", "fx_jpy_per_usd")

    df = inflation.merge(exports, on="year", how="outer")
    df = df.merge(unemp, on="year", how="outer")
    df = df.merge(invest, on="year", how="outer")
    df = df.merge(fx, on="year", how="outer")

    return df.sort_values("year").reset_index(drop=True)


def load_oil_price() -> pd.DataFrame:
    """I load annual oil prices and return [year, oil_price_usd]."""
    path = DATA_DIR / "oil_price.xlsx"
    df = pd.read_excel(path)

    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    value_col = "POILAPSPUSDA" if "POILAPSPUSDA" in df.columns else df.columns[1]

    df["year"] = pd.to_datetime(df[date_col]).dt.year
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.rename(columns={value_col: "oil_price_usd"}).dropna(subset=["oil_price_usd"])

    return df[["year", "oil_price_usd"]].sort_values("year").reset_index(drop=True)


if __name__ == "__main__":
    print(load_wdi_macros_japan().head())
    print(load_oil_price().head())
