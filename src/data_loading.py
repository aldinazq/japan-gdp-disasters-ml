"""I load and reshape the GDP, GDP growth, and natural disaster Excel files into tidy pandas DataFrames for my Japan project."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_gdp_japan_raw() -> pd.DataFrame:
    """Load GDP (current US$) for Japan and reshape it into year/value."""
    path = DATA_DIR / "GDP_Japan.xlsx"
    df = pd.read_excel(path, sheet_name="Data", skiprows=3)
    df = df[df["Country Code"] == "JPN"].dropna(axis=1, how="all")

    df_long = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="year",
        value_name="gdp"
    )
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long = df_long.dropna(subset=["gdp", "year"])
    return df_long[["year", "gdp"]].reset_index(drop=True)


def load_gdp_growth_japan_raw() -> pd.DataFrame:
    """Load GDP growth (annual %) for Japan and reshape it into year/value."""
    path = DATA_DIR / "GDP_Growth_japan.xlsx"
    df = pd.read_excel(path, sheet_name="Data", skiprows=3)
    df = df[df["Country Code"] == "JPN"].dropna(axis=1, how="all")

    df_long = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="year",
        value_name="gdp_growth"
    )
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long = df_long.dropna(subset=["gdp_growth", "year"])
    return df_long[["year", "gdp_growth"]].reset_index(drop=True)


def load_japan_disasters_raw() -> pd.DataFrame:
    """Load natural disasters dataset and clean relevant columns."""
    path = DATA_DIR / "Natural_disasters_Japan.xlsx"
    df = pd.read_excel(path, sheet_name="EM-DAT Data")

    keep_cols = [
        "Disaster Type", "Disaster Subtype", "Event Name", "Magnitude",
        "Magnitude Scale", "Total Deaths", "Total Damage ('000 US$)", "Start Year"
    ]
    df = df[keep_cols].rename(columns={
        "Start Year": "year",
        "Total Deaths": "deaths",
        "Total Damage ('000 US$)": "damage_usd_thousands",
        "Disaster Type": "disaster_type",
        "Disaster Subtype": "disaster_subtype",
        "Event Name": "event_name",
        "Magnitude": "magnitude",
        "Magnitude Scale": "magnitude_scale"
    })
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    gdp = load_gdp_japan_raw()
    growth = load_gdp_growth_japan_raw()
    dis = load_japan_disasters_raw()

    print("GDP_Japan.xlsx:")
    print(gdp.head(), "\n")
    print("GDP_Growth_japan.xlsx:")
    print(growth.head(), "\n")
    print("Natural_disasters_Japan.xlsx:")
    print(dis.head())
