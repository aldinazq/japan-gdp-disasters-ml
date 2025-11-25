"""I load the raw GDP, GDP growth, and Japan disaster CSV files into pandas DataFrames for my project."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_gdp_japan_raw() -> pd.DataFrame:
    path = DATA_DIR / "GDP_Japan.csv"
    df = pd.read_csv(path, sep=";")
    return df


def load_gdp_growth_japan_raw() -> pd.DataFrame:
    path = DATA_DIR / "GDP_Growth_japan.csv"
    df = pd.read_csv(path, sep=";")
    return df


def load_japan_disasters_raw() -> pd.DataFrame:
    path = DATA_DIR / "japan_disasters_clean.csv"
    df = pd.read_csv(path, sep=";")
    return df


if __name__ == "__main__":
    gdp = load_gdp_japan_raw()
    growth = load_gdp_growth_japan_raw()
    dis = load_japan_disasters_raw()

    print("GDP_Japan.csv:")
    print(gdp.head())
    print("\nGDP_Growth_japan.csv:")
    print(growth.head())
    print("\nJapan disasters:")
    print(dis.head())
