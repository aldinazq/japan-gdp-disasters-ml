"""I run a simple test to load my data and print basic information about each dataset."""

from src.data_loading import (
    load_gdp_japan_raw,
    load_gdp_growth_japan_raw,
    load_japan_disasters_raw,
)


def main() -> None:
    print("=== Test data loading ===")

    gdp = load_gdp_japan_raw()
    growth = load_gdp_growth_japan_raw()
    disasters = load_japan_disasters_raw()

    print(f"GDP_Japan.csv: {gdp.shape[0]} rows, {gdp.shape[1]} columns")
    print(f"GDP_Growth_japan.csv: {growth.shape[0]} rows, {growth.shape[1]} columns")
    print(f"japan_disasters_clean.csv: {disasters.shape[0]} rows, {disasters.shape[1]} columns")

    print("\nDisasters preview:")
    print(disasters.head())


if __name__ == "__main__":
    main()

