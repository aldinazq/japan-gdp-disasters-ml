"""I run the full pipeline to build the yearly dataset and evaluate my GDP growth prediction models."""

from src.features import build_master_table
from src.models import run_all_models


def main() -> None:
    df = build_master_table()
    print("=== Master dataset ===")
    print(df.head())
    print(f"\n{df.shape[0]} rows, {df.shape[1]} columns")

    print("\n=== Model evaluation ===")
    metrics = run_all_models()
    print(metrics)


if __name__ == "__main__":
    main()
