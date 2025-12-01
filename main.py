"""I run the full pipeline and provide a small interactive CLI to inspect GDP growth predictions by year."""

from pathlib import Path

import pandas as pd

from src.features import build_master_table
from src.models import run_all_models

PROJECT_ROOT = Path(__file__).resolve().parent


def interactive_cli() -> None:
    predictions_path = PROJECT_ROOT / "results" / "random_forest_predictions.csv"
    if not predictions_path.exists():
        print("\nNo predictions file found in results/random_forest_predictions.csv.")
        print("Run the models first to generate predictions.")
        return

    pred_df = pd.read_csv(predictions_path)
    print("\n=== Interactive: Random Forest prediction vs actual by year ===")
    print(
        "Type a year between {} and {}, or 'q' to quit.".format(
            int(pred_df["year"].min()), int(pred_df["year"].max())
        )
    )

    while True:
        user_inp = input("Year (or q to quit): ").strip()
        if user_inp.lower() in {"q", "quit", "exit"}:
            print("Exiting interactive mode.")
            break

        try:
            year = int(user_inp)
        except ValueError:
            print("Please enter a valid integer year.")
            continue

        row = pred_df[pred_df["year"] == year]
        if row.empty:
            print("No data available for that year.")
            continue

        row = row.iloc[0]
        print(f"\nYear {year} [{row['set']}]")
        print(f"  Actual GDP growth:         {row['gdp_growth_actual']:.2f}%")
        print(f"  Predicted (Random Forest): {row['gdp_growth_pred_rf']:.2f}%\n")


def main() -> None:
    df = build_master_table()
    print("=== Master dataset ===")
    print(df.head())
    print(f"\n{df.shape[0]} rows, {df.shape[1]} columns\n")

    print("=== Model evaluation ===")
    metrics = run_all_models()
    print(metrics.to_string(index=False))

    interactive_cli()


if __name__ == "__main__":
    main()
