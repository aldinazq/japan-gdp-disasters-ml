"""I run the full pipeline (main + robustness) and provide a small interactive CLI to inspect predictions by year."""

from pathlib import Path
import shutil

import pandas as pd

from src.features import build_master_table
from src.models import run_all_models

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _copy_results(suffix: str) -> None:
    """I copy the default results files to versioned filenames so runs don't overwrite each other."""
    RESULTS_DIR.mkdir(exist_ok=True)

    src_metrics = RESULTS_DIR / "model_metrics.csv"
    src_preds = RESULTS_DIR / "random_forest_predictions.csv"

    if src_metrics.exists():
        shutil.copyfile(src_metrics, RESULTS_DIR / f"model_metrics_{suffix}.csv")
    if src_preds.exists():
        shutil.copyfile(src_preds, RESULTS_DIR / f"random_forest_predictions_{suffix}.csv")


def interactive_cli(predictions_file: str = "random_forest_predictions_main.csv") -> None:
    """I let the user query predictions by year from a chosen predictions CSV."""
    predictions_path = RESULTS_DIR / predictions_file
    if not predictions_path.exists():
        print(f"\nNo predictions file found: results/{predictions_file}")
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

    # -------------------------
    # MAIN RUN (no oil): preferred because it usually keeps more years
    # -------------------------
    print("=== Model evaluation (MAIN: no oil) ===")
    metrics_main = run_all_models(include_oil=False)
    print(metrics_main.to_string(index=False))
    _copy_results("main")

    # -------------------------
    # ROBUSTNESS RUN (with oil): may reduce the sample, used as a robustness check
    # -------------------------
    print("\n=== Model evaluation (ROBUSTNESS: with oil) ===")
    metrics_robust = run_all_models(include_oil=True)
    print(metrics_robust.to_string(index=False))
    _copy_results("robust")

    print("\nSaved results files:")
    print(" - results/model_metrics_main.csv")
    print(" - results/random_forest_predictions_main.csv")
    print(" - results/model_metrics_robust.csv")
    print(" - results/random_forest_predictions_robust.csv")

    # Interactive uses the MAIN run by default
    interactive_cli(predictions_file="random_forest_predictions_main.csv")


if __name__ == "__main__":
    main()
