"""
I run my full GDP-growth forecasting benchmark from the terminal and save each run in results/ for reproducible grading.
"""

from __future__ import annotations

# This paragraph is for CLI (command-line) usage.
# I use argparse so I can run different configurations from the terminal without editing code.
import argparse

# This paragraph is for stable paths across machines.
# Path makes it easy to build OS-independent file paths (Mac/Windows/Linux).
from pathlib import Path

# This paragraph is for clean type hints.
# Optional[int] means the value can be an int or None.
from typing import Optional

# This paragraph is for pretty printing tables in the terminal.
# I use pandas DataFrames because run_all_models returns a metrics table.
import pandas as pd

# This paragraph is for importing the single entry point of the ML pipeline.
# main.py is only an orchestrator; the real work is in src/models.py.
from src.models import run_all_models

# This paragraph is for stable output paths.
# I anchor the results folder to this file so the script behaves the same on my laptop and on the TA machine.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _print_block(title: str, df: pd.DataFrame) -> None:
    """
    I print a DataFrame in a readable way in the terminal.

    Why it exists:
    - When grading, it is useful to see one clean table per run directly in stdout.
    - I use pandas options to prevent columns from being truncated.
    """
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")

    # This paragraph is for display readability (not for computation).
    # It only changes how pandas prints the DataFrame, not the data itself.
    with pd.option_context(
        "display.max_columns",
        60,
        "display.width",
        160,
        "display.max_colwidth",
        80,
        "display.float_format",
        "{:,.4f}".format,
    ):
        print(df.to_string(index=False))


def _run_one(
    *,
    title: str,
    tag: str,
    mode: str,
    covid_mode: str,
    start_year: Optional[int],
    test_ratio: float,
    tune_rf: bool,
    tune_gb: bool,
    include_macro: bool,
    include_oil: bool,
) -> pd.DataFrame:
    """
    I run one configuration (one “spec”) of the benchmark.

    Why it exists:
    - main() runs multiple specs (main, restricted, restricted+macro, restricted+oil).
    - This helper avoids repeating the same call pattern.
    """
    # This paragraph is for the COVID flag logic.
    # covid_mode controls whether I include a COVID dummy variable:
    # - strict: ex-ante forecasting (do NOT include COVID dummy)
    # - ex_post: robustness (include COVID dummy)
    include_covid = covid_mode == "ex_post"

    # This paragraph is the actual pipeline call.
    # run_all_models returns a DataFrame of model-level metrics and also writes CSVs to results/.
    df = run_all_models(
        include_oil=include_oil,
        include_macro=include_macro,
        include_covid=include_covid,
        start_year=start_year,
        test_ratio=test_ratio,
        tune_rf=tune_rf,
        tune_gb=tune_gb,
        mode=mode,
        tag=tag,
    )

    # This paragraph is for terminal visibility.
    # I print the returned DataFrame so that every run has a clear summary in stdout.
    _print_block(title, df)
    return df


def build_parser() -> argparse.ArgumentParser:
    """
    I define the CLI arguments used to control the benchmark.

    Why it exists:
    - It makes runs reproducible (all settings are explicitly logged in the command you ran).
    - It allows the TA to reproduce outputs with one command.
    """
    p = argparse.ArgumentParser(
        description="Japan GDP growth forecasting benchmark (CLI)."
    )

    # This paragraph is for the modeling mode.
    # forecast = strict lagged predictors; nowcast = allow contemporaneous controls (if available in features).
    p.add_argument(
        "--mode",
        type=str,
        default="forecast",
        choices=["forecast", "nowcast"],
        help="forecast: lagged-only predictors; nowcast: allow contemporaneous covariates if present.",
    )

    # This paragraph is for COVID handling.
    # strict: exclude covid dummy; ex_post: include covid dummy.
    p.add_argument(
        "--covid-mode",
        type=str,
        default="strict",
        choices=["strict", "ex_post"],
        help="strict: exclude COVID dummy (ex-ante). ex_post: include COVID dummy (robustness).",
    )

    # This paragraph is for optional tuning switches.
    # Tuning is slower, so I keep it optional.
    p.add_argument(
        "--tune-rf",
        action="store_true",
        help="If set, run a small RF grid search (time-series CV).",
    )
    p.add_argument(
        "--tune-gb",
        action="store_true",
        help="If set, tune Gradient Boosting via time-series CV.",
    )

    # This paragraph is for the test split.
    # Since this is a time series, I split by time (last portion is test), not random split.
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of observations in the test set (time-based split).",
    )

    # This paragraph is for sample-window control.
    # main-start-year allows restricting even the main run if needed.
    p.add_argument(
        "--main-start-year",
        type=int,
        default=None,
        help="Optional start year for the main run (default: full sample).",
    )

    # This paragraph is for the restricted sample run start year.
    # The restricted runs use this start year to avoid early-era inconsistencies.
    p.add_argument(
        "--restricted-start-year",
        type=int,
        default=1992,
        help="Start year for restricted sample runs (default: 1992).",
    )

    # This paragraph is for quickly running a single baseline configuration.
    p.add_argument(
        "--only-main",
        action="store_true",
        help="If set, only run the baseline main specification.",
    )

    # This paragraph is for consistent filenames.
    # All results are saved with tags like: {tag-prefix}_{mode}_{covid-mode}_...
    p.add_argument(
        "--tag-prefix",
        type=str,
        default="run",
        help="Prefix for all output tags, used in results filenames.",
    )

    return p


def main() -> None:
    """
    I run the full benchmark suite from the CLI.

    Why it exists:
    - It orchestrates multiple specifications consistently.
    - It ensures the folder structure exists and tags are consistent.
    """
    args = build_parser().parse_args()

    # This paragraph is for stable tag names.
    # I encode mode + covid_mode + tuning flags in the base tag for traceability.
    base = f"{args.tag_prefix}_{args.mode}_{args.covid_mode}"
    if args.tune_rf:
        base += "_tunerf"
    if args.tune_gb:
        base += "_tunegb"

    # ==========================
    # 1) MAIN baseline run
    # ==========================
    # This paragraph is for the primary benchmark spec.
    # It uses the full sample by default and includes only disaster + lagged GDP features.
    _run_one(
        title=f"MAIN | mode={args.mode} | covid={args.covid_mode} | start_year={args.main_start_year}",
        tag=f"{base}_main",
        mode=args.mode,
        covid_mode=args.covid_mode,
        start_year=args.main_start_year,
        test_ratio=args.test_ratio,
        tune_rf=args.tune_rf,
        tune_gb=args.tune_gb,
        include_macro=False,
        include_oil=False,
    )

    # This paragraph is for skipping robustness runs.
    # Useful when you want a fast check or when grading only needs the baseline.
    if args.only_main:
        return

    # ==========================
    # 2) RESTRICTED baseline run
    # ==========================
    # This paragraph is for a robustness check on a more recent sample window.
    _run_one(
        title=f"RESTRICTED | mode={args.mode} | covid={args.covid_mode} | start_year={args.restricted_start_year}",
        tag=f"{base}_restricted",
        mode=args.mode,
        covid_mode=args.covid_mode,
        start_year=args.restricted_start_year,
        test_ratio=args.test_ratio,
        tune_rf=args.tune_rf,
        tune_gb=args.tune_gb,
        include_macro=False,
        include_oil=False,
    )

    # ==========================
    # 3) RESTRICTED + MACRO run
    # ==========================
    # This paragraph is for testing whether extra macro controls improve performance.
    # I separate this run so the gain is clearly attributed to the macro block.
    _run_one(
        title=f"RESTRICTED + MACRO | mode={args.mode} | covid={args.covid_mode} | start_year={args.restricted_start_year}",
        tag=f"{base}_restricted_macro",
        mode=args.mode,
        covid_mode=args.covid_mode,
        start_year=args.restricted_start_year,
        test_ratio=args.test_ratio,
        tune_rf=args.tune_rf,
        tune_gb=args.tune_gb,
        include_macro=True,
        include_oil=False,
    )

    # ==========================
    # 4) RESTRICTED + OIL run
    # ==========================
    # This paragraph is for testing whether oil shocks add predictive power.
    _run_one(
        title=f"RESTRICTED + OIL | mode={args.mode} | covid={args.covid_mode} | start_year={args.restricted_start_year}",
        tag=f"{base}_restricted_oil",
        mode=args.mode,
        covid_mode=args.covid_mode,
        start_year=args.restricted_start_year,
        test_ratio=args.test_ratio,
        tune_rf=args.tune_rf,
        tune_gb=args.tune_gb,
        include_macro=False,
        include_oil=True,
    )


# This paragraph is for the standard Python entry point.
# It ensures that main() only runs when you execute: python main.py
if __name__ == "__main__":
    main()
