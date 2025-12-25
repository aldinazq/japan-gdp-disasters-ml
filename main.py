"""
I run the forecasting benchmark from the command line (non-interactive), with strict vs ex-post COVID handling
and optional hyperparameter tuning for Random Forest and Gradient Boosting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models import run_all_models


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _print_block(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    with pd.option_context("display.max_columns", 200, "display.width", 140):
        print(df.to_string(index=False))


def _covid_mode_to_include_covid(covid_mode: str) -> bool:
    if covid_mode == "strict":
        return False
    if covid_mode == "ex_post":
        return True
    raise ValueError("covid_mode must be 'strict' or 'ex_post'.")


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
    include_covid = _covid_mode_to_include_covid(covid_mode)

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
    _print_block(title, df)
    return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Japan GDP vs disasters ML benchmark (forecast/nowcast).")

    p.add_argument(
        "--mode",
        choices=["forecast", "nowcast"],
        default="forecast",
        help="forecast uses only t-1 predictors; nowcast may add current-year disaster info (not strict forecasting).",
    )

    p.add_argument(
        "--covid-mode",
        choices=["strict", "ex_post"],
        default="strict",
        help="strict excludes covid_dummy (not observable ex-ante); ex_post includes it for explanation/robustness.",
    )

    p.add_argument("--tune-rf", action="store_true", help="Tune Random Forest with TimeSeriesSplit CV.")
    p.add_argument("--tune-gb", action="store_true", help="Tune Gradient Boosting with TimeSeriesSplit CV.")

    p.add_argument("--test-ratio", type=float, default=0.2, help="Test split size (time-based). Default: 0.2")

    p.add_argument(
        "--main-start-year",
        type=int,
        default=None,
        help="Optional start year for MAIN sample. Default: no restriction.",
    )

    p.add_argument(
        "--restricted-start-year",
        type=int,
        default=1992,
        help="Start year for RESTRICTED sample. Default: 1992",
    )

    p.add_argument(
        "--only-main",
        action="store_true",
        help="Run only the MAIN benchmark (skip restricted/macro/oil blocks).",
    )

    p.add_argument(
        "--tag-prefix",
        type=str,
        default="run",
        help="Prefix used for saved results files (results/model_metrics_<tag>.csv).",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base = f"{args.tag_prefix}_{args.mode}_{args.covid_mode}"
    if args.tune_rf:
        base += "_tunerf"
    if args.tune_gb:
        base += "_tunegb"

    # MAIN (no macro/oil)
    _run_one(
        title=f"MAIN (no macro/oil) | mode={args.mode} | covid={args.covid_mode} | start_year={args.main_start_year}",
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

    if not args.only_main:
        # RESTRICTED (no macro/oil)
        _run_one(
            title=f"RESTRICTED (no macro/oil) | mode={args.mode} | covid={args.covid_mode} | start_year={args.restricted_start_year}",
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

        # RESTRICTED + MACRO
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

        # RESTRICTED + OIL
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


if __name__ == "__main__":
    main()
