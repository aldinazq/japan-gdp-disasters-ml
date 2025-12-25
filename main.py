"""I run my full GDP-growth forecasting benchmark from the terminal and save each run to results/ so I can build dashboards later."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models import run_all_models

# I anchor paths to this file so I can run `python main.py` from anywhere and it still finds results/.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _print_block(title: str, df: pd.DataFrame) -> None:
    # I print each run as a readable table so I can immediately see which model wins.
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    with pd.option_context("display.max_columns", 200, "display.width", 140):
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
    # I convert the CLI choice into the boolean flag expected by `run_all_models`.
    # If I choose "ex_post", I allow the covid dummy; if I choose "strict", I forbid it (ex-ante logic).
    include_covid = (covid_mode == "ex_post")

    # Then I run the full benchmark and let `src/models.py` write outputs into results/.
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
    p = argparse.ArgumentParser(
        description="Japan GDP growth forecasting with natural-disaster aggregates (ML benchmark)."
    )

    p.add_argument(
        "--mode",
        choices=["forecast", "nowcast"],
        default="forecast",
        help="forecast uses only t-1 predictors; nowcast may also use same-year disaster info.",
    )

    p.add_argument(
        "--covid-mode",
        choices=["strict", "ex_post"],
        default="strict",
        help="strict excludes covid_dummy (ex-ante). ex_post includes it (ex-post robustness).",
    )

    p.add_argument("--tune-rf", action="store_true", help="Tune Random Forest using TimeSeriesSplit.")
    p.add_argument("--tune-gb", action="store_true", help="Tune Gradient Boosting using TimeSeriesSplit.")
    p.add_argument("--test-ratio", type=float, default=0.2, help="Time-based test split ratio. Default: 0.2")

    p.add_argument(
        "--main-start-year",
        type=int,
        default=None,
        help="Optional start year for the MAIN sample (default: no restriction).",
    )
    p.add_argument(
        "--restricted-start-year",
        type=int,
        default=1992,
        help="Start year for the RESTRICTED sample (default: 1992).",
    )

    p.add_argument(
        "--only-main",
        action="store_true",
        help="Run only the MAIN benchmark (skip restricted + controls blocks).",
    )

    p.add_argument(
        "--tag-prefix",
        type=str,
        default="run",
        help="Prefix for saved outputs like results/model_metrics_<tag>.csv",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    # I build one base tag so outputs are easy to find and dashboards can parse the run settings from the filename.
    base = f"{args.tag_prefix}_{args.mode}_{args.covid_mode}"
    if args.tune_rf:
        base += "_tunerf"
    if args.tune_gb:
        base += "_tunegb"

    # 1) MAIN run: clean reference (no extra controls).
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

    if args.only_main:
        return

    # 2) RESTRICTED run: later sub-period to check stability.
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

    # 3) RESTRICTED + MACRO: add WDI macro controls (t-1).
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

    # 4) RESTRICTED + OIL: add oil controls (t-1) as another robustness check.
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
