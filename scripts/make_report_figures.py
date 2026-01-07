"""I generate report-ready PNG figures from my project outputs (data/, results/, src/) and save them into figures/."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
SRC_DIR = ROOT / "src"

FIG_DIR.mkdir(exist_ok=True)

# This paragraph is for local imports without installing as a package.
# I add the project root to sys.path so `from src...` works when this script is run directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# IMPORTANT: I import the dataset builder used by the main pipeline so my figures match the run settings.
from src.models import make_dataset, time_train_test_split  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    tag: str
    mode: str  # "forecast" or "nowcast"
    covid_mode: str  # "strict" or "ex_post"
    include_macro: bool
    include_oil: bool
    start_year: Optional[int]
    test_ratio: float


# ----------------------------
# Helpers
# ----------------------------
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # This paragraph is for using one metric definition everywhere (models, tables, and figures).
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _read_metrics(tag: str) -> pd.DataFrame:
    p = RESULTS_DIR / f"model_metrics_{tag}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_cv_scores(tag: str) -> pd.DataFrame:
    p = RESULTS_DIR / f"cv_scores_{tag}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _pick_latest_metrics_file() -> Path:
    files = sorted(RESULTS_DIR.glob("model_metrics_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No model_metrics_*.csv found in results/. Run main.py first.")
    return files[-1]


def _best_model_from_metrics(metrics_df: pd.DataFrame) -> str:
    # This paragraph is for robustness: if I cannot read metrics, I fall back to a sensible default.
    if metrics_df is None or metrics_df.empty or "test_RMSE" not in metrics_df.columns:
        return "gradient_boosting"
    return metrics_df.sort_values("test_RMSE", ascending=True).iloc[0]["model"]


def _infer_config_from_tag(tag: str, default_test_ratio: float = 0.2) -> RunConfig:
    # This paragraph is for reducing manual inputs: the run tag already encodes key choices,
    # so I infer the config to avoid generating figures with mismatched settings.
    t = tag.lower()
    mode = "nowcast" if "nowcast" in t else "forecast"
    covid_mode = "ex_post" if "ex_post" in t else "strict"
    include_macro = ("macro" in t) or ("withmacro" in t)
    include_oil = ("oil" in t)
    start_year = 1992 if "restricted" in t else None
    return RunConfig(
        tag=tag,
        mode=mode,
        covid_mode=covid_mode,
        include_macro=include_macro,
        include_oil=include_oil,
        start_year=start_year,
        test_ratio=float(default_test_ratio),
    )


def _config_from_metrics(metrics_df: pd.DataFrame, tag: str, *, default_test_ratio: float = 0.2) -> RunConfig:
    # This paragraph is for perfect reproducibility.
    # Instead of inferring flags from the tag string (which can drift),
    # I rebuild the figure config from the metrics CSV that main.py wrote.
    if metrics_df is None or metrics_df.empty:
        return _infer_config_from_tag(tag, default_test_ratio=default_test_ratio)

    row = metrics_df.iloc[0]

    def _get_bool(col: str, default: bool = False) -> bool:
        if col not in metrics_df.columns:
            return bool(default)
        v = row[col]
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(int(v))
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "t", "yes", "y"}
        return bool(default)

    def _get_str(col: str, default: str) -> str:
        if col not in metrics_df.columns:
            return default
        v = row[col]
        return default if pd.isna(v) else str(v)

    def _get_float(col: str, default: float) -> float:
        if col not in metrics_df.columns:
            return float(default)
        v = row[col]
        try:
            return float(v)
        except Exception:
            return float(default)

    def _get_int(col: str, default: Optional[int]) -> Optional[int]:
        if col not in metrics_df.columns:
            return default
        v = row[col]
        if pd.isna(v):
            return default
        try:
            return int(v)
        except Exception:
            return default

    mode = _get_str("mode", "forecast")
    include_covid = _get_bool("include_covid", default=False)
    include_macro = _get_bool("include_macro", default=False)
    include_oil = _get_bool("include_oil", default=False)
    start_year = _get_int("start_year", default=None)
    test_ratio = _get_float("test_ratio", default_test_ratio)

    covid_mode = "ex_post" if include_covid else "strict"

    return RunConfig(
        tag=tag,
        mode=mode,
        covid_mode=covid_mode,
        include_macro=include_macro,
        include_oil=include_oil,
        start_year=start_year,
        test_ratio=test_ratio,
    )


# ----------------------------
# Model utilities (lightweight, just for figures)
# ----------------------------
def _make_model(name: str) -> object:
    if name == "linear_regression":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=42)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=500, random_state=42)
    # default: gradient boosting
    return HistGradientBoostingRegressor(random_state=42)


def build_predictions_for_tag(
    tag: str, *, preferred_model: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Pipeline, RunConfig, str]:
    """
    I rebuild the dataset exactly like main.py/run_all_models (macro/oil/covid + optional start_year),
    then refit the chosen model to regenerate prediction and diagnostic figures.
    """
    metrics = _read_metrics(tag)
    cfg = _config_from_metrics(metrics, tag)

    # These flags are the exact knobs used by src/models.make_dataset().
    include_covid = bool(cfg.covid_mode == "ex_post")
    include_macro = bool(cfg.include_macro)
    include_oil = bool(cfg.include_oil)
    start_year = cfg.start_year

    df_model, X, y = make_dataset(
        include_oil=include_oil,
        include_macro=include_macro,
        include_covid=include_covid,
        start_year=start_year,
        mode=cfg.mode,
    )

    X_train, X_test, y_train, y_test = time_train_test_split(df_model, X, y, test_ratio=cfg.test_ratio)
    split = int(len(y_train))
    years_test = df_model["year"].iloc[split:].to_numpy()

    used_model = preferred_model or _best_model_from_metrics(metrics)
    model = _make_model(used_model)

    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    pred_df = pd.DataFrame({"year": years_test, "actual": y_test, "pred": pred})
    pred_df["error"] = pred_df["actual"] - pred_df["pred"]
    pred_df["abs_error"] = pred_df["error"].abs()
    pred_df.to_csv(FIG_DIR / f"predictions_{tag}_{used_model}.csv", index=False)

    return df_model, X_train, X_test, y_train, y_test, years_test, pipe, cfg, used_model


# ----------------------------
# Plots
# ----------------------------
def plot_model_comparison(metrics_df: pd.DataFrame, tag: str) -> Path:
    # This paragraph is for an at-a-glance leaderboard plot for the report.
    if metrics_df is None or metrics_df.empty:
        raise FileNotFoundError(f"No metrics found for tag={tag}. Run main.py first.")

    df = metrics_df.copy()
    df = df.sort_values("test_RMSE", ascending=True)

    plt.figure()
    plt.bar(df["model"].astype(str), df["test_RMSE"].astype(float))
    plt.ylabel("Test RMSE")
    plt.title(f"Model comparison (tag={tag})")
    plt.xticks(rotation=35, ha="right")

    out = FIG_DIR / f"model_comparison_{tag}.png"
    _save_fig(out)
    return out


def plot_actual_vs_pred(pred_df: pd.DataFrame, tag: str, model_name: str) -> Path:
    # This paragraph is for showing whether the best model tracks the realized series.
    df = pred_df.sort_values("year")

    plt.figure()
    plt.plot(df["year"], df["actual"], label="Actual")
    plt.plot(df["year"], df["pred"], label=model_name)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Year (test set)")
    plt.ylabel("GDP growth (%)")
    plt.title(f"Actual vs Predicted (tag={tag})")
    plt.legend()

    out = FIG_DIR / f"pred_vs_actual_{tag}.png"
    _save_fig(out)
    return out


def plot_abs_error_by_year(pred_df: pd.DataFrame, tag: str) -> Path:
    # This paragraph is for diagnosing unstable years (like crisis periods),
    # because averages can hide big misses on specific dates.
    df = pred_df.sort_values("year")

    plt.figure()
    plt.bar(df["year"].astype(int), df["abs_error"].astype(float))
    plt.xlabel("Year (test set)")
    plt.ylabel("|Actual - Pred|")
    plt.title(f"Absolute error by year (tag={tag})")

    out = FIG_DIR / f"error_by_year_{tag}.png"
    _save_fig(out)
    return out


def plot_errors_vs_disaster_proxy(pred_df: pd.DataFrame, df_model: pd.DataFrame, tag: str) -> Optional[Path]:
    # This paragraph is for a sanity check: if errors correlate with disaster intensity,
    # the model is missing nonlinearities or interactions tied to disasters.
    if "damage_share_gdp_lag1" not in df_model.columns:
        return None

    df = pred_df.merge(df_model[["year", "damage_share_gdp_lag1"]], on="year", how="left")
    x = pd.to_numeric(df["damage_share_gdp_lag1"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["error"], errors="coerce").to_numpy()

    plt.figure()
    plt.scatter(x, y)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("damage_share_gdp_lag1")
    plt.ylabel("Error (actual - pred)")
    plt.title(f"Errors vs disaster proxy (tag={tag})")

    out = FIG_DIR / f"errors_vs_disaster_{tag}.png"
    _save_fig(out)
    return out


def plot_permutation_importance(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    tag: str,
    *,
    top_k: int = 12,
) -> Path:
    # This paragraph is for interpretability: permutation importance tells which features matter for predictions.
    res = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    imp = pd.DataFrame(
        {"feature": X_test.columns, "importance": res.importances_mean, "std": res.importances_std}
    ).sort_values("importance", ascending=False)

    # This paragraph is for reproducibility: I save the table so the figure is not "just a picture".
    imp.to_csv(FIG_DIR / f"feature_importance_table_{tag}.csv", index=False)

    top = imp.head(top_k).copy()
    plt.figure()
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("Permutation importance (ΔRMSE)")
    plt.title(f"Top {top_k} features (tag={tag})")

    out = FIG_DIR / f"feature_importance_{tag}.png"
    _save_fig(out)
    return out


def plot_cv_stability(tag: str) -> Optional[Path]:
    # This paragraph is for showing stability across CV folds.
    # If one model jumps a lot across folds, I do not trust its performance as much.
    df = _read_cv_scores(tag)
    if df is None or df.empty or "fold" not in df.columns or "val_RMSE" not in df.columns:
        return None

    pv = df.pivot_table(index="fold", columns="model", values="val_RMSE", aggfunc="mean").sort_index()

    plt.figure()
    for col in pv.columns:
        plt.plot(pv.index.to_numpy(), pv[col].to_numpy(), marker="o", label=str(col))

    # This paragraph is for readability: one unstable model can explode and hide the rest,
    # so I default to a log y-axis when RMSE values are strictly positive.
    ymin = np.nanmin(pv.to_numpy())
    if np.isfinite(ymin) and ymin > 0:
        plt.yscale("log")
        plt.ylabel("Validation RMSE (log scale)")
    else:
        plt.yscale("symlog", linthresh=1e-3)
        plt.ylabel("Validation RMSE (symlog)")

    plt.xlabel("Fold")
    plt.title(f"TimeSeriesSplit CV stability (tag={tag})")
    plt.legend()
    plt.grid(True, which="both", axis="y", alpha=0.3)

    out = FIG_DIR / f"cv_stability_{tag}.png"
    _save_fig(out)
    return out


def plot_eda_master_tables() -> Dict[str, Path]:
    # This paragraph is for lightweight EDA figures that describe the dataset in the report,
    # without needing notebooks (GDP dynamics + disaster aggregates + damage distribution).
    df_model, X, y = make_dataset(include_oil=False, include_macro=False, include_covid=False, start_year=None, mode="forecast")
    df = df_model.copy().sort_values("year")

    out_paths: Dict[str, Path] = {}

    # Target series
    if "gdp_growth" in df.columns:
        plt.figure()
        plt.plot(df["year"], df["gdp_growth"])
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("Year")
        plt.ylabel("GDP growth (%)")
        plt.title("Japan GDP growth over time")
        p1 = FIG_DIR / "eda_gdp_growth.png"
        _save_fig(p1)
        out_paths["eda_gdp_growth"] = p1

    # Disaster aggregates
    if "n_events" in df.columns and "total_damage" in df.columns:
        plt.figure()
        plt.plot(df["year"], df["n_events"], label="n_events")
        plt.plot(df["year"], np.log1p(pd.to_numeric(df["total_damage"], errors="coerce").fillna(0.0)), label="log(1+total_damage)")
        plt.xlabel("Year")
        plt.title("Disaster aggregates over time (Japan)")
        plt.legend()
        p2 = FIG_DIR / "eda_disasters.png"
        _save_fig(p2)
        out_paths["eda_disasters"] = p2

        dmg = pd.to_numeric(df["total_damage"], errors="coerce").dropna()
        if len(dmg) > 5:
            # Raw-scale histogram
            plt.figure()
            plt.hist(dmg.clip(lower=0).to_numpy(), bins=30)
            plt.xlabel("total_damage (USD)")
            plt.title("Damage distribution (raw scale)")
            p3 = FIG_DIR / "eda_damage_dist.png"
            _save_fig(p3)
            out_paths["eda_damage_dist"] = p3

            # Log-scale histogram
            dmg_log = np.log1p(dmg.clip(lower=0))
            plt.figure()
            plt.hist(dmg_log.to_numpy(), bins=30)
            plt.xlabel("log(1 + total_damage)")
            plt.title("Damage distribution (log1p scale)")
            p4 = FIG_DIR / "eda_damage_dist_log.png"
            _save_fig(p4)
            out_paths["eda_damage_dist_log"] = p4

    return out_paths


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    # This paragraph is for a simple CLI so I can regenerate all report figures in one command.
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None, help="Run tag, e.g. run_forecast_strict_main")
    parser.add_argument("--model", type=str, default=None, help="Force model name (optional)")
    parser.add_argument("--top_k", type=int, default=12, help="Top K features for permutation importance")
    args = parser.parse_args()

    # If tag is not provided, I default to the latest metrics file.
    if args.tag is None:
        latest = _pick_latest_metrics_file()
        tag = latest.stem.replace("model_metrics_", "")
    else:
        tag = args.tag

    metrics = _read_metrics(tag)
    best_model = _best_model_from_metrics(metrics)
    chosen_model = args.model or best_model

    print(f"[make_report_figures] tag={tag}")
    print(f"[make_report_figures] best_model_from_metrics={best_model} | chosen_model={chosen_model}")
    print(f"[make_report_figures] writing PNG to: {FIG_DIR}")

    eda_out = plot_eda_master_tables()
    if eda_out:
        print("[make_report_figures] EDA figures:", ", ".join(str(p.name) for p in eda_out.values()))

    p_comp = plot_model_comparison(metrics, tag)
    print("[make_report_figures] wrote:", p_comp.name)

    df_model, X_train, X_test, y_train, y_test, years_test, pipe, cfg, used_model = build_predictions_for_tag(
        tag, preferred_model=chosen_model
    )
    pred_df = pd.read_csv(FIG_DIR / f"predictions_{tag}_{used_model}.csv")

    p_pred = plot_actual_vs_pred(pred_df, tag, used_model)
    print("[make_report_figures] wrote:", p_pred.name)

    p_abs = plot_abs_error_by_year(pred_df, tag)
    print("[make_report_figures] wrote:", p_abs.name)

    p_err = plot_errors_vs_disaster_proxy(pred_df, df_model, tag)
    if p_err is not None:
        print("[make_report_figures] wrote:", p_err.name)

    p_imp = plot_permutation_importance(pipe, X_test, y_test, tag, top_k=args.top_k)
    print("[make_report_figures] wrote:", p_imp.name)

    p_cv = plot_cv_stability(tag)
    if p_cv is not None:
        print("[make_report_figures] wrote:", p_cv.name)


if __name__ == "__main__":
    main()
