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
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Paths + import safety
# ----------------------------
# This paragraph is for making the script work from any working directory (including the grader's setup).
# I add the project root to sys.path so imports from src/ still work even if I run this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This paragraph is for keeping all outputs in predictable folders, so the report and dashboard can reuse them.
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# This paragraph is for coherence: I reuse the exact same pipeline code as main.py,
# so figures match the results and I avoid "two versions of the truth".
from src.features import build_master_table  # noqa: E402
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


def _imputer() -> SimpleImputer:
    # This paragraph is for compatibility across sklearn versions.
    # Some versions support keep_empty_features, others don't, so I fallback cleanly.
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


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
        test_ratio=default_test_ratio,
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
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "t", "yes", "y"}
        return bool(v)

    mode = str(row["mode"]) if "mode" in metrics_df.columns else "forecast"
    include_macro = _get_bool("include_macro", False)
    include_oil = _get_bool("include_oil", False)
    include_covid = _get_bool("include_covid", False)

    start_year = None
    if "start_year" in metrics_df.columns:
        try:
            sy = int(row["start_year"])
            start_year = None if sy < 0 else sy
        except Exception:
            start_year = None

    test_ratio = float(default_test_ratio)
    if "test_ratio" in metrics_df.columns:
        try:
            test_ratio = float(row["test_ratio"])
        except Exception:
            test_ratio = float(default_test_ratio)

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


def _apply_best_params_if_available(pipe: Pipeline, model_name: str, tag: str) -> Pipeline:
    # This paragraph is for tuned runs.
    # If main.py wrote best_params_*.csv, I load them so the report figures reproduce the tuned model.
    name = str(model_name).strip().lower()
    if name not in {"random_forest", "gradient_boosting"}:
        return pipe

    params_file = None
    if name == "random_forest":
        params_file = RESULTS_DIR / f"best_params_random_forest_{tag}.csv"
    if name == "gradient_boosting":
        params_file = RESULTS_DIR / f"best_params_gradient_boosting_{tag}.csv"

    if params_file is None or not params_file.exists():
        return pipe

    try:
        df = pd.read_csv(params_file)
        if df.empty:
            return pipe
        best_params = df.iloc[0].to_dict()
        best_params = {str(k): v for k, v in best_params.items() if pd.notna(v)}
        pipe.set_params(**best_params)
    except Exception:
        return pipe

    return pipe


def _pick_latest_metrics_file() -> Path:
    # This paragraph is for convenience: if I do not specify a tag,
    # I grab the most recent run so "python make_report_figures.py" still works.
    files = sorted(RESULTS_DIR.glob("model_metrics_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No results/model_metrics_*.csv found. Run main.py first.")
    return files[0]


def _read_metrics(tag: str) -> pd.DataFrame:
    # This paragraph is for safety: figures should fail early with a clear message
    # if results are missing, rather than silently producing empty plots.
    p = RESULTS_DIR / f"model_metrics_{tag}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run main.py first.")
    return pd.read_csv(p)


def _best_model_from_metrics(metrics_df: pd.DataFrame) -> str:
    # This paragraph is for automatic selection: for the report figures,
    # I want the strongest non-baseline model according to test RMSE.
    df = metrics_df.copy()
    df = df[~df["model"].astype(str).str.startswith("baseline")]
    df = df.sort_values("test_RMSE", ascending=True)
    if df.empty:
        return "ridge"
    return str(df.iloc[0]["model"])


def _pipeline_for(model_name: str) -> Optional[Pipeline]:
    # This paragraph is for a single "factory" place where model choices live.
    # It keeps my plotting code simple and reduces the risk of using different hyperparams in different scripts.
    name = model_name.strip().lower()

    if name == "linear_regression":
        return Pipeline([("imputer", _imputer()), ("scaler", StandardScaler()), ("model", LinearRegression())])

    if name == "ridge":
        return Pipeline([("imputer", _imputer()), ("scaler", StandardScaler()), ("model", Ridge(alpha=5.0))])

    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", _imputer()),
                ("model", RandomForestRegressor(n_estimators=600, max_depth=5, min_samples_leaf=2, random_state=42)),
            ]
        )

    if name == "gradient_boosting":
        return Pipeline(
            [
                ("imputer", _imputer()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=42,
                        early_stopping=False,
                        max_depth=2,
                        learning_rate=0.03,
                        max_iter=1200,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                    ),
                ),
            ]
        )

    if name == "neural_net_mlp":
        return Pipeline(
            [
                ("imputer", _imputer()),
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=5000, random_state=42)),
            ]
        )

    if name == "xgboost":
        # This paragraph is for optional dependencies.
        # If xgboost is not installed on the grader machine, I simply skip it instead of failing.
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception:
            return None

        return Pipeline(
            [
                ("imputer", _imputer()),
                (
                    "model",
                    XGBRegressor(
                        random_state=42,
                        objective="reg:squarederror",
                        n_estimators=600,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                    ),
                ),
            ]
        )

    # This paragraph is for robustness: if a metrics file contains an unexpected model name,
    # I let the caller fallback to a supported model instead of crashing here.
    return None


def _save_fig(outpath: Path) -> None:
    # This paragraph is for clean report visuals: tight_layout avoids label cutoffs,
    # and closing figures prevents memory issues when I generate many plots.
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


# ----------------------------
# Plot builders
# ----------------------------
def plot_model_comparison(metrics_df: pd.DataFrame, tag: str) -> Path:
    # This paragraph is for the report: a simple bar chart makes model ranking readable fast.
    df = metrics_df.copy().sort_values("test_RMSE", ascending=True)

    plt.figure()
    plt.bar(df["model"].astype(str), df["test_RMSE"].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test RMSE")
    plt.title(f"Model comparison (tag={tag})")

    out = FIG_DIR / f"model_comparison_{tag}.png"
    _save_fig(out)
    return out


def build_predictions_for_tag(
    tag: str, *, preferred_model: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Pipeline, RunConfig, str]:
    # This paragraph is for generating "actual vs predicted" plots using the same dataset logic as main.py.
    # I rebuild the dataset instead of reusing cached arrays so this script stays standalone.

    metrics_df = _read_metrics(tag)
    cfg = _config_from_metrics(metrics_df, tag)
    include_covid = cfg.covid_mode == "ex_post"

    df_model, X, y = make_dataset(
        include_oil=cfg.include_oil,
        include_macro=cfg.include_macro,
        include_covid=include_covid,
        start_year=cfg.start_year,
        mode=cfg.mode,
    )

    X_train, X_test, y_train, y_test = time_train_test_split(df_model, X, y, test_ratio=cfg.test_ratio)

    # This paragraph is for perfect alignment with the split logic.
    # Using len(X_train) avoids any mismatch from rounding or edge-case guardrails.
    split_idx = int(len(X_train))
    years_test = df_model["year"].iloc[split_idx:].astype(int).to_numpy()

    # This paragraph is for making the script usable: I either use the chosen model
    # or a sensible default, and I fallback if the name is not supported.
    model_name = preferred_model or "gradient_boosting"
    pipe = _pipeline_for(model_name)

    if pipe is not None:
        pipe = _apply_best_params_if_available(pipe, model_name, tag)

    if pipe is None:
        for candidate in ["gradient_boosting", "random_forest", "ridge", "linear_regression", "neural_net_mlp"]:
            pipe = _pipeline_for(candidate)
            if pipe is not None:
                pipe = _apply_best_params_if_available(pipe, candidate, tag)
                model_name = candidate
                break
    if pipe is None:
        raise RuntimeError("Could not instantiate any model pipeline.")

    pipe.fit(X_train, y_train)
    yhat_test = pipe.predict(X_test)

    # This paragraph is for traceability: I save a small table used in LaTeX and debugging,
    # so the figure has a data source on disk.
    pred_df = pd.DataFrame(
        {
            "year": years_test,
            "actual": y_test,
            "pred": yhat_test,
            "error": (y_test - yhat_test),
            "abs_error": np.abs(y_test - yhat_test),
        }
    )
    pred_df.to_csv(FIG_DIR / f"predictions_{tag}_{model_name}.csv", index=False)

    return df_model, X_train, X_test, y_train, y_test, years_test, pipe, cfg, model_name


def plot_actual_vs_pred(pred_df: pd.DataFrame, tag: str, model_name: str) -> Path:
    # This paragraph is for the main story in the report: do predictions track reality over the test years?
    plt.figure()
    plt.plot(pred_df["year"], pred_df["actual"], label="Actual")
    plt.plot(pred_df["year"], pred_df["pred"], label=model_name)
    plt.axhline(0.0, linewidth=1)
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


def plot_errors_vs_disaster_proxy(pred_df: pd.DataFrame, X_test: pd.DataFrame, tag: str) -> Optional[Path]:
    # This paragraph is for a quick sanity check: if disasters matter, errors might move with a disaster proxy.
    # I keep it simple because the sample is small and I only want a visual hint.
    if "damage_share_gdp_lag1" in X_test.columns:
        x = pd.to_numeric(X_test["damage_share_gdp_lag1"], errors="coerce").to_numpy()
        xlab = "damage_share_gdp_lag1"
    elif "n_events_lag1" in X_test.columns:
        x = pd.to_numeric(X_test["n_events_lag1"], errors="coerce").to_numpy()
        xlab = "n_events_lag1"
    else:
        return None

    plt.figure()
    plt.scatter(x, pred_df["error"].to_numpy())
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(xlab)
    plt.ylabel("Error (actual - pred)")
    plt.title(f"Errors vs disaster proxy (tag={tag})")

    out = FIG_DIR / f"errors_vs_disaster_{tag}.png"
    _save_fig(out)
    return out


def plot_feature_importance(
    pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, tag: str, top_k: int = 12
) -> Path:
    # This paragraph is for interpretability in the report.
    # Permutation importance can be noisy with small samples, but it gives a readable ranking fast.
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
    p = RESULTS_DIR / f"cv_scores_{tag}.csv"
    if not p.exists():
        return None

    df = pd.read_csv(p)
    if df.empty or "fold" not in df.columns or "val_RMSE" not in df.columns:
        return None

    pv = df.pivot_table(index="fold", columns="model", values="val_RMSE", aggfunc="mean").sort_index()

    plt.figure()
    for col in pv.columns:
        plt.plot(pv.index.to_numpy(), pv[col].to_numpy(), marker="o", label=str(col))
    plt.xlabel("Fold")
    plt.ylabel("Validation RMSE")
    plt.title(f"TimeSeriesSplit CV stability (tag={tag})")
    plt.legend()

    out = FIG_DIR / f"cv_stability_{tag}.png"
    _save_fig(out)
    return out


def plot_eda_master_tables() -> Dict[str, Path]:
    # This paragraph is for lightweight EDA figures that describe the dataset in the report
    # (GDP dynamics + disaster aggregates), without needing notebooks.
    df = build_master_table().copy()
    df = df.sort_values("year")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for c in ["gdp_growth", "n_events", "total_damage", "total_deaths", "avg_magnitude", "gdp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out_paths: Dict[str, Path] = {}

    # This paragraph is for explaining the target series in the report.
    if "gdp_growth" in df.columns:
        plt.figure()
        plt.plot(df["year"], df["gdp_growth"])
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Year")
        plt.ylabel("GDP growth (%)")
        plt.title("Japan GDP growth over time")
        p1 = FIG_DIR / "eda_gdp_growth.png"
        _save_fig(p1)
        out_paths["eda_gdp_growth"] = p1

    # This paragraph is for showing how the disaster signal looks over time,
    # and whether it has enough variation to be useful for prediction.
    if "n_events" in df.columns and "total_damage" in df.columns:
        log_damage = np.log1p(df["total_damage"].fillna(0.0).to_numpy())
        plt.figure()
        plt.plot(df["year"], df["n_events"], label="n_events")
        plt.plot(df["year"], log_damage, label="log(1+total_damage)")
        plt.xlabel("Year")
        plt.title("Disaster aggregates over time (Japan)")
        plt.legend()
        p2 = FIG_DIR / "eda_disasters.png"
        _save_fig(p2)
        out_paths["eda_disasters"] = p2

        # This paragraph is for scale awareness: damages are heavy-tailed,
        # so a histogram helps explain why I log-transform some features.
        dmg = df["total_damage"].dropna()
        if len(dmg) > 5:
            plt.figure()
            plt.hist(dmg.to_numpy(), bins=30)
            plt.xlabel("total_damage (USD)")
            plt.title("Damage distribution (raw scale)")
            p3 = FIG_DIR / "eda_damage_dist.png"
            _save_fig(p3)
            out_paths["eda_damage_dist"] = p3

    return out_paths


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    # This paragraph is for a simple CLI so I can regenerate all report figures in one command,
    # and so graders can do the same without opening notebooks.
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None, help="Run tag, e.g. run_forecast_strict_main")
    parser.add_argument("--model", type=str, default=None, help="Force model name (optional)")
    parser.add_argument("--top_k", type=int, default=12, help="Top K features for permutation importance")
    args = parser.parse_args()

    # This paragraph is for convenience: if I forget the tag, I default to the latest metrics file.
    if args.tag is None:
        latest = _pick_latest_metrics_file()
        tag = latest.stem.replace("model_metrics_", "")
    else:
        tag = args.tag

    metrics = _read_metrics(tag)

    # This paragraph is for consistency: I pick the best-performing model from the saved metrics,
    # unless I explicitly override it from the command line.
    best_model = _best_model_from_metrics(metrics)
    chosen_model = args.model or best_model

    print(f"[make_report_figures] tag={tag}")
    print(f"[make_report_figures] best_model_from_metrics={best_model} | chosen_model={chosen_model}")
    print(f"[make_report_figures] writing PNG to: {FIG_DIR}")

    # This paragraph is for the report intro/EDA section.
    eda_out = plot_eda_master_tables()
    if eda_out:
        print("[make_report_figures] EDA figures:", ", ".join(str(p.name) for p in eda_out.values()))

    # This paragraph is for the evaluation section: it summarizes test RMSE across models.
    p_comp = plot_model_comparison(metrics, tag)
    print("[make_report_figures] wrote:", p_comp.name)

    # This paragraph is for prediction figures and interpretability figures.
    df_model, X_train, X_test, y_train, y_test, years_test, pipe, cfg, used_model = build_predictions_for_tag(
        tag, preferred_model=chosen_model
    )
    pred_df = pd.read_csv(FIG_DIR / f"predictions_{tag}_{used_model}.csv")

    p_pred = plot_actual_vs_pred(pred_df, tag, used_model)
    print("[make_report_figures] wrote:", p_pred.name)

    p_err = plot_abs_error_by_year(pred_df, tag)
    print("[make_report_figures] wrote:", p_err.name)

    p_scatter = plot_errors_vs_disaster_proxy(pred_df, X_test, tag)
    if p_scatter is not None:
        print("[make_report_figures] wrote:", p_scatter.name)

    p_imp = plot_feature_importance(pipe, X_test, y_test, tag, top_k=int(args.top_k))
    print("[make_report_figures] wrote:", p_imp.name)
    print("[make_report_figures] wrote: feature_importance_table_" + tag + ".csv")

    p_cv = plot_cv_stability(tag)
    if p_cv is not None:
        print("[make_report_figures] wrote:", p_cv.name)

    print("[make_report_figures] done.")


if __name__ == "__main__":
    main()
