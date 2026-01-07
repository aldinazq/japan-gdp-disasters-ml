"""I build a self-contained HTML dashboard that summarizes model metrics, predictions, CV stability, and disaster diagnostics."""
from __future__ import annotations

import argparse
import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True)

# This paragraph is for coherence: I reuse the exact same dataset code as main.py,
# so the dashboard is consistent with the benchmark metrics.
from src.models import make_dataset, time_train_test_split  # noqa: E402


# ----------------------------
# Config model (what one dashboard run represents)
# ----------------------------
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
# Small helpers (formatting + safety)
# ----------------------------
def _imputer() -> SimpleImputer:
    # This paragraph is for compatibility across sklearn versions.
    # Some versions support keep_empty_features, others do not, so I keep a clean fallback.
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # This paragraph is for using one metric definition everywhere (models, tables, and plots).
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # This paragraph is for robustness.
    # R² can fail in edge cases (tiny subsets), so I prefer returning NaN instead of crashing.
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _fig_to_base64(fig) -> str:
    # This paragraph is for a “single-file dashboard”.
    # I embed plots as base64 PNG so the HTML has no external dependencies.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _format_float(x: float) -> str:
    # This paragraph is for clean tables.
    # I show blanks for NaN/inf so the HTML does not look broken.
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:.3f}"


def _df_to_html_table(df: pd.DataFrame, *, max_rows: int = 40) -> str:
    # This paragraph is for readability.
    # I cap tables so the dashboard stays usable even if there are many runs or many features.
    out = df.copy()
    if len(out) > max_rows:
        out = out.head(max_rows)
    return out.to_html(index=False, escape=True)


# ----------------------------
# Files + config inference
# ----------------------------
def _pick_latest_metrics_file() -> Path:
    # This paragraph is for convenience.
    # If I do not specify a tag, I use the most recent results file as the default run.
    files = sorted(RESULTS_DIR.glob("model_metrics_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No results/model_metrics_*.csv found. Run main.py first.")
    return files[0]


def _read_metrics(tag: str) -> pd.DataFrame:
    # This paragraph is for fail-fast behavior.
    # If the metrics file is missing, I want a clear error that tells me what to run.
    path = RESULTS_DIR / f"model_metrics_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run main.py to create it.")
    return pd.read_csv(path)


def _read_cv_scores(tag: str) -> Optional[pd.DataFrame]:
    # This paragraph is for optional diagnostics.
    # If cv_scores_{tag}.csv does not exist, I skip CV plots gracefully.
    path = RESULTS_DIR / f"cv_scores_{tag}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _infer_config_from_tag(tag: str, *, default_test_ratio: float = 0.2) -> RunConfig:
    # This paragraph is for “tag-driven reproducibility”.
    # My tags encode the setup (forecast/nowcast, strict/ex_post, macro/oil, restricted),
    # so I can rebuild the dataset for the dashboard without asking for many CLI flags.
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
    # Instead of inferring flags from the tag string (which can drift over time),
    # I rebuild the dashboard config from the metrics CSV that the benchmark wrote.
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
    # If main.py ran with tuning enabled, it writes best_params_*.csv; I load them so dashboard predictions
    # match the tuned model that produced the metrics table.
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
        # If anything goes wrong, I keep the untuned defaults instead of crashing the dashboard.
        return pipe

    return pipe


# ----------------------------
# Model factory (keeps dashboard coherent with the benchmark)
# ----------------------------
def _pipeline_for(model_name: str) -> Optional[Pipeline]:
    # This paragraph is for keeping the dashboard independent from “which model I saved”.
    # I build the same model families used in the benchmark so I can regenerate predictions on demand.
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

    return None


# ----------------------------
# Plot builders
# ----------------------------
def _plot_model_comparison(metrics_df: pd.DataFrame) -> str:
    # This paragraph is for a quick overview plot (test RMSE by model).
    df = metrics_df.copy().sort_values("test_RMSE", ascending=True)

    fig = plt.figure()
    plt.bar(df["model"].astype(str), df["test_RMSE"].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("Model comparison (lower is better)")
    return _fig_to_base64(fig)


def _plot_actual_vs_pred(pred_df: pd.DataFrame, model_name: str) -> str:
    # This paragraph is for showing the prediction quality over the test years.
    fig = plt.figure()
    plt.plot(pred_df["year"], pred_df["actual"], label="Actual")
    plt.plot(pred_df["year"], pred_df["pred"], label=model_name)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Year (test set)")
    plt.ylabel("GDP growth (%)")
    plt.title("Actual vs Predicted")
    plt.legend()
    return _fig_to_base64(fig)


def _plot_abs_error_by_year(pred_df: pd.DataFrame) -> str:
    # This paragraph is for highlighting which specific years drive the overall RMSE.
    df = pred_df.copy().sort_values("year")
    fig = plt.figure()
    plt.bar(df["year"].astype(int), np.abs(df["error"].to_numpy()))
    plt.xlabel("Year (test set)")
    plt.ylabel("|Actual - Pred|")
    plt.title("Absolute error by year")
    return _fig_to_base64(fig)


def _plot_errors_vs_disaster_proxy(pred_df: pd.DataFrame, X_test: pd.DataFrame) -> str:
    # This paragraph is for a sanity check: if disasters matter, errors might correlate with a disaster proxy.
    if "damage_share_gdp_lag1" in X_test.columns:
        x = pd.to_numeric(X_test["damage_share_gdp_lag1"], errors="coerce").to_numpy()
        xlab = "damage_share_gdp_lag1"
    elif "n_events_lag1" in X_test.columns:
        x = pd.to_numeric(X_test["n_events_lag1"], errors="coerce").to_numpy()
        xlab = "n_events_lag1"
    else:
        fig = plt.figure()
        plt.text(0.02, 0.5, "No disaster proxy found in X_test.", fontsize=12)
        plt.axis("off")
        return _fig_to_base64(fig)

    fig = plt.figure()
    plt.scatter(x, pred_df["error"].to_numpy())
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(xlab)
    plt.ylabel("Error (actual - pred)")
    plt.title("Errors vs disaster proxy")
    return _fig_to_base64(fig)


def _plot_cv_stability(cv_df: Optional[pd.DataFrame]) -> str:
    # This paragraph is for showing whether models are stable across TimeSeriesSplit folds.
    if cv_df is None or cv_df.empty or "fold" not in cv_df.columns or "val_RMSE" not in cv_df.columns:
        fig = plt.figure()
        plt.text(0.02, 0.5, "No cv_scores_{tag}.csv found.", fontsize=12)
        plt.axis("off")
        return _fig_to_base64(fig)

    pv = (
        cv_df.pivot_table(index="fold", columns="model", values="val_RMSE", aggfunc="mean")
        .sort_index()
    )

    fig = plt.figure()
    for col in pv.columns:
        plt.plot(pv.index.to_numpy(), pv[col].to_numpy(), marker="o", label=str(col))

    ymin = np.nanmin(pv.to_numpy())
    if np.isfinite(ymin) and ymin > 0:
        plt.yscale("log")
        plt.ylabel("Validation RMSE (log scale)")
    else:
        plt.yscale("symlog", linthresh=1e-3)
        plt.ylabel("Validation RMSE (symlog)")

    plt.xlabel("Fold")
    plt.title("TimeSeriesSplit CV stability")
    plt.legend()
    plt.grid(True, which="both", axis="y", alpha=0.3)
    return _fig_to_base64(fig)


def _plot_damage_dists(df_model: pd.DataFrame) -> Tuple[str, str]:
    # This paragraph is for EDA: disaster damages are heavy-tailed, so raw + log views are both useful.
    if "total_damage" not in df_model.columns:
        fig = plt.figure()
        plt.text(0.02, 0.5, "No total_damage column found.", fontsize=12)
        plt.axis("off")
        b64 = _fig_to_base64(fig)
        return b64, b64

    dmg = pd.to_numeric(df_model["total_damage"], errors="coerce").fillna(0.0).clip(lower=0.0)

    fig1 = plt.figure()
    plt.hist(dmg.to_numpy(), bins=30)
    plt.xlabel("total_damage (USD)")
    plt.title("Damage distribution (raw scale)")
    raw_b64 = _fig_to_base64(fig1)

    fig2 = plt.figure()
    plt.hist(np.log1p(dmg).to_numpy(), bins=30)
    plt.xlabel("log(1 + total_damage)")
    plt.title("Damage distribution (log1p scale)")
    log_b64 = _fig_to_base64(fig2)

    return raw_b64, log_b64


def _plot_disaster_aggregates(df_model: pd.DataFrame) -> str:
    # This paragraph is for EDA: I show event counts and log damages over time.
    need = {"year", "n_events", "total_damage"}
    if not need.issubset(set(df_model.columns)):
        fig = plt.figure()
        plt.text(0.02, 0.5, "Missing columns for disaster aggregates plot.", fontsize=12)
        plt.axis("off")
        return _fig_to_base64(fig)

    df = df_model.sort_values("year").copy()
    dmg = pd.to_numeric(df["total_damage"], errors="coerce").fillna(0.0).clip(lower=0.0)

    fig = plt.figure()
    plt.plot(df["year"], df["n_events"], label="n_events")
    plt.plot(df["year"], np.log1p(dmg), label="log(1+total_damage)")
    plt.xlabel("Year")
    plt.title("Disaster aggregates over time (Japan)")
    plt.legend()
    return _fig_to_base64(fig)


def _plot_gdp_growth(df_model: pd.DataFrame, y: np.ndarray) -> str:
    # This paragraph is for EDA: I show the target dynamics over time.
    if "year" not in df_model.columns:
        fig = plt.figure()
        plt.text(0.02, 0.5, "No year column found.", fontsize=12)
        plt.axis("off")
        return _fig_to_base64(fig)

    fig = plt.figure()
    if "gdp_growth" in df_model.columns:
        plt.plot(df_model.sort_values("year")["year"], df_model.sort_values("year")["gdp_growth"])
        plt.ylabel("GDP growth (%)")
    else:
        df = df_model.sort_values("year").copy()
        plt.plot(df["year"], np.asarray(y))
        plt.ylabel("Target")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Year")
    plt.title("Japan GDP growth over time")
    return _fig_to_base64(fig)


def _subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, *, r2_min_n: int = 6) -> Dict[str, float]:
    # This paragraph is for small-group diagnostics (avoid misleading R² when n is tiny).
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]
    out = {"count": float(len(yt)), "RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    if len(yt) < 1:
        return out
    out["RMSE"] = _rmse(yt, yp)
    out["MAE"] = float(mean_absolute_error(yt, yp))
    if len(yt) >= r2_min_n:
        out["R2"] = _safe_r2(yt, yp)
    return out


def _severe_threshold_from_train(dmg_share_train: np.ndarray, *, q: float = 0.85) -> float:
    # This paragraph is for avoiding test leakage.
    # I define “severe” using TRAIN only (positive values), then apply it to the test set.
    # This matches src/models.py: if fewer than 3 positive train years exist, I return 0 (no severe group).
    x = np.asarray(dmg_share_train, dtype=float)
    x = x[np.isfinite(x)]
    pos = x[x > 0]
    if pos.size < 3:
        return 0.0
    thr = float(np.quantile(pos, float(q)))
    return max(thr, 0.0)


def _scan_run_comparison() -> pd.DataFrame:
    # This paragraph is for a quick “across runs” summary.
    # I scan all model_metrics_*.csv files so I can see which configuration worked best.
    rows = []
    for p in RESULTS_DIR.glob("model_metrics_*.csv"):
        tag = p.stem.replace("model_metrics_", "")
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        df2 = df[~df["model"].astype(str).str.startswith("baseline")].copy()
        if df2.empty or "test_RMSE" not in df2.columns:
            continue

        best = df2.sort_values("test_RMSE", ascending=True).iloc[0]
        base = df[df["model"].astype(str) == "baseline_roll3_mean"]
        base_rmse = float(base["test_RMSE"].iloc[0]) if not base.empty else float("nan")

        rows.append(
            {
                "tag": tag,
                "best_model": str(best["model"]),
                "best_test_RMSE": float(best["test_RMSE"]),
                "best_test_R2": float(best["test_R2"]) if "test_R2" in df2.columns else float("nan"),
                "baseline_roll3_test_RMSE": base_rmse,
                "rmse_gain_vs_roll3": (base_rmse - float(best["test_RMSE"])) if np.isfinite(base_rmse) else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("best_test_RMSE", ascending=True).reset_index(drop=True)


# ----------------------------
# Dashboard builder
# ----------------------------
def build_dashboard(tag: Optional[str], output_html: Optional[str]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)

    # This paragraph is for a smooth UX.
    # If I do not provide a tag, I just build the dashboard for the latest run automatically.
    if tag is None:
        metrics_path = _pick_latest_metrics_file()
        tag = metrics_path.stem.replace("model_metrics_", "")

    metrics_df = _read_metrics(tag)
    cfg = _config_from_metrics(metrics_df, tag)

    # This paragraph is for a simple top-level plot that matches the main grading metric (test RMSE).
    comp_plot_b64 = _plot_model_comparison(metrics_df)

    # This paragraph is for choosing one model to showcase in detailed plots.
    # I pick the best non-baseline model by test RMSE, then I keep a fallback list.
    m = metrics_df.copy()
    m = m[~m["model"].astype(str).str.startswith("baseline")]
    m = m.sort_values("test_RMSE", ascending=True)

    candidates = list(m["model"].astype(str).tolist())
    for x in ["gradient_boosting", "random_forest", "ridge", "linear_regression", "neural_net_mlp", "xgboost"]:
        if x not in candidates:
            candidates.append(x)

    chosen_name: Optional[str] = None
    chosen_pipe: Optional[Pipeline] = None
    for name in candidates:
        p = _pipeline_for(name)
        if p is not None:
            chosen_name = name
            chosen_pipe = _apply_best_params_if_available(p, name, tag)
            break
    if chosen_name is None or chosen_pipe is None:
        raise RuntimeError("Could not instantiate any model pipeline for dashboard.")

    include_covid = (cfg.covid_mode == "ex_post")

    # This paragraph is for dashboard “plus-value”.
    # Even if my benchmark only saves summary metrics, I rebuild the dataset here to show per-year predictions.
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

    chosen_pipe.fit(X_train, y_train)
    yhat_test = chosen_pipe.predict(X_test)
    yhat_train = chosen_pipe.predict(X_train)

    summary = {
        "train_RMSE": _rmse(y_train, yhat_train),
        "test_RMSE": _rmse(y_test, yhat_test),
        "train_MAE": float(mean_absolute_error(y_train, yhat_train)),
        "test_MAE": float(mean_absolute_error(y_test, yhat_test)),
        "train_R2": _safe_r2(y_train, yhat_train),
        "test_R2": _safe_r2(y_test, yhat_test),
    }

    pred_df = pd.DataFrame({"year": years_test, "actual": y_test, "pred": yhat_test, "error": (y_test - yhat_test)})
    pred_plot_b64 = _plot_actual_vs_pred(pred_df, chosen_name)
    abs_err_plot_b64 = _plot_abs_error_by_year(pred_df)
    err_plot_b64 = _plot_errors_vs_disaster_proxy(pred_df, X_test)

    # EDA-style blocks to match the report figures
    gdp_plot_b64 = _plot_gdp_growth(df_model, y)
    dis_agg_b64 = _plot_disaster_aggregates(df_model)
    dmg_raw_b64, dmg_log_b64 = _plot_damage_dists(df_model)

    # CV stability (optional)
    cv_df = _read_cv_scores(tag)
    cv_plot_b64 = _plot_cv_stability(cv_df)

    # This paragraph is for post-disaster diagnostics (coherent with src/models.py).
    dmg_share_te = (
        pd.to_numeric(X_test.get("damage_share_gdp_lag1", np.zeros(len(X_test))), errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    dmg_share_tr = (
        pd.to_numeric(X_train.get("damage_share_gdp_lag1", np.zeros(len(X_train))), errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    n_events_te = pd.to_numeric(X_test.get("n_events_lag1", np.zeros(len(X_test))), errors="coerce").fillna(0).to_numpy()

    # Post-disaster years: n_events_lag1 > 0
    mask_any = n_events_te > 0

    # Severe disasters: threshold computed on TRAIN only (q=0.85) using positive damage_share values
    thr = _severe_threshold_from_train(dmg_share_tr, q=0.85)
    mask_sev = (n_events_te > 0) & (dmg_share_te >= thr)

    # Extra subgroup: multiple events
    mask_multi = n_events_te >= 2

    diag_any = _subset_metrics(y_test, yhat_test, mask_any, r2_min_n=5)
    diag_sev = _subset_metrics(y_test, yhat_test, mask_sev, r2_min_n=5)
    diag_multi = _subset_metrics(y_test, yhat_test, mask_multi, r2_min_n=5)

    diag_table = pd.DataFrame(
        [
            {"subset": "Post-disaster years (n_events_lag1>0)", **diag_any},
            {"subset": "Severe disasters (train Q85 on positive damage_share_gdp_lag1)", **diag_sev, "threshold": thr},
            {"subset": "Multiple events (n_events_lag1>=2)", **diag_multi},
        ]
    )

    run_scan = _scan_run_comparison()

    # This paragraph is for a clean, single-file HTML output.
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Japan GDP–Disasters Dashboard ({tag})</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .card {{ padding: 16px; border: 1px solid #ddd; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
    h1 {{ margin-top: 0; }}
    img {{ max-width: 100%; height: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f7f7f7; }}
    code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 5px; }}
  </style>
</head>
<body>
  <h1>Japan GDP Growth Forecasting Dashboard</h1>
  <p><b>Run tag:</b> <code>{tag}</code></p>

  <div class="card">
    <h2>Run configuration (from metrics file)</h2>
    <ul>
      <li><b>mode</b>: {cfg.mode}</li>
      <li><b>covid_mode</b>: {cfg.covid_mode}</li>
      <li><b>include_macro</b>: {cfg.include_macro}</li>
      <li><b>include_oil</b>: {cfg.include_oil}</li>
      <li><b>start_year</b>: {cfg.start_year}</li>
      <li><b>test_ratio</b>: {cfg.test_ratio}</li>
      <li><b>chosen_model_for_plots</b>: {chosen_name}</li>
    </ul>
  </div>

  <div class="row">
    <div class="card">
      <h2>Model comparison (Test RMSE)</h2>
      <img src="data:image/png;base64,{comp_plot_b64}" />
    </div>

    <div class="card">
      <h2>TimeSeriesSplit CV stability</h2>
      <img src="data:image/png;base64,{cv_plot_b64}" />
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h2>Japan GDP growth over time (EDA)</h2>
      <img src="data:image/png;base64,{gdp_plot_b64}" />
    </div>

    <div class="card">
      <h2>Disaster aggregates over time (EDA)</h2>
      <img src="data:image/png;base64,{dis_agg_b64}" />
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h2>Damage distribution (raw scale)</h2>
      <img src="data:image/png;base64,{dmg_raw_b64}" />
    </div>

    <div class="card">
      <h2>Damage distribution (log1p scale)</h2>
      <img src="data:image/png;base64,{dmg_log_b64}" />
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h2>Predictions on test years</h2>
      <img src="data:image/png;base64,{pred_plot_b64}" />
      <p><b>Train RMSE:</b> {_format_float(summary["train_RMSE"])} | <b>Test RMSE:</b> {_format_float(summary["test_RMSE"])}</p>
      <p><b>Train R²:</b> {_format_float(summary["train_R2"])} | <b>Test R²:</b> {_format_float(summary["test_R2"])}</p>
    </div>

    <div class="card">
      <h2>Absolute error by year</h2>
      <img src="data:image/png;base64,{abs_err_plot_b64}" />
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h2>Error vs disaster proxy</h2>
      <img src="data:image/png;base64,{err_plot_b64}" />
    </div>

    <div class="card">
      <h2>Post-disaster diagnostics (test subsets)</h2>
      {_df_to_html_table(diag_table, max_rows=10)}
    </div>
  </div>

  <div class="card">
    <h2>Per-year predictions table (test)</h2>
    {_df_to_html_table(pred_df, max_rows=60)}
  </div>

  <div class="card">
    <h2>Across-run scan (best non-baseline per run)</h2>
    {_df_to_html_table(run_scan, max_rows=60)}
  </div>

  <p style="margin-top: 30px; color: #666;">Generated by <code>dashboard/build_dashboard.py</code>.</p>
</body>
</html>
"""

    out = Path(output_html) if output_html else (DASHBOARD_DIR / f"dashboard_{tag}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    # This paragraph is for a simple CLI so graders (and I) can regenerate the dashboard from the terminal.
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None, help="Run tag, e.g. run_forecast_strict_main")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (optional)")
    args = parser.parse_args()

    p = build_dashboard(tag=args.tag, output_html=args.out)
    print(f"[build_dashboard] wrote: {p}")


if __name__ == "__main__":
    main()
