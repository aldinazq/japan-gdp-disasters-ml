"""I build a self-contained HTML dashboard that summarizes model performance, out-of-sample predictions, and post-disaster diagnostics for my project."""

from __future__ import annotations

import argparse
import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


# --- Paths + import safety ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

# Reuse your project dataset builder to guarantee feature coherence
from src.models import make_dataset, time_train_test_split  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    tag: str
    mode: str
    covid_mode: str
    include_macro: bool
    include_oil: bool
    start_year: Optional[int]
    test_ratio: float


# ----------------------------
# Helpers
# ----------------------------
def _imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _format_float(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:.3f}"


def _df_to_html_table(df: pd.DataFrame, *, max_rows: int = 40) -> str:
    out = df.copy()
    if len(out) > max_rows:
        out = out.head(max_rows)
    return out.to_html(index=False, escape=True)


def _pick_latest_metrics_file() -> Path:
    files = sorted(RESULTS_DIR.glob("model_metrics_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No results/model_metrics_*.csv found. Run main.py first.")
    return files[0]


def _read_metrics(tag: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"model_metrics_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run main.py to create it.")
    return pd.read_csv(path)


def _infer_config_from_tag(tag: str, *, default_test_ratio: float = 0.2) -> RunConfig:
    t = tag.lower()

    mode = "nowcast" if "nowcast" in t else "forecast"
    covid_mode = "ex_post" if "ex_post" in t else "strict"

    include_macro = ("macro" in t) or ("withmacro" in t)
    include_oil = ("oil" in t)

    # Heuristic consistent with your main conventions
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


def _pipeline_for(model_name: str) -> Optional[Pipeline]:
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
# Dashboard computations
# ----------------------------
def _plot_model_comparison(metrics_df: pd.DataFrame) -> str:
    df = metrics_df.copy().sort_values("test_RMSE", ascending=True)
    fig = plt.figure()
    plt.bar(df["model"].astype(str), df["test_RMSE"].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("Model comparison (lower is better)")
    return _fig_to_base64(fig)


def _plot_actual_vs_pred(pred_df: pd.DataFrame, model_name: str) -> str:
    fig = plt.figure()
    plt.plot(pred_df["year"], pred_df["actual"], label="Actual")
    plt.plot(pred_df["year"], pred_df["pred"], label=model_name)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Year (test set)")
    plt.ylabel("GDP growth (%)")
    plt.title("Out-of-sample: actual vs predicted")
    plt.legend()
    return _fig_to_base64(fig)


def _plot_errors_vs_disaster_proxy(pred_df: pd.DataFrame, X_test: pd.DataFrame) -> Optional[str]:
    if "damage_share_gdp_lag1" not in X_test.columns and "n_events_lag1" not in X_test.columns:
        return None

    if "damage_share_gdp_lag1" in X_test.columns:
        x = pd.to_numeric(X_test["damage_share_gdp_lag1"], errors="coerce").to_numpy()
        xlab = "damage_share_gdp_lag1"
    else:
        x = pd.to_numeric(X_test["n_events_lag1"], errors="coerce").to_numpy()
        xlab = "n_events_lag1"

    fig = plt.figure()
    plt.scatter(x, pred_df["error"].to_numpy())
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(xlab)
    plt.ylabel("Prediction error (actual - pred)")
    plt.title("Errors vs disaster proxy (t-1)")
    return _fig_to_base64(fig)


def _subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n < 2:
        return {"count": float(n), "RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "count": float(n),
        "RMSE": _rmse(yt, yp),
        "MAE": float(mean_absolute_error(yt, yp)),
        "R2": _safe_r2(yt, yp) if n >= 5 else float("nan"),
    }


def _severe_threshold_from_train(dmg_share_train: np.ndarray, q: float = 0.75) -> float:
    dmg_share_train = np.asarray(dmg_share_train, dtype=float)
    dmg_share_train = dmg_share_train[np.isfinite(dmg_share_train)]
    pos = dmg_share_train[dmg_share_train > 0]
    if pos.size >= 5:
        thr = float(np.quantile(pos, q))
    elif dmg_share_train.size:
        thr = float(np.quantile(dmg_share_train, q))
    else:
        thr = 0.0
    return max(thr, 0.0)


def _top_permutation_importance(model: Pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 30) -> pd.DataFrame:
    res = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, scoring="neg_root_mean_squared_error"
    )
    df = pd.DataFrame({"feature": X.columns, "importance": res.importances_mean, "std": res.importances_std})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _plot_top_features(imp_df: pd.DataFrame, top_k: int = 12) -> str:
    top = imp_df.head(top_k).copy()
    fig = plt.figure()
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("Permutation importance (ΔRMSE)")
    plt.title(f"Top {top_k} features (permutation importance)")
    return _fig_to_base64(fig)


def _scan_run_comparison() -> pd.DataFrame:
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


def build_dashboard(tag: Optional[str], output_html: Optional[str]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)

    if tag is None:
        metrics_path = _pick_latest_metrics_file()
        tag = metrics_path.stem.replace("model_metrics_", "")

    cfg = _infer_config_from_tag(tag)
    metrics_df = _read_metrics(tag)

    # --- Plots: model comparison ---
    comp_plot_b64 = _plot_model_comparison(metrics_df)

    # --- Choose a model for plots (best RMSE among non-baselines; fallback to common ones) ---
    m = metrics_df.copy()
    m = m[~m["model"].astype(str).str.startswith("baseline")]
    m = m.sort_values("test_RMSE", ascending=True)

    candidates = list(m["model"].astype(str).tolist())
    for x in ["gradient_boosting", "random_forest", "ridge", "linear_regression"]:
        if x not in candidates:
            candidates.append(x)

    chosen_name = None
    chosen_pipe = None
    for name in candidates:
        p = _pipeline_for(name)
        if p is not None:
            chosen_name = name
            chosen_pipe = p
            break
    if chosen_name is None or chosen_pipe is None:
        raise RuntimeError("Could not instantiate any model pipeline for dashboard.")

    include_covid = (cfg.covid_mode == "ex_post")

    # --- Rebuild dataset to generate per-year predictions (dashboard adds value even if models.py does not save preds) ---
    df_model, X, y = make_dataset(
        include_oil=cfg.include_oil,
        include_macro=cfg.include_macro,
        include_covid=include_covid,
        start_year=cfg.start_year,
        mode=cfg.mode,
    )
    X_train, X_test, y_train, y_test = time_train_test_split(df_model, X, y, test_ratio=cfg.test_ratio)

    split_idx = int(np.floor(len(df_model) * (1 - cfg.test_ratio)))
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

    pred_df = pd.DataFrame(
        {"year": years_test, "actual": y_test, "pred": yhat_test, "error": (y_test - yhat_test)}
    )

    pred_plot_b64 = _plot_actual_vs_pred(pred_df, chosen_name)
    err_plot_b64 = _plot_errors_vs_disaster_proxy(pred_df, X_test)

    # --- Post-disaster diagnostics (real project value) ---
    if "n_events_lag1" in X_test.columns:
        n_events = pd.to_numeric(X_test["n_events_lag1"], errors="coerce").fillna(0).to_numpy()
    else:
        n_events = np.zeros(len(X_test))

    if "damage_share_gdp_lag1" in X_train.columns:
        dmg_tr = pd.to_numeric(X_train["damage_share_gdp_lag1"], errors="coerce").fillna(0).to_numpy()
        dmg_te = pd.to_numeric(X_test["damage_share_gdp_lag1"], errors="coerce").fillna(0).to_numpy()
        thr = _severe_threshold_from_train(dmg_tr, q=0.75)
    else:
        dmg_te = np.zeros(len(X_test))
        thr = 0.0

    mask_any = n_events > 0
    mask_multi = n_events >= 2
    mask_severe = (n_events > 0) & (dmg_te >= thr)

    post_any = _subset_metrics(y_test, yhat_test, mask_any)
    post_multi = _subset_metrics(y_test, yhat_test, mask_multi)
    post_severe = _subset_metrics(y_test, yhat_test, mask_severe)

    # --- Feature importance (permutation) ---
    imp_df = _top_permutation_importance(chosen_pipe, X_test, y_test, n_repeats=30)
    imp_plot_b64 = _plot_top_features(imp_df, top_k=12)
    imp_table_html = _df_to_html_table(imp_df.head(20))

    # --- Compare all saved runs in results/ ---
    runs_df = _scan_run_comparison()
    runs_table = _df_to_html_table(runs_df, max_rows=60) if not runs_df.empty else "<p>No other runs found.</p>"

    # --- Baseline gain for the selected run ---
    base_roll = metrics_df[metrics_df["model"].astype(str) == "baseline_roll3_mean"]
    base_roll_rmse = float(base_roll["test_RMSE"].iloc[0]) if not base_roll.empty else float("nan")
    gain = (base_roll_rmse - summary["test_RMSE"]) if np.isfinite(base_roll_rmse) else float("nan")

    # --- Output file ---
    out_dir = PROJECT_ROOT / "dashboard"
    out_dir.mkdir(exist_ok=True)
    out_html = Path(output_html) if output_html else out_dir / f"dashboard_{tag}.html"

    def img_tag(b64: str, *, alt: str) -> str:
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;height:auto;border:1px solid #eee;border-radius:10px;">'

    post_df = pd.DataFrame(
        [
            {"subset": "any_disaster (n_events_lag1>0)", **post_any},
            {"subset": "severe_disaster (q75 on train)", **post_severe, "severe_threshold": thr},
            {"subset": "multi_event (n_events_lag1>=2)", **post_multi},
        ]
    )

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Japan GDP & Disasters — Dashboard ({tag})</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; color: #111; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ padding: 16px 18px; border: 1px solid #eee; border-radius: 14px; margin: 14px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
    h1 {{ margin: 0 0 6px 0; }}
    h2 {{ margin: 0 0 10px 0; }}
    .muted {{ color: #555; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #eee; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #fafafa; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    @media (min-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr 1fr; }} }}
    code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 7px; }}
    pre {{ background: #f6f6f6; padding: 10px; border-radius: 10px; overflow-x: auto; }}
  </style>
</head>
<body>
<div class="container">
  <h1>Japan GDP Growth Forecasting & Natural Disasters</h1>
  <div class="muted">Generated from <code>results/model_metrics_{tag}.csv</code></div>

  <div class="card">
    <h2>Run configuration</h2>
    <ul>
      <li><b>tag</b>: {cfg.tag}</li>
      <li><b>mode</b>: {cfg.mode}</li>
      <li><b>covid_mode</b>: {cfg.covid_mode}</li>
      <li><b>include_macro</b>: {cfg.include_macro}</li>
      <li><b>include_oil</b>: {cfg.include_oil}</li>
      <li><b>start_year</b>: {cfg.start_year}</li>
      <li><b>test_ratio</b>: {cfg.test_ratio}</li>
    </ul>
  </div>

  <div class="card">
    <h2>Key takeaways</h2>
    <ul>
      <li><b>Chosen model for plots</b>: <code>{chosen_name}</code></li>
      <li><b>Test RMSE</b>: {_format_float(summary["test_RMSE"])} (R²: {_format_float(summary["test_R2"])})</li>
      <li><b>Baseline roll3 RMSE</b>: {_format_float(base_roll_rmse)} → <b>RMSE gain</b>: {_format_float(gain)}</li>
    </ul>
  </div>

  <div class="card">
    <h2>Model comparison</h2>
    {img_tag(comp_plot_b64, alt="Model comparison")}
    <div style="margin-top:10px;">
      {_df_to_html_table(metrics_df.sort_values("test_RMSE", ascending=True))}
    </div>
  </div>

  <div class="card">
    <h2>Out-of-sample predictions (test window)</h2>
    <div class="grid2">
      <div>{img_tag(pred_plot_b64, alt="Actual vs predicted")}</div>
      <div>
        <h3 style="margin-top:0;">Test metrics (recomputed)</h3>
        <ul>
          <li>RMSE: {_format_float(summary["test_RMSE"])}</li>
          <li>MAE: {_format_float(summary["test_MAE"])}</li>
          <li>R²: {_format_float(summary["test_R2"])}</li>
        </ul>
        <h3>Per-year table (test)</h3>
        {_df_to_html_table(pred_df.sort_values("year"))}
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Post-disaster performance (dashboard “plus-value”)</h2>
    <p class="muted">Same test window, but evaluated only on years following disasters (t-1).</p>
    {_df_to_html_table(post_df)}
  </div>

  <div class="card">
    <h2>Error diagnostics</h2>
    <p class="muted">Do forecast errors increase after larger disasters (t-1)?</p>
    {img_tag(err_plot_b64, alt="Errors vs disaster proxy") if err_plot_b64 else "<p>No disaster proxy columns available in this run.</p>"}
  </div>

  <div class="card">
    <h2>Feature importance (permutation)</h2>
    <p class="muted">Increase in RMSE when a feature is permuted (computed on test set).</p>
    {img_tag(imp_plot_b64, alt="Top feature importance")}
    <div style="margin-top:10px;">{imp_table_html}</div>
  </div>

  <div class="card">
    <h2>Comparison across all saved runs</h2>
    <p class="muted">Best test RMSE found in each <code>model_metrics_*.csv</code> in your results folder.</p>
    {runs_table}
  </div>

  <div class="card">
    <h2>How to regenerate</h2>
    <pre><code>./.venv/bin/python dashboard/build_dashboard.py --tag {tag}</code></pre>
  </div>

</div>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    return out_html


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an HTML dashboard from model_metrics_*.csv.")
    p.add_argument("--tag", type=str, default=None, help="Tag used in results/model_metrics_<tag>.csv (default: latest).")
    p.add_argument("--output", type=str, default=None, help="Output html path (default: dashboard/dashboard_<tag>.html).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build_dashboard(args.tag, args.output)
    print(f"Dashboard written to: {out}")


if __name__ == "__main__":
    main()
