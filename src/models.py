"""
I run a one-year-ahead GDP growth forecasting benchmark with time-series validation, multiple ML models,
and a professor-friendly post-disaster evaluation, while reporting non-finite prediction fallbacks explicitly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import build_master_table


warnings.filterwarnings(
    "ignore",
    message=r"Skipping features without any observed values:.*",
    category=UserWarning,
    module=r"sklearn\.impute\._base",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _imputer() -> SimpleImputer:
    # keep_empty_features=True avoids dropping all-NaN columns inside early CV folds
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def _pipeline_linear() -> Pipeline:
    return Pipeline(
        [
            ("imputer", _imputer()),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def _pipeline_ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("imputer", _imputer()),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _pipeline_rf(**rf_kwargs) -> Pipeline:
    return Pipeline(
        [
            ("imputer", _imputer()),
            ("model", RandomForestRegressor(random_state=42, **rf_kwargs)),
        ]
    )


def _pipeline_hgb(**hgb_kwargs) -> Pipeline:
    """
    I build a boosting model and force internal early-stopping off, because TimeSeriesSplit already provides validation.
    """
    hgb_kwargs = dict(hgb_kwargs)
    hgb_kwargs.pop("validation_fraction", None)
    hgb_kwargs.pop("n_iter_no_change", None)
    hgb_kwargs["early_stopping"] = False

    # Safe defaults for small samples
    hgb_kwargs.setdefault("max_depth", 2)
    hgb_kwargs.setdefault("learning_rate", 0.03)
    hgb_kwargs.setdefault("max_iter", 1200)
    hgb_kwargs.setdefault("min_samples_leaf", 20)
    hgb_kwargs.setdefault("l2_regularization", 0.1)

    return Pipeline(
        [
            ("imputer", _imputer()),
            ("model", HistGradientBoostingRegressor(random_state=42, **hgb_kwargs)),
        ]
    )


def _pipeline_mlp(**mlp_kwargs) -> Pipeline:
    """
    I build an MLP pipeline and force early_stopping off to avoid internal validation splits inside CV folds.
    """
    mlp_kwargs = dict(mlp_kwargs)
    mlp_kwargs.pop("early_stopping", None)
    mlp_kwargs.pop("validation_fraction", None)
    mlp_kwargs["early_stopping"] = False

    mlp_kwargs.setdefault("hidden_layer_sizes", (32, 16))
    mlp_kwargs.setdefault("alpha", 1e-3)
    mlp_kwargs.setdefault("max_iter", 5000)

    return Pipeline(
        [
            ("imputer", _imputer()),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(random_state=42, **mlp_kwargs)),
        ]
    )


def _try_pipeline_xgb(**xgb_kwargs) -> Optional[Pipeline]:
    """
    I try to build an XGBoost model if xgboost is installed; otherwise return None.
    """
    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception:
        return None

    model = XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        **xgb_kwargs,
    )
    return Pipeline([("imputer", _imputer()), ("model", model)])


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("year").reset_index(drop=True)

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["year2"] = out["year"] ** 2

    # EX-POST only (for explain mode)
    out["covid_dummy"] = (out["year"] >= 2020).astype(int)

    # GDP growth lags
    out["gdp_growth_lag1"] = out["gdp_growth"].shift(1)
    out["gdp_growth_lag2"] = out["gdp_growth"].shift(2)

    g_shift = out["gdp_growth"].shift(1)
    out["gdp_growth_roll3_mean"] = g_shift.rolling(3).mean()
    out["gdp_growth_roll5_mean"] = g_shift.rolling(5).mean()
    out["gdp_growth_roll5_std"] = g_shift.rolling(5).std()

    # Disaster features (current year)
    out["log_total_damage"] = np.log1p(pd.to_numeric(out["total_damage"], errors="coerce").fillna(0))
    out["damage_share_gdp"] = np.where(
        pd.to_numeric(out["gdp"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(out["total_damage"], errors="coerce").fillna(0) / pd.to_numeric(out["gdp"], errors="coerce"),
        0.0,
    )
    out["has_disaster"] = (pd.to_numeric(out["n_events"], errors="coerce").fillna(0) > 0).astype(int)

    # Oil change
    if "oil_price_usd" in out.columns:
        out["oil_price_usd"] = pd.to_numeric(out["oil_price_usd"], errors="coerce")
        out["oil_price_usd_change"] = out["oil_price_usd"].diff()

    # Lag everything (strict forecast uses t-1 only)
    to_lag = [
        # disasters
        "n_events",
        "total_deaths",
        "log_total_damage",
        "damage_share_gdp",
        "avg_magnitude",
        # macro
        "inflation_cpi",
        "exports_pct_gdp",
        "unemployment_rate",
        "investment_pct_gdp",
        "fx_jpy_per_usd",
        # oil
        "oil_price_usd",
        "oil_price_usd_change",
    ]
    for c in to_lag:
        if c in out.columns:
            out[f"{c}_lag1"] = pd.to_numeric(out[c], errors="coerce").shift(1)

    return out


def make_dataset(
    *,
    include_oil: bool = False,
    include_macro: bool = False,
    include_covid: bool = False,
    start_year: Optional[int] = None,
    mode: str = "forecast",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    I return (df_model, X, y) for the prediction task:
    predict GDP growth in year t using features observable at t-1 (strict forecast) or optionally nowcast.
    """
    if mode not in {"forecast", "nowcast"}:
        raise ValueError("mode must be 'forecast' or 'nowcast'.")

    df = _add_features(build_master_table())

    base_features: List[str] = [
        "year",
        "year2",
        "gdp_growth_lag1",
        "gdp_growth_lag2",
        "gdp_growth_roll3_mean",
        "gdp_growth_roll5_mean",
        "gdp_growth_roll5_std",
        # disasters (lagged)
        "n_events_lag1",
        "total_deaths_lag1",
        "log_total_damage_lag1",
        "damage_share_gdp_lag1",
        "avg_magnitude_lag1",
    ]

    # EX-POST only (if requested)
    if include_covid and "covid_dummy" in df.columns:
        base_features.insert(2, "covid_dummy")

    macro_features: List[str] = [
        "inflation_cpi_lag1",
        "exports_pct_gdp_lag1",
        "unemployment_rate_lag1",
        "investment_pct_gdp_lag1",
        "fx_jpy_per_usd_lag1",
    ]

    oil_features: List[str] = [
        "oil_price_usd_lag1",
        "oil_price_usd_change_lag1",
    ]

    feature_cols = base_features.copy()
    if include_macro:
        feature_cols.extend(macro_features)
    if include_oil:
        for c in oil_features:
            if c in df.columns and c not in feature_cols:
                feature_cols.append(c)

    # nowcast (current-year info) option
    if mode == "nowcast":
        nowcast_cols = [
            "n_events",
            "total_deaths",
            "log_total_damage",
            "damage_share_gdp",
            "avg_magnitude",
            "has_disaster",
        ]
        if include_oil:
            for c in ["oil_price_usd", "oil_price_usd_change"]:
                if c in df.columns and c not in nowcast_cols:
                    nowcast_cols.append(c)
        feature_cols.extend([c for c in nowcast_cols if c in df.columns])

    keep = [c for c in feature_cols if c in df.columns]

    df_model = df.dropna(subset=["gdp_growth", "gdp_growth_lag1"]).copy()
    if start_year is not None:
        df_model = df_model[df_model["year"] >= int(start_year)].copy()

    df_model = df_model.reset_index(drop=True)
    X = df_model[keep].copy().reset_index(drop=True)
    y = df_model["gdp_growth"].astype(float).to_numpy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return df_model, X, y


def time_train_test_split(
    df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    n = len(df)
    split = int(np.floor(n * (1 - test_ratio)))
    return X.iloc[:split].copy(), X.iloc[split:].copy(), y[:split].copy(), y[split:].copy()


def _subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, *, r2_min_n: int = 5) -> Dict[str, float]:
    """
    I compute subgroup metrics, but I only report R2 if subgroup size >= r2_min_n (otherwise R2 is too unstable).
    """
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count < 2:
        return {"count": float(count), "RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}

    yt = y_true[mask]
    yp = y_pred[mask]
    out = {
        "count": float(count),
        "RMSE": _rmse(yt, yp),
        "MAE": float(mean_absolute_error(yt, yp)),
        "R2": float("nan"),
    }
    if count >= r2_min_n:
        out["R2"] = float(r2_score(yt, yp))
    return out


def tune_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    tag: str,
    n_splits: int = 5,
) -> Pipeline:
    """
    I tune HistGradientBoostingRegressor using TimeSeriesSplit and return the best estimator.
    """
    n_train = len(X_train)
    n_splits_eff = min(n_splits, max(2, n_train // 6))
    tscv = TimeSeriesSplit(n_splits=n_splits_eff)

    base = _pipeline_hgb()

    param_grid = {
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_iter": [400, 800, 1200, 1600],
        "model__min_samples_leaf": [5, 10, 20],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }

    grid = GridSearchCV(
        base,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    pd.DataFrame([grid.best_params_]).to_csv(RESULTS_DIR / f"best_params_gradient_boosting_{tag}.csv", index=False)
    return grid.best_estimator_


def time_series_cv_report(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    tag: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    I run TimeSeriesSplit CV and save fold metrics to results/cv_scores_<tag>.csv.
    """
    n_train = len(X_train)
    n_splits_eff = min(n_splits, max(2, n_train // 6))
    tscv = TimeSeriesSplit(n_splits=n_splits_eff)

    models: Dict[str, Pipeline] = {
        "linear_regression": _pipeline_linear(),
        "ridge": _pipeline_ridge(alpha=5.0),
        "random_forest": _pipeline_rf(n_estimators=600, max_depth=5, min_samples_leaf=2),
        "gradient_boosting": _pipeline_hgb(),
        "neural_net_mlp": _pipeline_mlp(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=5000),
    }

    xgb = _try_pipeline_xgb(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    if xgb is not None:
        models["xgboost"] = xgb

    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        for name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                pred_tr = model.predict(X_tr)
                pred_va = model.predict(X_va)
                fold_rows.append(
                    {
                        "fold": fold,
                        "model": name,
                        "train_RMSE": _rmse(y_tr, pred_tr),
                        "val_RMSE": _rmse(y_va, pred_va),
                        "train_R2": float(r2_score(y_tr, pred_tr)),
                        "val_R2": float(r2_score(y_va, pred_va)),
                        "status": "ok",
                    }
                )
            except Exception as e:
                fold_rows.append(
                    {
                        "fold": fold,
                        "model": name,
                        "train_RMSE": float("nan"),
                        "val_RMSE": float("nan"),
                        "train_R2": float("nan"),
                        "val_R2": float("nan"),
                        "status": f"fail:{type(e).__name__}",
                    }
                )

    cv_df = pd.DataFrame(fold_rows)
    cv_df.to_csv(RESULTS_DIR / f"cv_scores_{tag}.csv", index=False)
    return cv_df


def _compute_severe_threshold_from_train(
    dmg_share_train: np.ndarray,
    *,
    severe_q: float = 0.75,
) -> Tuple[float, int]:
    """
    I compute the severe threshold from TRAIN only, using the quantile among strictly positive damage shares.
    Returns (threshold, n_positive_train).
    """
    dmg_share_train = np.asarray(dmg_share_train, dtype=float)
    dmg_share_train = dmg_share_train[np.isfinite(dmg_share_train)]
    pos = dmg_share_train[dmg_share_train > 0]
    n_pos = int(pos.size)

    if n_pos >= 5:
        thr = float(np.quantile(pos, severe_q))
    else:
        thr = float(np.quantile(dmg_share_train, severe_q)) if dmg_share_train.size else 0.0

    thr = max(thr, 0.0)
    return thr, n_pos


def run_all_models(
    *,
    include_oil: bool = False,
    include_macro: bool = False,
    include_covid: bool = False,
    start_year: Optional[int] = None,
    test_ratio: float = 0.2,
    tune_rf: bool = False,
    tune_gb: bool = False,
    mode: str = "forecast",
    tag: str = "run",
) -> pd.DataFrame:
    """
    I run the full benchmark, save metrics to results/model_metrics_<tag>.csv,
    and include subgroup evaluation for post-disaster years.
    """
    df, X, y = make_dataset(
        include_oil=include_oil,
        include_macro=include_macro,
        include_covid=include_covid,
        start_year=start_year,
        mode=mode,
    )

    X_train, X_test, y_train, y_test = time_train_test_split(df, X, y, test_ratio=test_ratio)
    _ = time_series_cv_report(X_train, y_train, tag=tag)

    rows: List[Dict[str, float]] = []
    mean_train = float(np.mean(y_train))

    def _sanitize_pred(arr: np.ndarray, fallback: float) -> np.ndarray:
        a = np.asarray(arr, dtype=float).copy()
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = fallback
        return a

    # -------------------------
    # Post-disaster definitions
    # -------------------------
    n_events_te = pd.to_numeric(
        X_test.get("n_events_lag1", pd.Series([0] * len(X_test))), errors="coerce"
    ).fillna(0).to_numpy()

    dmg_share_te = pd.to_numeric(
        X_test.get("damage_share_gdp_lag1", pd.Series([0.0] * len(X_test))), errors="coerce"
    ).fillna(0.0).to_numpy()

    dmg_share_tr = pd.to_numeric(
        X_train.get("damage_share_gdp_lag1", pd.Series([0.0] * len(X_train))), errors="coerce"
    ).fillna(0.0).to_numpy()

    mask_any = n_events_te > 0

    severe_q = 0.75
    severe_thr, train_pos_damage_count = _compute_severe_threshold_from_train(dmg_share_tr, severe_q=severe_q)
    mask_severe = (n_events_te > 0) & (dmg_share_te >= severe_thr)

    mask_multi_event = n_events_te >= 2

    def add_row(model_name: str, yhat_tr: np.ndarray, yhat_te: np.ndarray) -> None:
        # ✅ NEW: explicitly report how many predictions were non-finite before fallback
        raw_tr = np.asarray(yhat_tr, dtype=float)
        raw_te = np.asarray(yhat_te, dtype=float)
        n_bad_tr = int((~np.isfinite(raw_tr)).sum())
        n_bad_te = int((~np.isfinite(raw_te)).sum())

        yhat_tr_s = _sanitize_pred(raw_tr, mean_train)
        yhat_te_s = _sanitize_pred(raw_te, mean_train)

        m_any = _subset_metrics(y_test, yhat_te_s, mask_any, r2_min_n=5)
        m_sev = _subset_metrics(y_test, yhat_te_s, mask_severe, r2_min_n=5)
        m_multi = _subset_metrics(y_test, yhat_te_s, mask_multi_event, r2_min_n=5)

        rows.append(
            {
                "model": model_name,
                "train_MAE": float(mean_absolute_error(y_train, yhat_tr_s)),
                "train_RMSE": _rmse(y_train, yhat_tr_s),
                "train_R2": float(r2_score(y_train, yhat_tr_s)),
                "test_MAE": float(mean_absolute_error(y_test, yhat_te_s)),
                "test_RMSE": _rmse(y_test, yhat_te_s),
                "test_R2": float(r2_score(y_test, yhat_te_s)),
                # ✅ NEW: diagnostic columns
                "n_nonfinite_pred_train": n_bad_tr,
                "n_nonfinite_pred_test": n_bad_te,
                # any-disaster
                "test_count_any_disaster": int(m_any["count"]),
                "test_RMSE_any_disaster": float(m_any["RMSE"]),
                "test_MAE_any_disaster": float(m_any["MAE"]),
                "test_R2_any_disaster": float(m_any["R2"]),
                # severe (q75 among positive damage shares, train-only threshold)
                "test_count_severe_disaster": int(m_sev["count"]),
                "test_RMSE_severe_disaster": float(m_sev["RMSE"]),
                "test_MAE_severe_disaster": float(m_sev["MAE"]),
                "test_R2_severe_disaster": float(m_sev["R2"]),  # NaN if count<5
                "severe_quantile_used": severe_q,
                "severe_threshold_damage_share_train": severe_thr,
                "train_count_positive_damage_share": train_pos_damage_count,
                "test_count_positive_damage_share": int(np.sum(dmg_share_te > 0)),
                # multi-event (optional subgroup)
                "test_count_multi_event": int(m_multi["count"]),
                "test_RMSE_multi_event": float(m_multi["RMSE"]),
                "test_MAE_multi_event": float(m_multi["MAE"]),
                "test_R2_multi_event": float(m_multi["R2"]),  # NaN if count<5
            }
        )

    # -------------------------
    # Baselines
    # -------------------------
    add_row(
        "baseline_mean_train",
        np.full(shape=y_train.shape, fill_value=mean_train, dtype=float),
        np.full(shape=y_test.shape, fill_value=mean_train, dtype=float),
    )

    add_row(
        "baseline_last_year",
        X_train["gdp_growth_lag1"].to_numpy(),
        X_test["gdp_growth_lag1"].to_numpy(),
    )

    if "gdp_growth_roll3_mean" in X_train.columns:
        tr = X_train["gdp_growth_roll3_mean"].fillna(X_train["gdp_growth_lag1"]).to_numpy()
        te = X_test["gdp_growth_roll3_mean"].fillna(X_test["gdp_growth_lag1"]).to_numpy()
        add_row("baseline_roll3_mean", tr, te)

    # -------------------------
    # Linear / Ridge
    # -------------------------
    lr = _pipeline_linear()
    lr.fit(X_train, y_train)
    add_row("linear_regression", lr.predict(X_train), lr.predict(X_test))

    ridge = _pipeline_ridge(alpha=5.0)
    ridge.fit(X_train, y_train)
    add_row("ridge", ridge.predict(X_train), ridge.predict(X_test))

    # -------------------------
    # Random Forest (optional tuning)
    # -------------------------
    if tune_rf:
        rf_base = _pipeline_rf()
        rf_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }
        n_train = len(X_train)
        n_splits_eff = min(5, max(2, n_train // 6))
        tscv = TimeSeriesSplit(n_splits=n_splits_eff)
        grid = GridSearchCV(
            rf_base,
            param_grid=rf_grid,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        rf = grid.best_estimator_
        pd.DataFrame([grid.best_params_]).to_csv(RESULTS_DIR / f"best_params_random_forest_{tag}.csv", index=False)
    else:
        rf = _pipeline_rf(n_estimators=600, max_depth=5, min_samples_leaf=2)

    rf.fit(X_train, y_train)
    add_row("random_forest", rf.predict(X_train), rf.predict(X_test))

    # -------------------------
    # Gradient Boosting (optional tuning)
    # -------------------------
    if tune_gb:
        gb = tune_gradient_boosting(X_train, y_train, tag=tag)
    else:
        gb = _pipeline_hgb()

    gb.fit(X_train, y_train)
    add_row("gradient_boosting", gb.predict(X_train), gb.predict(X_test))

    # -------------------------
    # Optional XGBoost
    # -------------------------
    xgb = _try_pipeline_xgb(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    if xgb is not None:
        xgb.fit(X_train, y_train)
        add_row("xgboost", xgb.predict(X_train), xgb.predict(X_test))

    # -------------------------
    # Neural Net (kept as benchmark; often overfits)
    # -------------------------
    mlp = _pipeline_mlp(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=5000)
    mlp.fit(X_train, y_train)
    add_row("neural_net_mlp", mlp.predict(X_train), mlp.predict(X_test))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / f"model_metrics_{tag}.csv", index=False)
    return results_df
