"""I train baseline and machine learning models, then save metrics and predictions for my project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import build_master_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_dataset(include_oil: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    I create a time-ordered dataset with:
    - target lags (gdp growth lags),
    - disaster intensity (current year + lag1),
    - macro variables lagged by 1 year (to reduce leakage).
    """
    df = build_master_table().sort_values("year").reset_index(drop=True)

    # --- Target lags (strong baseline signal)
    df["gdp_growth_lag1"] = df["gdp_growth"].shift(1)
    df["gdp_growth_lag2"] = df["gdp_growth"].shift(2)

    # --- Disaster lags (possible delayed effects)
    df["n_events_lag1"] = df["n_events"].shift(1)
    df["total_deaths_lag1"] = df["total_deaths"].shift(1)
    df["log_total_damage_lag1"] = df["log_total_damage"].shift(1)
    df["damage_share_gdp_lag1"] = df["damage_share_gdp"].shift(1)

    # --- Macro lags (use lag1 to avoid using year-t information)
    macro_cols = [
        "inflation_cpi",
        "exports_pct_gdp",
        "unemployment_rate",
        "investment_pct_gdp",
        "fx_jpy_per_usd",
        "oil_price_usd",
    ]
    for c in macro_cols:
        if c in df.columns:
            df[f"{c}_lag1"] = df[c].shift(1)

    feature_cols: List[str] = [
        # autoregressive memory
        "gdp_growth_lag1",
        "gdp_growth_lag2",
        # disasters (current year)
        "n_events",
        "total_deaths",
        "log_total_damage",
        "damage_share_gdp",
        "avg_magnitude",
        "has_disaster",
        # disasters (lagged)
        "n_events_lag1",
        "total_deaths_lag1",
        "log_total_damage_lag1",
        "damage_share_gdp_lag1",
        # macros (lagged)
        "inflation_cpi_lag1",
        "exports_pct_gdp_lag1",
        "unemployment_rate_lag1",
        "investment_pct_gdp_lag1",
        "fx_jpy_per_usd_lag1",
    ]

    # Oil is optional because it can reduce the usable sample (if missing early years)
    if include_oil and "oil_price_usd_lag1" in df.columns:
        feature_cols.append("oil_price_usd_lag1")

    # Drop rows with missing inputs (ONLY the needed columns, not the entire df)
    df = df.dropna(subset=feature_cols + ["gdp_growth"]).reset_index(drop=True)

    X = df[feature_cols]
    y = df["gdp_growth"]
    return df, X, y


def time_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """I split a time series into a train block then a test block (no shuffling)."""
    n = len(X)
    n_train = int(np.floor((1 - test_ratio) * n))
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train]
    y_test = y.iloc[n_train:]
    return X_train, X_test, y_train, y_test


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """I compute MAE, RMSE, and R2 for regression."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


def run_all_models(include_oil: bool = False, test_ratio: float = 0.2) -> pd.DataFrame:
    """I train all models, evaluate them, and save outputs to the results/ folder."""
    df, X, y = make_dataset(include_oil=include_oil)
    X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_ratio=test_ratio)

    results: Dict[str, Dict[str, Any]] = {}

    # --- Baseline: predict last year's growth
    y_pred_baseline_train = X_train["gdp_growth_lag1"].values
    y_pred_baseline_test = X_test["gdp_growth_lag1"].values
    metrics_train_baseline = regression_metrics(y_train, y_pred_baseline_train)
    metrics_test_baseline = regression_metrics(y_test, y_pred_baseline_test)
    results["baseline_last_year"] = {
        "train_MAE": metrics_train_baseline["MAE"],
        "train_RMSE": metrics_train_baseline["RMSE"],
        "train_R2": metrics_train_baseline["R2"],
        "test_MAE": metrics_test_baseline["MAE"],
        "test_RMSE": metrics_test_baseline["RMSE"],
        "test_R2": metrics_test_baseline["R2"],
    }

    # --- Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr_train = lr.predict(X_train)
    y_pred_lr_test = lr.predict(X_test)
    metrics_train_lr = regression_metrics(y_train, y_pred_lr_train)
    metrics_test_lr = regression_metrics(y_test, y_pred_lr_test)
    results["linear_regression"] = {
        "train_MAE": metrics_train_lr["MAE"],
        "train_RMSE": metrics_train_lr["RMSE"],
        "train_R2": metrics_train_lr["R2"],
        "test_MAE": metrics_test_lr["MAE"],
        "test_RMSE": metrics_test_lr["RMSE"],
        "test_R2": metrics_test_lr["R2"],
    }

    # --- Ridge
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge_train = ridge.predict(X_train)
    y_pred_ridge_test = ridge.predict(X_test)
    metrics_train_ridge = regression_metrics(y_train, y_pred_ridge_train)
    metrics_test_ridge = regression_metrics(y_test, y_pred_ridge_test)
    results["ridge"] = {
        "train_MAE": metrics_train_ridge["MAE"],
        "train_RMSE": metrics_train_ridge["RMSE"],
        "train_R2": metrics_train_ridge["R2"],
        "test_MAE": metrics_test_ridge["MAE"],
        "test_RMSE": metrics_test_ridge["RMSE"],
        "test_R2": metrics_test_ridge["R2"],
    }

    # --- Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        random_state=42,
        min_samples_leaf=2,
    )
    rf.fit(X_train, y_train)
    y_pred_rf_train = rf.predict(X_train)
    y_pred_rf_test = rf.predict(X_test)
    metrics_train_rf = regression_metrics(y_train, y_pred_rf_train)
    metrics_test_rf = regression_metrics(y_test, y_pred_rf_test)
    results["random_forest"] = {
        "train_MAE": metrics_train_rf["MAE"],
        "train_RMSE": metrics_train_rf["RMSE"],
        "train_R2": metrics_train_rf["R2"],
        "test_MAE": metrics_test_rf["MAE"],
        "test_RMSE": metrics_test_rf["RMSE"],
        "test_R2": metrics_test_rf["R2"],
    }

    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    results_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    # Save predictions for RF (train + test)
    rf_train_df = pd.DataFrame(
        {
            "year": df.loc[X_train.index, "year"].values,
            "gdp_growth_actual": y_train.values,
            "gdp_growth_pred_rf": y_pred_rf_train,
            "set": "train",
        }
    )
    rf_test_df = pd.DataFrame(
        {
            "year": df.loc[X_test.index, "year"].values,
            "gdp_growth_actual": y_test.values,
            "gdp_growth_pred_rf": y_pred_rf_test,
            "set": "test",
        }
    )
    rf_pred_df = pd.concat([rf_train_df, rf_test_df], ignore_index=True)
    rf_pred_df.to_csv(RESULTS_DIR / "random_forest_predictions.csv", index=False)

    return results_df


if __name__ == "__main__":
    metrics = run_all_models(include_oil=False, test_ratio=0.2)
    print("=== Model evaluation summary ===")
    print(metrics.to_string(index=False))
