"""I train baseline and machine learning models to predict Japan's GDP growth from my enriched yearly dataset."""

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import build_master_table


def make_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = build_master_table().sort_values("year").reset_index(drop=True)

    df["gdp_growth_lag1"] = df["gdp_growth"].shift(1)
    df["n_events_lag1"] = df["n_events"].shift(1)
    df["total_damage_lag1"] = df["total_damage"].shift(1)
    df["log_total_damage_lag1"] = df["log_total_damage"].shift(1)
    df["damage_share_gdp_lag1"] = df["damage_share_gdp"].shift(1)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "gdp_growth_lag1",
        "n_events",
        "n_events_lag1",
        "total_damage_lag1",
        "log_total_damage_lag1",
        "damage_share_gdp_lag1",
        "has_disaster",
    ]
    X = df[feature_cols]
    y = df["gdp_growth"]
    return df, X, y


def time_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    n_train = int(np.floor((1 - test_ratio) * n))
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train]
    y_test = y.iloc[n_train:]
    return X_train, X_test, y_train, y_test


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def run_all_models() -> pd.DataFrame:
    df, X, y = make_dataset()
    X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_ratio=0.2)

    results: Dict[str, Dict[str, Any]] = {}

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

    results_df = (
        pd.DataFrame(results)
        .T.reset_index()
        .rename(columns={"index": "model"})
    )
    return results_df


if __name__ == "__main__":
    metrics = run_all_models()
    print("=== Model evaluation summary ===")
    print(metrics)
