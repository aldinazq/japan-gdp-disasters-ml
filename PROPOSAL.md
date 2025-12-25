# Project Proposal: Forecasting Japan’s GDP Growth with Disaster and Macro Signals

## Project Category
Machine Learning / Time-Series Forecasting (Predictive Analytics)

## Problem Statement & Motivation
Japan is frequently hit by natural disasters (earthquakes, typhoons, floods) and also has high-quality macroeconomic data. The aim of this project is predictive: I build a reproducible machine learning pipeline to **forecast annual GDP growth one year ahead** using **lagged disaster information** and optional macro/oil controls. The project specifies a clear prediction task, compares multiple ML models against strong baselines, uses time-series validation (no shuffle), and produces reproducible outputs and a dashboard for interpretation.

## Prediction Task
**Target:** \( y_t = \) Japan’s GDP growth in year \( t \) (annual %).  
**Forecast horizon:** one year ahead.  
**Strict ex-ante features:** to predict \( y_t \), I use only information observable at \( t-1 \):
- GDP dynamics: \( y_{t-1}, y_{t-2} \) and rolling statistics based only on past values.
- Disaster aggregates (lagged): number of events, deaths, total damage, and average magnitude at \( t-1 \).
- Optional macro controls (lagged, WDI): inflation, unemployment, exports/GDP, investment/GDP, FX (JPY/USD).
- Optional oil controls (lagged): oil price level and oil price change.

**COVID clarity:** I implement two modes:
- **Strict forecast:** excludes `covid_dummy` (not observable ex-ante at \( t-1 \) to forecast 2020).
- **Ex-post / explain:** includes `covid_dummy` only for robustness/interpretation, clearly labeled as non-strict forecasting.

## Data Sources & Feature Engineering
- **World Bank (WDI):** GDP level and GDP growth + macro indicators.
- **EM-DAT:** event-level disasters aggregated to yearly features (`n_events`, `total_deaths`, `total_damage`, `avg_magnitude`). Damage values in “’000 US$” are converted to USD for unit consistency.
- **Oil price series:** annual oil price used as a global shock/control.

## Models & Evaluation
**Baselines:** training mean, last-year growth \( y_{t-1} \), and rolling-mean baseline (when available).  
**Models:** Linear Regression, Ridge, Random Forest, Gradient Boosting (HistGradientBoostingRegressor), Neural Network (MLPRegressor), and optional XGBoost if installed.  
**Validation:** time-based train/test split + TimeSeriesSplit CV on the training set. Metrics: RMSE, MAE, and \(R^2\).

## Added Value: Post-disaster Performance
Beyond average test performance, I evaluate models specifically on **post-disaster years** where \( n\_events_{t-1} > 0 \), and on a “severe” subset defined by a train-only damage-share threshold (to avoid leakage). This shows whether ML performs differently during shock periods.

## Deliverables & Success Criteria
Deliverables include a reproducible CLI pipeline (`python main.py`) producing CSV outputs, a dashboard to visualize results, and unit tests validating data integrity, time-series splitting, and pipeline outputs. Success is defined by clean reproducibility, correct time-series methodology, and measurable gains over strong baselines.
