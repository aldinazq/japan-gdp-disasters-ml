# Project Proposal: Forecasting Japan’s GDP Growth with Disaster Signals and Optional Macro Controls

## Project Category
Machine Learning / Time-Series Forecasting (Predictive Analytics)

## Problem Statement & Motivation
Japan is frequently hit by natural disasters (earthquakes, typhoons, floods) and has high-quality macroeconomic data. The objective of this project is **predictive (not causal)**: I build a reproducible machine-learning pipeline to **forecast annual GDP growth one year ahead** using **lagged disaster information** and optional macro/oil controls. The pipeline compares multiple models against strong baselines, uses time-series validation (no shuffling), and produces reproducible outputs plus an optional dashboard for interpretation.

## Prediction Task
**Target:** \( y_t \) = Japan’s GDP growth in year \( t \) (annual %).  
**Horizon:** one-year ahead.

**Strict ex-ante features (headline setting):** to predict \( y_t \), I use only information observable at \( t-1 \):
- GDP dynamics: \( y_{t-1}, y_{t-2} \) and rolling statistics computed only from past values.
- Disaster aggregates (lagged): number of events, deaths, \(\log(1+\text{damage})\), damage share of GDP, and average magnitude at \( t-1 \).
- Optional macro controls (lagged, WDI): inflation, unemployment, exports/GDP, investment/GDP, FX (JPY/USD).
- Optional oil controls (lagged): oil price level and oil price change.

**COVID clarity:** two modes are implemented:
- **Strict forecast:** excludes `covid_dummy` (not observable ex-ante at \( t-1 \) to forecast 2020).
- **Ex-post / explain:** includes `covid_dummy` for robustness/interpretation, clearly labeled as non-strict forecasting.

## Data Sources & Feature Engineering
- **World Bank (WDI):** GDP level and GDP growth + macro indicators.
- **EM-DAT (CRED):** event-level disasters aggregated to yearly features (`n_events`, `total_deaths`, `total_damage`, `avg_magnitude`).When damages are provided in thousands (e.g., “'000 US$”), they are converted to USD (×1000) for unit consistency before constructing \(\log(1+\text{damage})\) and damage/GDP proxies.
- **Oil price series:** annual oil price used as a global shock/control.
Feature engineering focuses on stability (e.g., lagging, rolling statistics, and scaling/transforming damage proxies) while respecting the strict timing constraint.

## Models & Evaluation
**Baselines:** training mean, last-year growth \( y_{t-1} \), and rolling-mean baseline (when available).  
**Models:** Linear Regression (baseline), Ridge regression, Random Forest, Histogram Gradient Boosting (scikit-learn HistGradientBoostingRegressor), and an MLP benchmark (scikit-learn MLPRegressor).
**Validation:** chronological train/test split + TimeSeriesSplit CV on the training set.  
**Metrics:** RMSE, MAE, and \( R^2 \).

## Added Value: Post-disaster Performance
Beyond average test performance, I evaluate performance during:
- **post-disaster years:** years where \( n\_events_{t-1} > 0 \)
- **severe disasters:** defined using a **train-only quantile threshold \( q = 0.85 \)** on `damage_share_gdp_lag1` (no test peeking). When enough positive values exist, the threshold is computed on **positive** train values.

## Deliverables & Success Criteria
Deliverables include a reproducible CLI pipeline (`python3 main.py`) producing tagged CSV outputs, an optional dashboard to visualize results, and unit tests validating data integrity, time-based splitting, and pipeline outputs. Success is defined by clean reproducibility, correct time-series methodology, and measurable gains over strong baselines where achievable.
