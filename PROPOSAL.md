# Project Proposal: Natural Disasters and Japan's GDP Growth

## Project Category
Data Analysis & Visualization / Statistical Analysis Tools

## Problem Statement & Motivation
Japan is one of the countries most exposed to natural disasters (earthquakes, typhoons, floods). At the same time, it is a large, advanced economy with detailed macroeconomic statistics. I want to explore how natural disasters are related to Japan’s yearly GDP growth and whether simple machine learning models can extract useful predictive patterns from this data.

The goal is not to make a perfect macroeconomic model, but to build a clean, well-structured Python project that combines multiple real-world datasets (World Bank GDP series and EM-DAT disaster data) and uses them to model GDP growth over time.

## Planned Approach and Technologies
I will first build a yearly dataset for Japan that combines:
- GDP level (current USD) and GDP growth from the World Bank,
- Disaster features aggregated by year from EM-DAT: number of events, total deaths, total economic damages, and average magnitude.

This dataset will be constructed in `src/features.py` using `pandas`. The main target variable will be annual GDP growth. I will compare:
- A simple baseline model (for example “use last year’s growth”),
- Linear regression and regularized regression (Ridge),
- A tree-based model such as `RandomForestRegressor`.

The implementation will follow the course structure: `main.py` as entry point, `src/` for data loading, feature construction, and models, `results/` for plots and metrics, and `requirements.txt` for dependencies. I will use `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.

## Expected Challenges and How I Will Address Them
The dataset is relatively small (one observation per year) and disasters are rare and unevenly distributed. This can lead to unstable models and overfitting. I will:
- Keep the feature set simple and interpretable,
- Use train/test splits that respect the time dimension (train on early years, test on recent years),
- Use cross-validation on the training set where appropriate,
- Compare results to the naive baseline to check if the models actually add value.

Another challenge is interpretation: this is not a causal model. I will clearly state that the analysis is descriptive and predictive, not causal.

## Success Criteria
The project will be considered successful if:
- The repository has a clean structure and `python main.py` runs without errors on a fresh environment.
- I can build a reproducible yearly dataset combining GDP, GDP growth, and disaster features.
- At least one ML model with disaster features outperforms a naive baseline on test data (for example in terms of RMSE/MAE).
- The code is documented and the results are summarized in a short report.

## Stretch Goals (If Time Permits)
If I have time, I would like to build a small interactive interface (for example a simple command-line menu or notebook widget) where the user can:
- choose a year and see the predicted vs actual GDP growth,
- play with hypothetical disaster features (number of events, damages, etc.) and see how the model’s prediction changes.
