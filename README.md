## Feature engineering and modeling choices

The main target of the project is annual GDP growth for Japan. I started from a simple specification using only last year’s GDP growth and yearly disaster aggregates. I then improved the feature set to better capture the economic effect of natural disasters.

In the final version, the yearly dataset includes, in addition to GDP and GDP growth:
- `n_events`: number of disasters in a given year,
- `total_deaths`: total deaths from disasters,
- `total_damage`: total economic damage in thousand USD,
- `avg_magnitude`: average magnitude of recorded events,
- `log_total_damage`: log-transformed damages to reduce the impact of extreme values,
- `damage_share_gdp`: damages scaled by the level of GDP (relative intensity of disasters),
- `has_disaster`: indicator for whether at least one disaster occurred in a year.

I also include lagged versions of some variables (for example `gdp_growth_lag1`, `n_events_lag1`, `log_total_damage_lag1`, `damage_share_gdp_lag1`) to allow the models to capture delayed effects of disasters on GDP growth.
