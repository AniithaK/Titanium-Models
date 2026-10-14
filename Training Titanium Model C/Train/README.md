# Model C monthly training data

`model_c_freight_training_2005_2026.csv` is the single prepared training dataset. It contains 260 monthly rows from January 2005 through August 2026 and 67 columns: raw measures, eight model inputs, future labels and audit/eligibility fields. No models have been trained at this preparation stage.

## Sources and processing

1. Read `Data/bls_deep_sea_freight_ppi_monthly_1988-06_to_2026-08.csv`, series PCU483111483111, downloaded from https://fred.stlouisfed.org/series/PCU483111483111. Values are monthly, not-seasonally-adjusted index points (June 1988=100), not a quoted freight charge.
2. Read the `Monthly Prices` sheet and `Crude oil, Brent` column in `Data/world_bank_pink_sheet_monthly_1960-01_to_2026-09.xlsx`, from https://www.worldbank.org/en/research/commodity-markets. Brent values are USD per barrel.
3. Require unique, consecutive calendar months and positive observations. Join the sources on month using a one-to-one merge. Do not fill missing observations or interpolate.
4. Use 2004 solely to construct the 12-month lag and volatility features. Export reference months from January 2005 onward. This provides complete features in the first exported month without adding 2004 labelled training rows.
5. Compute the eight historical predictors below. No scaling, imputation, outlier clipping or model fitting is performed on the full dataset. A future training pipeline must fit any preprocessing only on its training fold.
6. Create H3/H6/H12 future targets from observed freight values. Retain unobserved targets as blank cells.
7. Record the requested split, prototype availability policy, target observation flags and validation cutoff controls.
8. Reconcile raw values, lag features, volatility and targets independently. Preserve both source files and record their SHA256 hashes in `model_c_preparation_manifest.json`.

## Input features

Only these eight columns should be passed to the estimator by default:

| Column | Calculation | Meaning |
| --- | --- | --- |
| freight_log_change_1m | ln(F[t]/F[t-1]) | Most recent freight movement |
| freight_log_change_2m | ln(F[t]/F[t-2]) | Two-month freight movement |
| freight_log_change_3m | ln(F[t]/F[t-3]) | Three-month freight movement |
| freight_log_change_6m | ln(F[t]/F[t-6]) | Six-month freight movement |
| freight_log_change_12m | ln(F[t]/F[t-12]) | Annual freight movement |
| freight_volatility_6m | Population standard deviation of latest six monthly log changes | Recent variability |
| freight_volatility_12m | Population standard deviation of latest twelve monthly log changes | Longer-term variability |
| brent_log_change_1m | ln(B[t]/B[t-1]) | Most recent crude-oil movement |

Volatility uses `ddof=0`, includes the current reference month's observed change and is not annualized. It is the variability of returns, not the standard deviation of raw index levels. All features use present/past source observations only.

## Targets

For h=3,6,12:

- `target_month_h<h>`: reference month t plus h calendar months.
- `target_freight_index_h<h>`: observed F[t+h], when available.
- `target_freight_log_change_h<h>`: ln(F[t+h]/F[t]); the supervised target for quantile models.
- `target_freight_pct_change_h<h>`: 100*(exp(log change)-1), for human interpretation.
- `target_observed_h<h>`: whether the future actual value exists in the input snapshot.
- `target_available_date_assumed_h<h>`: scheduled prototype availability of that target. It is a timing assumption, not proof that the value exists.

Target columns must not be passed as input features. H3 has 257 observed targets, H6 254 and H12 248. The final 3/6/12 target values respectively remain blank. H12 is reporting-only; select models using H3/H6.

## Agreed split

The requested periods describe the month of the origin observation, stored in `date` and `period`:

| Stage | Reference months | Origin rows |
| --- | --- | ---: |
| Initial training | 2005-2014 | 120 |
| Main expanding validation | 2015-2019 | 60 |
| Disruption test | 2020-2021 | 24 |
| Post-disruption evaluation | 2022-August 2026 | 56 |

`origin_period_group` records those periods. Because publication happens later than the reference month, `forecast_date_assumed` and `forecast_issue_period_group_assumed` separately record the prototype issue date and its period. For a strict test of forecasts actually issued in January 2020-December 2021, future training code must select by the issue-date field rather than blindly using the reference-month label. This distinction can move boundary observations between periods.

The specification mentions selection forecasts during 2012-2019. The user chose 2015-2019 as the main validation period. `optional_early_backtest_origin` marks 2012-2014 for an additional chronological diagnostic. Those rows remain part of the initial training history when fitting the first main-validation model.

## Preventing future-label leakage

`research_training_eligible_h<h>` checks complete features and an observed target. It does not alone enforce a particular forecast cutoff. For every historical fit, training code must also require:

```text
training origin date < evaluated origin date
target_available_date_assumed_h<h> <= evaluated forecast_date_assumed
```

`initial_training_fit_eligible_h<h>` adds the initial-training split and target-availability condition at the first validation origin. This gives 118 H3, 115 H6 and 109 H12 initial-fit rows.

`validation_selection_eligible_h<h>` requires a validation origin with features and target assumed available before 1 January 2020. This gives 56 H3, 53 H6 and 47 H12 selection rows. Late-2019 origins remain in the CSV but do not enter pre-disruption model selection if their outcomes are still unknown. These counts depend on the explicit timing assumption and must be refreshed once verified release dates are available.

`origin_period_test_eligible_h<h>` checks observed targets in the 2020-2021 reference-month group. `forecast_issue_test_eligible_assumed_h<h>` separately checks the 2020-2021 issue-date group. Future modelling must explicitly choose the intended evaluation definition and report it.

## Timing and revision limitations

Neither downloaded file contains the full historical publication-date/vintage records. To make a prototype pipeline possible, the separate fields named `*_available_date_assumed` treat observations as available at the end of the following calendar month. This is a conservative research convention, not a verified historical release calendar; actual delays can differ.

The verified release-date columns remain blank. `historical_release_dates_verified` and `historical_data_vintages_verified` are false. Therefore `strict_point_in_time_verified_h<h>` is false for every row. This does not mean the numerical features are missing: they are complete. It means strict historical information availability has not been established.

Current historical downloads may contain revisions made after the original month. An assumed lag cannot reconstruct the values available at the time. Confirm release dates and retrieve appropriate historical vintages before claiming strict point-in-time test compliance. Source file modification timestamps in the manifest are recorded as file provenance, not claimed download dates.

The required BLS-versus-FBX comparison for 2020-2021 remains outstanding because FBX data has not been supplied.

## Reproduce

The preparation script is `../code/prepare_model_c_training_data.py`. Run it from the Model C folder:

```powershell
python code\prepare_model_c_training_data.py
```

It refuses to overwrite an existing CSV. Source paths, feature list, split counts, formulas, assumptions, source hashes and verification results are recorded in `model_c_preparation_manifest.json`.
