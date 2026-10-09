# Titanium Model A - FX and ERUC Experiment

## Purpose

Predict the distribution of the direct GBP/kg titanium import-price change at H3, H6 and H12. HS 810820 and HS 810890 are modelled separately. H3 and H6 determine acceptance; H12 is report-only.

## Data

- Source training CSV: `C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM\TRAIN READY Processed data- Titanium 2010-2026\titanium_model_a_training_with_fx_eruc_2010_2026.csv`
- Coverage: January 2010 to June 2026
- Bank of England XUMAUSS converts USD values to GBP.
- ONS ERUC supplies broader imported non-ferrous-metal inflation changes.
- This experiment intentionally does not use World Bank aluminium/nickel or USGS titanium sponge, so it is not literal full-spec compliance.

## Models

1. Regularised linear quantile regression.
2. Gradient-boosted quantile regression.
3. Random-forest predictive distribution using quantiles across trees.
4. Calibrated no-change baseline for comparison.

Every forecast returns P05, P10, P25, P50, P75, P90 and P95. A split-conformal adjustment based only on blocked pre-test residuals calibrates the P05-P95 range.

## Evaluation

- Model selection: five blocked validation windows ending January 2022.
- Frozen disruption test: monthly origins from February 2022 to December 2023.
- Rolling retraining uses only targets available at each forecast date.
- Acceptance at H3/H6 requires at least 90% P05-P95 coverage, MIS below the baseline, no missing output and six-month coverage recovery by August 2022.
- Validation-selected H3/H6 combinations passing all rules: **1/4**.

## Outputs

- Dashboard: `C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM\TRAIN READY Processed data- Titanium 2010-2026\training titanium model A\visualizations\titanium_model_a_eruc_dashboard.html`
- Validation and test metrics: `results/`
- Validation-selected model bundles: `models/`
- Charts: `visualizations/`
- Latest direct reference forecasts: `results/latest_reference_forecasts.csv` and `.json`

## Run

```powershell
python "C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM\TRAIN READY Processed data- Titanium 2010-2026\training titanium model A\code\train_titanium_model_a_eruc.py"
```

Random seed: 42.
