# Titanium Model B: Probability Improvement V2

This experiment improves the required H3/H6 probability forecasts using the existing country-month dataset. It selects model and calibration settings on pre-July-2021 validation, then freezes them for historical evaluation. It does not achieve all acceptance gates and is not promoted as a replacement production model.

## Changes

- Compared seven classifiers and two forest ensembles using original, supplier-history and expanded feature sets.
- Selected by validation Brier score, preferring candidates that also pass both baseline comparisons and the maximum 10-bin calibration gap of 0.10.
- Compared temperature scaling and a blend with each country's historical class-frequency probabilities.
- Tested a mathematical share-bound rule: a future share lies between zero and one, so its possible change is between minus the current share and one minus the current share. This can remove classes that are impossible for that observation.
- Preserved the original class thresholds. No fabricated observations or benchmark series were added.

## Selected models

| Horizon | Estimator | Feature inputs | Calibration | Share-bound rule |
| --- | --- | ---: | --- | --- |
| H3 | Equal-weight Random Forest + Extra Trees ensemble | 66 | Temperature 1.25; no frequency blend | Enabled |
| H6 | Original-family Extra Trees classifier | 35 | Temperature 0.75; no frequency blend | Enabled |

The unselected frequency-blending options and all candidate scores are saved in `results/validation_configuration_comparison.csv`. Class prediction is always the largest of the four output probabilities.

## Results

Lower Brier is better. Pass requires beating both baselines and maximum calibration gap no greater than 0.10.

| Period | Horizon | Original Brier | New Brier | New accuracy | Maximum calibration gap | Probability gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Validation | H3 | 0.6428 | 0.6323 | 48.7% | 0.0901 | Pass |
| Validation | H6 | 0.6316 | 0.6300 | 45.8% | 0.0553 | Pass |
| Disruption | H3 | 0.6025 | 0.6064 | 52.8% | 0.2636 | Fail calibration |
| Disruption | H6 | 0.6237 | 0.6218 | 47.2% | 0.0826 | Pass |
| Post-disruption | H3 | 0.5835 | 0.5666 | 55.6% | 0.1124 | Fail calibration |
| Post-disruption | H6 | 0.5851 | 0.5827 | 54.4% | 0.5294 | Fail calibration |

All four historical test groups beat both Brier baselines. Post-disruption H3 probability error decreases by approximately 2.9% relative to the original model. The overall result is mixed: new H3 has worse disruption calibration, while H6 improves Brier slightly but retains later calibration failures. Exact accuracy remains well below 80%.

The largest H6 post-disruption calibration error occurs in a bin with five class-probability observations. H3 post-disruption misses the limit in a bin with 64 observations. Both count as failures; sparse bins were not dropped or merged to obtain a pass.

## Chronological evaluation

- Initial history: 2010 onward, with lag-complete observations beginning later.
- Selection folds: 2018, 2019, and January 2020-June 2021 origins, with target months restricted to June 2021 or earlier.
- For each selection block, only targets assumed available at its first forecast date are used for fitting.
- Disruption origins: July 2021-June 2023; each monthly forecast uses expanding history and only completed, available targets.
- Post-disruption origins: July 2023 onward. H3 ends at March 2026 origins; H6 ends at December 2025 origins, because observed targets end in June 2026.
- The existing test periods were already inspected in earlier experiments. They are reused historical evaluations, not a fresh prospective holdout. A new holdout remains necessary before production acceptance.

## Data and assumptions

The input is `../../data/titanium_model_b_country_month_training_2010_2026.csv`. It has 1,188 supplier-month observations through June 2026. The input hash, labels, month alignment and share sums are recorded in `results/source_audit.json`.

Sources are the existing UN Comtrade data, Bank of England GBP/USD rates and ONS ERUC/PPI. Supplier quantities are reconstructed from upstream supplier shares multiplied by HS weights. This experiment reconciles that reconstruction against the prepared upstream file, but does not establish validity against the original raw partner downloads.

FX availability is assumed at month-end plus seven days; ERUC at month-end plus 31 days. Trade availability is inherited from the prepared input. These assumptions remain part of the historical evaluation.

Specification gaps remain: HMRC rather than UN Comtrade is requested; 2005-2009 history is absent; required nickel/aluminium benchmark inputs remain absent under the previous source restriction; country membership was hardcoded upstream and needs full raw-partner verification. No Russia H3/H6 disruption observation has a collapse-class target under the frozen boundaries. That requirement is unresolved.

## Artifacts

- `code/train_probability_model_b.py`: training, calibration comparison, rolling evaluation, final model fitting, dashboard generation and prediction.
- `code/verify_probability_model_b.py`: independent checks for saved Brier scores and baselines, target alignment, training chronology, reload forecasts and class feasibility.
- `models/model_b_h3_calibrated.joblib`: portable H3 ensemble, preprocessing pipelines, calibration settings and historical frequencies.
- `models/model_b_h6_calibrated.joblib`: portable H6 pipeline and settings.
- `results/all_ground_truth_predictions.csv`: validation and both test periods with actual class, predicted class, four probabilities, correctness and target dates.
- `results/previous_vs_new_metrics.csv`: original model, previous accuracy experiment and this experiment compared.
- `results/reliability_bins.csv`: all ten bins, including empty and sparse bins.
- `results/country_metrics.csv` and `results/class_recall.csv`: supplier and class diagnostics.
- `results/latest_forecasts.csv` and `results/model_b_response.json`: H3/H6 country forecasts based on the last available June 2026 origin.
- `results/verification_report.json`: independent verification results.
- `visualizations/model_b_probability_improvement_dashboard.html`: standalone dashboard with period, horizon and supplier filters plus CSV download.

The reusable scripts refer to the existing original and earlier experiment files in their parent folders for feature definitions and comparisons. Keep this experiment within the existing Model B folder structure.

## Run

From this experiment folder:

```powershell
python code\train_probability_model_b.py
python code\verify_probability_model_b.py
python code\train_probability_model_b.py --predict-only
python code\train_probability_model_b.py --render-only
```

The four probabilities sum to one. `confidence` is their maximum; it is not a guaranteed probability that an individual prediction is correct. The latest forecast month follows the supplied observation cutoff, not the computer's current date.

Verification reproduced all six evaluation groups' Brier scores and baseline scores, confirmed target alignment and training timing, and reloaded the saved models to reproduce all 12 latest country-horizon forecasts. The local HTML filters and CSV download were exercised in a browser on desktop and mobile.
