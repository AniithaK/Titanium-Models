# Model C improvement experiments

Two improvement experiments are saved here. They produced some gains in H6 median forecasts during disruption, but no consistent overall improvement. Do not replace the original `training_v1` models with these experiments as a production recommendation.

## V2: recent models and out-of-sample error calibration

The root folder contains four new ML candidates plus a no-change baseline:

| Candidate | Change from the earlier study |
| --- | --- |
| Recent-weighted Ridge | Stronger regularization, latest ten years of eligible labels, exponentially greater weight on recent observations |
| Regularized linear median | Median quantile regression with L1 regularization and recent-observation weights |
| Shallow weighted Random Forest | Shallower trees and larger leaves to reduce overfitting |
| Regularized median boosting | Thirty shallow boosting stages rather than seven separate deep quantile regressors |
| No-change baseline | Median change remains exactly zero; its distribution is calibrated from its own completed historical errors |

All models use the same eight existing freight/Brent inputs. No source observations, labels or external predictor series were invented or altered. Training uses the latest 120 eligible observations and a 60-month recency half-life. These are model settings, not new measured data.

The previous study held out 24 recent labels and fitted the estimator on older proper-training data. V2 instead fits every eligible recent label and calibrates using earlier forecasts that were generated out of sample. A calibration observation enters only after its target is assumed published. Its original model fit also predates that target. Completed historical errors are optionally divided by the volatility scale recorded when the forecast was made, then their ordered quantiles generate the current forecast distribution.

Pre-2020 validation compares 24-, 36- and 48-month calibration windows, with no scaling, six-month volatility, twelve-month volatility or their maximum. A data-derived tenth-percentile historical volatility floor avoids dividing by negligible values. The numerical epsilon is only a zero-division guard. The scale uses sqrt(h) for horizon h. These are explicit uncertainty-model choices; they are not verified physical freight laws or guaranteed coverage.

Every ML configuration is compared with the no-change baseline using the identical window and scale. It must also beat the strongest no-change configuration selected using validation. This prevents claiming improvement merely by choosing a weak comparison baseline.

## V3: additional RMS calibration study

`rms_calibration_v3` preserves a further calibration experiment without overwriting V2. It adds six-/twelve-month RMS return scales:

```text
RMS = sqrt(return_variance + mean_monthly_return^2)
mean_monthly_return = historical 6m/12m freight log change / window
```

This includes persistent observed trends, which standard deviation alone removes. Both variance and mean movement come from existing features. The V2 raw forecast cache is reused; models are not retrained across the entire history. Final bundles are refitted only to save the chosen latest model artifacts.

The RMS options did not improve the configurations chosen by validation. V3 therefore produced the same selected-model results as V2. Its development note explicitly records that the new calibration candidate was proposed after observing V2 failures. Historical test results are reused development evidence, not an untouched final holdout.

## Split and historical procedure

- Input: the existing 260-row `Train/model_c_freight_training_2005_2026.csv` in the parent Model C folder.
- Initial labelled history begins in 2005. Raw rolling forecasts start in 2009 to supply past out-of-sample calibration errors before validation.
- Validation: eligible 2015-2019 reference-month origins, with labels available before 2020. Estimators refit quarterly; calibration updates sequentially from known outcomes.
- Disruption: monthly refits during 2020-2021 reference months, with a separate file for forecasts assumed issued during calendar 2020-2021.
- Later monitoring: 2022 onward, with monthly refits and only completed targets scored.
- H12: reporting-only. Its family is chosen using H3/H6 validation results; its calibration settings are borrowed from H6 rather than tuned on H12 outcomes.

The existing disruption and post-disruption data were already viewed during `training_v1`. Neither experiment selects settings using those rows, but they cannot serve as a new untouched test set. A prospective holdout is still required for a stronger acceptance claim.

## What the results show

The new H3 validation selection is the shallow Random Forest with 48 past errors and no volatility normalization. The selected H6 remains the no-change baseline with 48 errors scaled by six-month volatility. No H6 ML candidate beats both validation baselines while meeting coverage.

| Comparison | Earlier result | New result | Interpretation |
| --- | ---: | ---: | --- |
| Selected H3 validation MIS | 0.1510 | 0.1275 | 15.6% lower, with validation coverage still above 90% |
| Selected H3 disruption coverage | 87.5% | 79.2% | Worse generalization |
| Selected H3 disruption MIS | 0.2438 | 0.2893 | 18.7% worse |
| Selected H3 later MIS | 0.3687 | 0.4356 | 18.2% worse |
| H6 disruption median error: earlier baseline vs new linear candidate | 8.38% | 5.44% | Central-forecast improvement, not an acceptance pass |
| H6 disruption coverage: earlier best ML coverage vs new linear candidate | 75.0% | 83.3% | Improvement, but still below 90% |
| H6 later median error: earlier baseline vs new linear candidate | 7.62% | 11.58% | Improvement did not persist |

The new linear H6 candidate is not the validation-selected model. It has disruption MIS 0.4644, but its matched baseline MIS is lower at approximately 0.4183. It therefore fails the baseline comparison as well as coverage. No new ML candidate passes all required H3/H6 historical coverage, MIS and recovery gates.

The original Random Forest H3 candidate remains stronger during the previously reported disruption test (95.8% coverage and MIS 0.1970), though it was not the original validation selection. This experiment does not promote it by looking at the reused test.

## Outputs

- `code/train_improved_model_c.py`: V2 fitting, historical raw forecasts, validation selection, evaluation, latest model saving and dashboard.
- `code/verify_improved_model_c.py`: independent score, calibration, chronology and model-reload checks. Supports `--run-dir` for V3.
- `models/<candidate>_h3/h6/h12.joblib`: 15 portable candidate/baseline artifacts per experiment.
- `results/evaluation_metrics.csv`: coverage, MIS, matched and best-baseline ratios, pinball loss, median MAE/RMSE/MAPE, and recovery status.
- `results/validation_configuration_comparison.csv`: all calibration choices scored on validation.
- `results/previous_vs_improved_selected.csv`: old versus new validation-selected models across periods.
- `predictions/all_predictions.csv`: actual and predicted freight-index levels, log changes and all seven quantiles.
- `predictions/disruption_test_actual_vs_predicted.csv`: the reference-month disruption test.
- `predictions/specification_issue_date_test.csv`: the alternative issue-date-aligned test records.
- `predictions/matched_baseline_predictions.csv`: exact calibration-matched baseline records for each candidate.
- `predictions/latest_forecasts.csv` and `results/model_c_response.json`: latest direct H3/H6/H12 distributions.
- `visualizations/improved_model_c_dashboard.html`: standalone page with filters, actual/prediction curves and CSV download.
- `results/verification_report.json`: checked quantiles, actual-value alignment, past-only calibration, scores and model reloads.

All numerical source values are preserved. Historical availability dates are still assumptions; original release-date and revision-vintage records are absent. The BLS-versus-FBX comparison remains outstanding. These limitations prevent strict point-in-time production acceptance regardless of a favorable historical metric.

## Reproduce

From this folder:

```powershell
python code\train_improved_model_c.py
python code\verify_improved_model_c.py
python code\train_improved_model_c.py --render-only
python rms_calibration_v3\code\train_rms_calibration.py
python code\verify_improved_model_c.py --run-dir rms_calibration_v3
```

Keep this folder under Model C beside `Train` and `training_v1`. The code reads those existing files for definitions and comparisons. Completed experiments refuse to overwrite their final results.
