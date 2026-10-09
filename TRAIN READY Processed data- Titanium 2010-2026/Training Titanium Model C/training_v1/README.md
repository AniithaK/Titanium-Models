# Model C training and evaluation, run 1

This run compares four regression approaches with a no-change baseline for international freight-cost movements. It uses the existing BLS freight index and World Bank Brent dataset. It is a research evaluation because publication dates are assumed and the source downloads do not reconstruct historical revision vintages.

## Models and outputs

| Model | How it predicts the distribution |
| --- | --- |
| Ridge + residual quantiles | Standardize the eight inputs within the fitting fold, fit a Ridge mean-change model, and add empirical training residual quantiles. |
| Quantile Random Forest | Fit trees to log changes; predict quantiles from historical target values in the leaves reached by the input. Each tree contributes equal total weight. |
| Quantile Extra Trees | The same leaf-distribution quantile method with more randomized tree splits. |
| Gradient Boosting quantiles | Fit a separate quantile-loss regressor at each of the seven required probabilities. |
| No-change baseline | Predict median log change zero, with initial quantiles based on past no-change errors. |

Each horizon is trained directly against `target_freight_log_change_h3`, `target_freight_log_change_h6` or `target_freight_log_change_h12`. H6/H12 are not recursive H3 predictions. Every bundle returns P05, P10, P25, P50, P75, P90 and P95 log changes. Quantiles are sorted to prevent crossing, then calibrated.

Convert a predicted log change q to percentage movement using `100*(exp(q)-1)`, or to an index level using `origin_freight_index*exp(q)`. An index level is not a route-specific shipping price in GBP.

There are 15 portable joblib bundles: four ML models and the baseline for each of three horizons. A Gradient Boosting bundle contains seven quantile estimators. H12 is reporting-only and its metrics are excluded from model selection.

## Results of this run

Validation selected Gradient Boosting for H3. It achieved 94.6% validation coverage and a MIS ratio of 0.989. No H6 ML candidate passed both validation coverage and MIS requirements, so the baseline remains selected; Ridge is the best H6 ML candidate by MIS, but its validation coverage is only 86.8%. H12 reports Gradient Boosting based on the H3/H6 family comparison, not on H12 scores.

| Algorithm | H3 disruption coverage | H3 MIS / baseline | H6 disruption coverage | H6 MIS / baseline |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 91.7% | 0.800 | 75.0% | 1.110 |
| Random Forest | 95.8% | 0.683 | 62.5% | 1.280 |
| Extra Trees | 91.7% | 0.725 | 62.5% | 1.175 |
| Gradient Boosting | 87.5% | 0.845 | 70.8% | 0.937 |
| No-change baseline | 79.2% | 1.000 | 66.7% | 1.000 |

Random Forest is the strongest H3 disruption candidate and passes coverage, MIS and the recorded recovery gate. However, it did not beat the baseline MIS in validation, so it does not replace the validation-frozen H3 selection after looking at the test. The selected Gradient Boosting H3 model fails test coverage and recovery. No ML model passes H6 test coverage. This is a mixed research result, not production acceptance.

H3 median index MAPE during disruption ranges from approximately 3.7% to 3.9% for the ML models, versus 4.5% for the baseline. H6 median MAPE ranges from 7.3% to 8.6%, versus 8.4% for the baseline. Good median error alone cannot compensate for poor interval coverage or MIS.

## Inputs and split

The input CSV is `../Train/model_c_freight_training_2005_2026.csv`: 260 reference months, January 2005-August 2026. Only the eight named historical freight/Brent features in the preparation manifest enter the estimators. Target columns, dates, eligibility fields and assumed-status flags are not predictors.

- Initial history: 2005-2014.
- Main validation: 2015-2019, with eligible labels known before 1 January 2020. Fits expand annually at the first eligible origin of each year; each month's prediction uses its own observed features. There are 56 H3, 53 H6 and 47 H12 eligible validation origins.
- Reference-month disruption test: January 2020-December 2021, 24 forecasts per model and horizon, retraining each month.
- Additional specification issue-date test: forecasts whose assumed issue dates fall in January 2020-December 2021. This also contains 24 forecasts per model/horizon but moves boundary reference months. It is reported separately.
- Post-disruption monitoring: 2022 onward, limited to observed future targets and also fitted monthly.
- Late pre-2020 origins whose outcomes are not yet known at the selection cutoff are recorded as `pretest_bridge`; they do not affect selection. They supply completed historical predictions for recovery diagnostics.

The disruption period labels forecast origins, not outcome dates. H3 targets extend through March 2022, H6 through June 2022 and H12 through December 2022 for the reference-month test. Targets are never cut off at December 2021 merely to improve coverage.

The user approved 2015-2019 main validation rather than the specification's broader 2012-2019 period. No separate 2012-2014 backtest was run in this version. Those years remain in the initial history.

## Historical timing and calibration

For every fit, origin dates must precede the evaluated origin, and target availability must be on or before its assumed issue date. The latest 24 eligible labelled observations are held out for calibration. The estimator fits earlier proper-training rows; their targets are additionally purged so they were already observable at the first calibration origin's assumed issue date.

Calibration uses the held-out scores `max(P05-y, y-P95)` and their finite-sample 90% order statistic. Negative expansions are set to zero. P05/P95 expand by the same nonnegative amount; intermediate quantiles receive proportional shifts around P50. This retains ordered quantiles and the model's median. The baseline uses exactly the same holdout, purge and calibration method with its own errors.

Both fitting and calibration use only historical observations. Overlapping targets and nonstationary time series violate the simple independent/exchangeable assumptions behind standard conformal guarantees, so nominal 90% coverage is tested rather than promised.

Model settings, random seed and calibration protocol were fixed before testing. Per-horizon H3/H6 selections prefer an ML candidate with validation coverage at least 90%, no missing forecasts and MIS lower than the baseline. If none qualifies, the baseline is selected and the best ML candidate is still identified. H12 uses the H3/H6 family comparison only.

## Metrics and recovery

- P05-P95 coverage, with missing forecasts counted as misses.
- Mean interval score at alpha=0.10, in log-change units. The MIS ratio is model MIS divided by the equally calibrated baseline MIS; lower than one is better.
- Mean pinball loss across all seven quantiles.
- Median MAE/RMSE in log-change and index units, and MAPE/sMAPE of the median index prediction.
- Recovery of six-completed-prediction coverage after disruption. A recovery credit requires all six target months to be January 2020 or later. Scores are calculated only after outcomes are assumed published; the deadline is six months after the January 2020 issue-month end.

With six completed observations, achieving at least 90% coverage requires six of six inside the interval. `recovery_summary.csv` records the precise definition and dates. This operational recovery interpretation is explicit because the specification does not fully define origin-month versus completed-outcome timing.

The baseline cannot beat its own MIS, so its acceptance-comparison flag is false by definition; it is a reference, not an ML candidate failing a self-comparison. It may still be selected when no ML candidate meets validation requirements.

## Requested artifacts

- `visualizations/model_c_training_dashboard.html`: standalone local dashboard with model, horizon and period filters, actual-versus-P50 chart, 90% interval, tables and CSV download.
- `visualizations/actual_vs_predicted_<period>_h<h>.png`: comparison panels for every model, including baseline, for validation, disruption and post-disruption.
- `visualizations/train_validation_loss_<model>.png`: each algorithm's diagnostic training/validation curves.
- `results/evaluation_metrics.csv`: all metrics and baseline comparisons, with the supplementary issue-date test separately labelled.
- `results/validation_metrics.csv` and `results/selected_models.json`: pre-test selection evidence.
- `predictions/disruption_test_actual_vs_predicted.csv`: actual index, predicted index, actual log change, seven predicted quantiles, target months and errors for all models/horizons.
- `predictions/specification_issue_date_test_actual_vs_predicted.csv`: the alternative issue-date-aligned test records.
- `predictions/validation_actual_vs_predicted.csv` and `post_disruption_actual_vs_predicted.csv`: other completed-outcome comparisons.
- `predictions/in_sample_training_diagnostic.csv`: diagnostic predictions on the first proper-training set. These are explicitly in-sample and are not used for acceptance.
- `results/train_validation_loss_curves.csv`: loss-curve numeric records.
- `predictions/latest_forecasts_all_models.csv` and `latest_selected_forecasts.csv`: direct forecasts based on the last August 2026 observation.
- `results/model_c_response.json`: selected H3/H6/H12 response example.
- `models/<algorithm>/model_c_h<h>.joblib`: fitted estimator(s), leaf/residual distributions, calibration parameters and fitting metadata.
- `results/feature_influence.csv`: diagnostic importances; Ridge's standardized coefficient magnitudes and tree impurity importances are different measures and do not establish causal effects.
- `visualizations/feature_importance_<model>.png`: one chart per ML model, each containing H3/H6/H12 panels. Ridge displays unsigned coefficient magnitudes normalized within horizon; tree models display impurity importance and Gradient Boosting uses its P50 estimator only. The graphs are also embedded in the dashboard.
- `results/feature_importance_normalized.csv`: plotted percentages, original scores, methods and readable feature descriptions. All 96 scores were checked against their saved fitted models. The baseline has no learned feature-importance graph.
- `results/run_manifest.json`: frozen settings, source hash and limitations.
- `results/verification_report.json`: independent checks after training.

## Training and validation loss curves

All curves use raw, uncalibrated mean pinball loss. The diagnostic training set is the initial proper-training subset; the fixed validation set contains eligible 2015-2017 origins.

- Ridge: vary the number of chronological training observations. It has no epoch curve.
- Forests: vary the number of trees.
- Gradient Boosting: use actual staged predictions from the fitted quantile regressors.

These diagnostics help inspect fit behavior. They do not use the disruption test, determine acceptance or claim classification accuracy. The calibrated out-of-sample metrics are stored separately.

## Limitations

The preparation assumes source observations become usable at the end of the following calendar month. Actual historical release dates and vintage values remain unverified, so these results must not be described as verified point-in-time production acceptance. The BLS index is a contract-oriented US deep-sea freight proxy. The required comparison with published FBX spot rates for 2020-2021 is still outstanding. COVID and its dates were known when setting up the study; the test window comes from the specification.

## Run

Keep `training_v1` inside the Model C folder beside the existing `Train` directory.

```powershell
python code\train_model_c.py
python code\verify_model_c.py
python code\train_model_c.py --predict-only
python code\train_model_c.py --render-only
python code\plot_feature_importance.py
```

Run these from `training_v1`. A completed run refuses to retrain over its results; `--render-only` rebuilds visualizations from saved results. The final selected forecasts use August 2026 as reference month, with the recorded September 2026 assumed issue date. Target months therefore follow August plus the respective horizon, not the computer's current month.
