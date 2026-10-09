# Titanium Model B Accuracy Improvement Experiment

## Objective

Improve Model B's exact four-class supplier-share movement accuracy without changing the established `collapse`, `fall`, `stable`, and `rise` definitions, using future information, or selecting models on the test periods.

## Data and evaluation design

- Source dataset: `../data/titanium_model_b_country_month_training_2010_2026.csv`
- Validation/model selection: January 2018 through June 2021
- Disruption test: July 2021 through June 2023 forecast origins
- Post-disruption test: July 2023 through the latest completed target
- Forecast horizons: H3, H6, and H12
- Class thresholds: unchanged from `../data/model_b_fixed_class_thresholds.json`

## Improvements tested

- Supplier-share lag levels at 1, 3, 6, and 12 months.
- Rolling supplier-share means, volatility, ranges, and distance from recent means.
- Supplier rank, distance from the market leader, momentum, and acceleration.
- Tuned logistic regression, random forest, Extra Trees, histogram gradient boosting, and gradient boosting classifiers.
- Continuous future share-change regressors converted to the original four classes.
- A validation-selected top-three soft-voting ensemble.

## Selected validation models

| Horizon | Selected model | Validation exact accuracy |
| --- | --- | ---: |
| H3 | Extra Trees classifier | 49.1% |
| H6 | Gradient boosting share-change regressor | 45.8% |
| H12 | Extra Trees share-change regressor | 48.3% |

## Untouched test results

| Period | Horizon | Exact accuracy | Adjacent-class accuracy | Change from original |
| --- | --- | ---: | ---: | ---: |
| Disruption | H3 | 50.7% | 84.7% | +0.7 pp |
| Disruption | H6 | 50.7% | 86.8% | +3.5 pp |
| Disruption | H12 | 41.7% | 84.0% | -3.5 pp |
| Post-disruption | H3 | 54.0% | 91.4% | -2.0 pp |
| Post-disruption | H6 | 55.0% | 93.9% | +0.6 pp |
| Post-disruption | H12 | 56.3% | 97.2% | -2.1 pp |

## Conclusion

The 80% exact-accuracy target was not achieved. The expanded features and algorithms produced only small, inconsistent changes relative to the original Model B. Therefore this experiment should not replace the original model solely on accuracy.

Adjacent-class accuracy is above 80%, meaning most errors are one neighboring movement band away. This is useful as a diagnostic but is not the same as exact four-class accuracy and must not be reported as such.

Reaching 80% would likely require a materially different task or new predictive data, such as supplier-level contracts, capacity, sanctions, freight constraints, or lead-time indicators. Merging classes into a simpler target could also raise accuracy, but that would be a specification change and requires explicit approval.

## Outputs

- Training code: `train_model_b_improved_accuracy.py`
- Saved experimental model: `models/titanium_model_b_improved_accuracy.joblib`
- Full validation comparison: `results/validation_candidate_comparison.csv`
- Test predictions: `results/disruption_and_post_disruption_predictions.csv`
- Original-versus-improved metrics: `results/original_vs_improved_metrics.csv`
- Latest forecasts: `results/latest_improved_forecasts.csv`
- Interactive dashboard: `visualizations/titanium_model_b_improved_accuracy_dashboard.html`
- Charts: `visualizations/*.png`

Use `python train_model_b_improved_accuracy.py` to rerun training and evaluation. Use `python train_model_b_improved_accuracy.py --render-only` to rebuild the dashboard and charts from saved CSV results.
