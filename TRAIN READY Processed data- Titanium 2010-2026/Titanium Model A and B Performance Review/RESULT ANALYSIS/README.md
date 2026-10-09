# Titanium Model A and B Result Analysis

This folder separates historical ground-truth evaluation from forecasts whose target months occur after the June 2026 source-data cutoff.

## Main files

- `Titanium_Model_A_B_Result_Analysis.xlsx`: summary, editable charts, complete scored rows, future forecasts and metric definitions.
- `Titanium_Model_A_B_Result_Analysis_Report.pdf`: end-to-end interpretation with tables and visualizations.
- `model_a_ground_truth_vs_predictions_analysis.csv`: 630 scored price-forecast rows with signed and absolute percentage deviation, tolerance flags, direction correctness and interval coverage.
- `model_a_performance_analysis.csv`: Model A metrics by evaluation period, HS code and horizon.
- `model_b_ground_truth_vs_predictions_analysis.csv`: 1,584 supplier-shift rows with actual class, predicted class, probabilities, confidence and correctness.
- `model_b_performance_analysis.csv`: Model B accuracy, balanced accuracy, macro F1, Brier score, log loss and calibration metrics.
- `model_a_future_forecasts_2026_2027.csv`: H3, H6 and H12 price forecasts from the June 2026 origin.
- `model_b_future_forecasts_2026_2027.csv`: H3, H6 and H12 supplier-shift probabilities from the June 2026 origin.
- `*.png`: standalone performance and forecast visualizations.

## Important interpretation

- Model A percentage deviation compares the P50 price with the observed GBP/kg price.
- Model B confidence is the probability assigned to the predicted class. It is not guaranteed to equal the real-world chance of being correct because calibration is imperfect.
- September 2026, December 2026 and June 2027 targets do not have ground truth in the current source dataset and are not included in historical performance scores.
