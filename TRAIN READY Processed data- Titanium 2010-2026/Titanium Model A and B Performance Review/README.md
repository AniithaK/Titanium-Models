# Titanium Model A and Model B Performance Review

This package creates one local interactive dashboard from the evaluated outputs
of Titanium Model A and Titanium Model B. It does not change or retrain either
selected production-research model.

## Run

```powershell
& ".\RUN_REPORT.cmd"
```

The dashboard is written to:

`visualizations/titanium_model_a_b_performance_dashboard.html`

## Evidence

- Model A: actual GBP/kg, P05/P50/P95, coverage, errors, MIS and pinball loss.
- Model B: actual and predicted classes, full probability vectors, accuracy,
  macro F1, Brier, log loss and calibration.
- Model B apparent training metrics are diagnostic only. Model A does not save
  comparable in-sample predictions across its heterogeneous quantile models,
  so the report does not manufacture a train-loss comparison.

The generator verifies Model A interval metrics and Model B saved test metrics
before writing the dashboard.
