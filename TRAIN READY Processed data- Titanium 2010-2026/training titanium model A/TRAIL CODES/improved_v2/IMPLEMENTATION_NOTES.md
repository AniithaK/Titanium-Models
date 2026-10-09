# Titanium Model A Improved V2

This directory is an isolated experiment. It does not overwrite the original ERUC model evidence.

Run:

```powershell
& ".\RUN_IMPROVED_V2.cmd"
```

The preparation step adds leakage-safe historical volatility, volume-regime, supplier-concentration, FX, seasonality and data-quality features. The training step compares weighted quantile/tree models, Elastic Net, Huber and an adaptive no-change fallback using pre-February-2022 validation only.

The frozen disruption test remains February 2022 through December 2023. Test results are diagnostic and never used to switch the selected model.
