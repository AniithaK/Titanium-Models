# Titanium Model A Production V3

## Data directories

- `raw/`: immutable snapshots of the two upstream feature matrices used by the V3 preparation step.
- `prepared/`: engineered V3 matrix plus source-audit and target-quality diagnostics.
- `data/`: synchronized training snapshot consumed by model training and reusable prediction.

`RUN_PRODUCTION_V3.cmd` rebuilds `prepared/` from `raw/`, copies the exact training matrix into `data/`, and then trains and evaluates the models.

## Implemented

- Separate HS 810820 and 810890 direct H3, H6 and H12 models; HS 810830 remains excluded.
- GBP/kg target with Bank of England conversion.
- Price, quantity, supplier-share, supplier-concentration, FX, ERUC, World Bank and USGS features.
- Explicit low-volume, partner-anomaly, estimated-weight and target-reliability diagnostics.
- Quality-weighted model fitting.
- Six validation candidates, including robust Huber and Elastic Net models and a safe no-change fallback.
- Candidate selection using blocked validation ending January 2022.
- Rolling retraining with target-availability checks.
- Volatility-regime-aware rolling conformal P05-P95 calibration.
- Separate 2022-2023 disruption and 2024-2025 post-disruption evaluations.
- Saved model bundles, production-style JSON, feature-drift monitoring and an HTML dashboard.

## Current result

- 2022-2023 stress combinations passing: **2/4**.
- 2024-2025 post-disruption combinations passing: **3/4**.

## Blocking limitations

- The official World Bank September 2026 workbook URL was identified, but command-line DNS prevented download. The local World Bank source is used with source-age and stale flags.
- The official USGS titanium-metal workbook available locally ends in 2020. Later rows carry the last known annual observation with a source-age feature and stale flag.
- The 2024-2025 evaluation is useful additional evidence but is not pristine because the dataset was previously explored.
- Deployment remains blocked until required sources are refreshed and a genuinely prospective holdout confirms performance.

## Reproduce

```powershell
& "C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM\TRAIN READY Processed data- Titanium 2010-2026\training titanium model A\production_v3\RUN_PRODUCTION_V3.cmd"
```
