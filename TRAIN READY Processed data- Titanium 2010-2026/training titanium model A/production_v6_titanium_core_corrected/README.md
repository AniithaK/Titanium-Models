# Titanium Model A Production V6: Titanium Core Corrected

## Data directories

- `data/`: the 134-column training snapshot consumed by model training.
- `models/`: saved selected-model bundles for both titanium HS codes.
- `results/`: validation, stress-test, post-disruption and latest forecast outputs.
- `monitoring/`: historical forecast and current feature-drift diagnostics.
- `visualizations/`: evaluation charts and the HTML dashboard.

`RUN_TRAINING_V6.cmd` retrains and evaluates the models from the isolated V6 data snapshot.

## Implemented

- Separate HS 810820 and 810890 direct H3, H6 and H12 models; HS 810830 remains excluded.
- GBP/kg target with Bank of England conversion.
- Price, quantity, supplier-share, supplier-concentration, Bank of England FX and ERUC/PPI features.
- World Bank aluminium/nickel and USGS titanium sponge features are explicitly excluded.
- Explicit low-volume, partner-anomaly, estimated-weight and target-reliability diagnostics.
- Quality-weighted model fitting.
- Six validation candidates, including robust Huber and Elastic Net models and a safe no-change fallback.
- Candidate selection using blocked validation ending January 2022.
- Acceptance-aligned selection minimizes calibrated MIS among validation-gate-eligible models.
- Selected models and the fixed baseline use separate validation-residual histories for calibration.
- Rolling retraining with target-availability checks.
- Volatility-regime-aware rolling conformal P05-P95 calibration.
- Separate 2022-2023 disruption and 2024-2025 post-disruption evaluations.
- Saved model bundles, production-style JSON, feature-drift monitoring and an HTML dashboard.

## Current result

- 2022-2023 stress combinations passing: **4/4**.
- 2024-2025 post-disruption combinations passing: **3/4**.

## Blocking limitations

- ERUC/PPI is a broad non-ferrous import-price proxy rather than a titanium-only benchmark.
- The 2024-2025 evaluation is useful additional evidence but is not pristine because the dataset was previously explored.
- Deployment remains blocked until acceptance criteria and a genuinely prospective holdout confirm performance.

## Reproduce

```powershell
& "C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM\TRAIN READY Processed data- Titanium 2010-2026\training titanium model A\production_v6_titanium_core_corrected\RUN_TRAINING_V6.cmd"
```
