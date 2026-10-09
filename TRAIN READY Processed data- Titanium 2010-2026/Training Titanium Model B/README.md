# Titanium Model B: Supply Shift by Country

This package predicts the probability that each main supplying country's share
of UK titanium imports will **collapse**, **fall**, remain **stable**, or **rise**
over 3, 6 and 12 months. H3 and H6 are the primary acceptance horizons. H12 is
reporting-only.

## Permitted Sources

- UN Comtrade monthly UK imports for HS 810820 and 810890.
- Bank of England monthly USD-per-GBP exchange rates.
- ONS ERUC/C24 basic-metals import-price inflation.

USGS, Metals-API and World Bank cross-metal price inputs are not used. HS 810830
waste and scrap is excluded.

## Important Specification Deviation

The supplied monthly data begins in January 2010. The specification requested
class-boundary history from 2005, so the fixed class boundaries use January
2010 through June 2021. This must remain visible in any review or deployment
decision.

## Structure

- `code/`: deterministic preparation and training pipelines.
- `data/`: country-month training matrix and frozen class thresholds.
- `models/`: selected fitted H3, H6 and H12 estimators in one joblib bundle.
- `results/`: validation, disruption, post-disruption and latest forecasts.
- `visualizations/`: standalone HTML dashboard and PNG charts.
- `documentation/`: model card and methodology limitations.

## Run

```powershell
& ".\RUN_MODEL_B.cmd"
```

The preparation step does not modify either source CSV. It reconstructs country
weights from the fixed supplier shares, combines HS 810820 and 810890 by weight,
and writes a new country-month dataset.

## Current Result

During the July 2021-June 2023 disruption test:

- H3 Random Forest passes: Brier 0.6026, below both baselines; maximum
  calibration gap 0.0767.
- H6 Extra Trees passes: Brier 0.6237, below both baselines; maximum calibration
  gap 0.0826.
- H12 is reporting-only and does not meet the calibration criterion.

Post-disruption H3 and H6 still beat both Brier baselines, but fail the strict
maximum-bin calibration rule. The package is therefore a research prototype,
not a production-approved model.
