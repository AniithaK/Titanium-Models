# Titanium Model B: Country Supply Shift

## Intended Use

Provide SCI with country-level probabilities that UK titanium supply share will
collapse, fall, remain stable or rise at H3, H6 and H12. The model supports
supply-risk simulation and does not score individual companies.

## Target

For every supplier group and horizon, the target is the future absolute change
in UK import-weight share:

```text
share_change_h = share[t+h] - share[t]
```

The fixed class boundaries come from pooled H3 changes ending by June 2021:

- Collapse: below -14.4608 percentage points.
- Fall: -14.4608 to below -2.2945 percentage points.
- Stable: -2.2945 through 2.0887 percentage points.
- Rise: above 2.0887 percentage points.

The supplied data starts in 2010, not the requested 2005.

## Inputs

- Country quantity and supply-share history over 1, 3, 6 and 12 months.
- Total UK titanium import-weight changes.
- Combined titanium GBP/kg import-unit-value changes.
- Bank of England FX changes.
- ONS ERUC/PPI changes.
- Supplier concentration, seasonality and source-quality indicators.

Only information assumed published by the simulated forecast date is used.

## Models

- H3: Random Forest, probability temperature 1.15.
- H6: Extra Trees, probability temperature 0.75.
- H12: Random Forest, probability temperature 1.35; reporting-only.

Candidate families were multinomial logistic regression, Random Forest and
Extra Trees. Selection used blocked pre-July-2021 validation only. The selected
family and probability temperature were frozen before the disruption test.

## Evaluation

### July 2021-June 2023 disruption test

| Horizon | Model Brier | Last-change baseline | Historical-frequency baseline | Maximum calibration gap | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| H3 | 0.6026 | 1.4306 | 0.6539 | 0.0767 | Pass |
| H6 | 0.6237 | 1.4028 | 0.6900 | 0.0826 | Pass |
| H12 | 0.6215 | 1.5556 | 0.6890 | 0.1976 | Reporting only |

### Post-disruption monitoring

H3 and H6 continue to beat both Brier baselines, but their maximum 10-bin
calibration gaps exceed 0.10. This blocks production approval and indicates that
probability calibration needs prospective monitoring or recalibration.

## Limitations

- Supplier quantities are reconstructed from the aggregated supplier shares in
  the supplied trade-core data rather than rebuilt from raw partner rows.
- The source history begins in 2010 rather than 2005.
- The specification requested nickel and aluminium price changes. The user
  restricted this implementation to UN Comtrade, Bank of England and ONS
  inflation data, so those cross-metal inputs are intentionally absent.
- ERUC is a broad basic-metals inflation proxy, not a titanium-specific index.
- H12 has too few independent periods for model acceptance.
- The model is a research prototype until post-disruption calibration and a
  genuinely prospective holdout are satisfactory.
