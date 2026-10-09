# Methodology and Interpretation

## Target

For horizon h, the target is `ln(GBP_price[t+h]) - ln(GBP_price[t])`. H3, H6 and H12 are fitted directly; they are not recursive H1 forecasts.

## Feature set

The predictors are GBP price changes over 1, 2, 3, 6 and 12 months; quantity changes over 1, 3 and 12 months; fixed supplier-country shares; ERUC changes over 1, 3, 6 and 12 months; and the January 2021 EU collection-change flag.

## Selection and test boundary

Five blocked validation windows ending January 2022 select the model family by mean seven-quantile pinball loss. The selected family is frozen before the February 2022-December 2023 disruption test. Test rankings are diagnostic and do not change the selected family.

## Calibration

For each HS code, horizon and model family, out-of-sample pre-test validation residuals calculate a finite-sample split-conformal expansion. This expansion is subtracted from P05 and added to P95. It is fixed before the disruption test. Interior quantiles come directly from the fitted model. All quantiles are made non-crossing.

## Baseline

The baseline central forecast is no change. Its surrounding quantiles come from historical target changes available at the forecast date and receive the same pre-test conformal interval expansion.

## Interpretation boundary

Passing the numerical test rules supports further review; it does not establish causal validity or production readiness. ERUC is a broad non-ferrous import-price index, not a titanium benchmark. PPI revisions and assumed publication dates remain limitations.
