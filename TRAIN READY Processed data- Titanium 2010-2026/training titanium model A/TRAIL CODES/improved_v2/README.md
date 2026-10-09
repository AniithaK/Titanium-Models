# Titanium Model A Improved V2

## Implemented improvements

- Added FX movements and FX volatility.
- Added price and quantity volatility.
- Added import-volume level, relative-volume and low-volume indicators.
- Added supplier concentration, top-supplier share and partner-count features.
- Added seasonality and observed data-quality indicators.
- Down-weighted low-volume, estimated-weight, supplier-anomaly and historical price-shock observations during fitting.
- Added weighted Elastic Net and robust Huber candidates.
- Added volatility-adaptive predictive distributions.
- Added a validation-selectable no-change fallback.
- Selection first requires pre-test validation coverage >=90% and MIS below baseline where possible, then minimises pinball loss.

## Evaluation boundary

The February 2022 to December 2023 disruption test remains frozen and is not used for selection. H3 and H6 are acceptance horizons; H12 remains report-only.

H3/H6 combinations passing every frozen-test rule: **1/4**.

## Remaining specification gaps

World Bank aluminium/nickel changes and the annual USGS titanium sponge series are still absent. They were not fabricated. ERUC remains an experimental broad non-ferrous import-price proxy, so this is not literal full-spec compliance.
