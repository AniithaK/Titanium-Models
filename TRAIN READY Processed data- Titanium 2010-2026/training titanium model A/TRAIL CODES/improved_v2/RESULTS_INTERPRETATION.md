# Improved V2 Results

## Outcome

The improved experiment does not increase the overall acceptance count. One of four required H3/H6 combinations passes every frozen disruption-test rule.

| HS code | Horizon | Validation-selected candidate | Coverage | MIS ratio to fixed baseline | Recovery by Aug 2022 | Result |
|---|---:|---|---:|---:|---|---|
| 810820 | H3 | adaptive no-change | 100.0% | 1.063 | Yes | Fail: interval score is worse than baseline |
| 810820 | H6 | adaptive no-change | 100.0% | 1.069 | Yes | Fail: interval score is worse than baseline |
| 810890 | H3 | adaptive no-change | 95.7% | 0.993 | Yes | Pass |
| 810890 | H6 | adaptive no-change | 95.7% | 1.061 | No | Fail: interval score and recovery |

H12 remains report-only.

## What changed

- HS 810890 H3 improved from 87.0% to 95.7% coverage and now passes.
- HS 810820 H3 no longer passes because pre-test validation selected the conservative adaptive fallback instead of the original linear quantile model.
- The total therefore remains 1/4 rather than improving to 2/4 or more.
- Several non-selected ML candidates pass individual frozen-test combinations. They cannot be substituted after inspecting the test without creating test leakage.

## Decision

Do not promote Improved V2 as a replacement for the existing candidate. Keep it as diagnostic evidence. The next valid iteration needs refreshed external predictors and a new untouched holdout period. Repeatedly tuning against February 2022 to December 2023 would turn that period into validation data.

## Remaining data work

The project contains historical World Bank and USGS workbooks, but the World Bank local copy ends in 2023 and the USGS titanium series ends in 2020. They must be refreshed and availability dates must be preserved before they can support a current production forecast. ERUC remains a broad non-ferrous import-price proxy rather than a titanium-specific benchmark.
