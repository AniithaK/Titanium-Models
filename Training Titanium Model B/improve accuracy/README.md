# Model B Selected Package

The selected Model B package is **probability_calibration_v2**, not the earlier accuracy-only experiment files in this parent folder.

Read the [complete Model B README](probability_calibration_v2/README.md) for data sources, preprocessing, exact feature groups, train/validation/test dates, algorithms, baseline comparisons, performance tables, prediction examples, graphs and limitations.

| Location | Purpose |
| --- | --- |
| [probability_calibration_v2/data](probability_calibration_v2/data/) | Package-local 2010-June 2026 country-month panel, fixed thresholds and provenance inputs |
| [probability_calibration_v2/code](probability_calibration_v2/code/) | Selected training/inference and local verification code |
| [probability_calibration_v2/models](probability_calibration_v2/models/) | Selected H3 forest ensemble and H6 Extra Trees bundles |
| [probability_calibration_v2/results](probability_calibration_v2/results/) | Validation, disruption/post-disruption actuals/predictions, metrics and latest outputs |
| [probability_calibration_v2/visualizations](probability_calibration_v2/visualizations/) | Comparison, confusion/reliability charts and HTML dashboard |

Calibration fails some historical test groups; this is a research/demo package, not production-approved. H12 is not supported by the selected bundle. Earlier scripts/results, if retained here, are provenance rather than the recommended entry point. The selected package does not need them for local inference.
