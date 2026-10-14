# Package-Local Model B Data

`titanium_model_b_country_month_training_2010_2026.csv` is the original prepared country-month training panel. `model_b_fixed_class_thresholds.json` contains the fixed class boundaries. Both are unchanged copies.

The loader derives the expanded features in memory using the local `code/model_b_feature_engine.py`. No targets, thresholds or model parameters are changed. The calibrated training/inference entry point reads this directory and the local feature-engine file rather than the parent Model B folders.

The original model/selection manifests, previous experiment manifest and evaluation CSVs support feature selection definitions and historical comparison rendering. They are provenance inputs, not additional training observations. The upstream Model A enriched CSV is included solely for supplier-weight reconstruction reconciliation, not as a new model feature source. Existing assumptions and specification gaps are unchanged.

Run `python code/verify_local_data.py` from `probability_calibration_v2` to verify the source hash, fixed thresholds and reproduction of saved probabilities. The initial top-level copy manifest predates these additional packaging files and code-path changes.
