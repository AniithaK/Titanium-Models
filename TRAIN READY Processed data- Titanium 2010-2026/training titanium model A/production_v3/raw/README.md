# Raw Input Snapshots

This folder contains immutable upstream inputs required to rebuild Production V3.

- `upstream_improved_v2_training_2010_2026.csv`: trade, price, quantity, supplier, FX, ERUC, seasonality and quality features from the Improved V2 preparation pipeline.
- `upstream_external_features_2010_2026.csv`: World Bank aluminium/nickel and USGS titanium-sponge feature snapshot used by the last successful V3 preparation run. `world_bank_input_observed_flag` distinguishes genuinely observed months through December 2023 from neutral zero placeholders after the local source ends.

The external-feature snapshot was recovered from the existing V3 prepared matrix because the former upstream `Model A/data/titanium_model_a_training_matrix.csv` file is no longer present. Source-age and stale-data fields remain explicit in the rebuilt output.

Do not edit these CSV files manually. Replace them only through a documented source-refresh process.
