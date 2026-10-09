# Training Snapshot

This folder contains the exact engineered CSV consumed by training and reusable prediction:

- `titanium_model_a_production_v3_training.csv`

The preparation script generates the canonical output in `prepared/` and then copies it here. This separation makes the training input explicit and prevents model runs from accidentally consuming diagnostics or an unrelated upstream file.
