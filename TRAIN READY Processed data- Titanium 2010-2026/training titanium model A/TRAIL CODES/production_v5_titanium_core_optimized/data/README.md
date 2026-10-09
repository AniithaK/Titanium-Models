# Training Data

This folder contains the unchanged 396-row, 134-column titanium-core snapshot
used by V5. It covers January 2010 through June 2026 for HS 810820 and HS
810890 and excludes all USGS and World Bank features.

V5 changes model selection, not the training observations. Candidate ranking
uses only blocked validation ending January 2022 and minimizes calibrated mean
interval score among models meeting the validation coverage and baseline gates.
