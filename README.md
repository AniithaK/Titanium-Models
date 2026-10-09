# Titanium Models

Research code, source datasets, prepared training data, fitted models, predictions, evaluation records and visualizations for the titanium SCI studies.

The local project's directory structure is retained. Earlier experiments and reports are included alongside the later studies. Generated Python caches and Excel temporary lock files are excluded.

## Contents

| Folder / file | Contents |
| --- | --- |
| `Titanium UK monthly imports data 2010-2026/` | UN Comtrade titanium import downloads, Bank of England exchange rates, inflation data and source-data analysis |
| `TRAIN READY Processed data- Titanium 2010-2026/` | Prepared data, Model A/B/C code and models, performance reviews, prediction records and visualizations |
| `SCI_Model_Specifications_October_2026.md` | Model tasks, requested data, forecast horizons and acceptance requirements |

## Studies

- Model A predicts distributions of titanium import-price changes. The selected study folder is `training titanium model A/production_v6_titanium_core_corrected/` under the prepared-data folder; earlier versions are also retained.
- Model B predicts country-level titanium supplier-share movement probabilities. Its code, data, models and reports are under `Training Titanium Model B/`, including the `improve accuracy/` experiments.
- Model C predicts freight-index movement distributions using the BLS freight PPI and World Bank Brent prices. Its source files, prepared data, original training run and improvement studies are under `Training Titanium Model C/`.
- `Titanium Model A and B Performance Review/` contains combined reports and actual-versus-predicted analyses.

Follow each study's README and dependency file for its entry points. Some earlier scripts contain machine-specific paths and need configuration when used on another machine. Standalone HTML dashboards can be opened locally in a browser; repository pages do not execute Python models.

## Interpretation

These are research artifacts. Folder names such as `production` do not independently establish readiness or compliance. Consult the saved evaluation results and documented limitations: several historical publication dates are assumed, revision vintages are not fully reconstructed, and some models or horizons fail the stated acceptance requirements. The Model C improvement studies did not produce a consistently superior production replacement.

Datasets retain their original sources and metadata. Freight-index levels are index points rather than route-specific shipping quotations. Model outputs and performance measures should be interpreted using the definitions in the corresponding study documentation.
