# Model A Titanium Trade-Data Preparation

## Deliverable

- Dataset: `titanium_model_a_trade_core_2010_2026.csv`
- Shape: 396 rows and 71 columns
- Coverage: 2010-01-01 through 2026-06-01
- Grain: one row per calendar month and HS code
- Products: HS 810820 and HS 810890 are kept separate. HS 810830 scrap is excluded.
- Source: 17 annual UN Comtrade CSV exports for UK monthly imports.

This is the consolidated **trade-core** dataset for Model A. It contains every feature and direct target that can be derived from the supplied UN Comtrade files. It is not yet the final specification-complete training matrix because the source folder does not contain monthly USD-to-GBP exchange rates, World Bank nickel/aluminium prices or the lagged USGS titanium sponge series.

## Processing Steps

1. Loaded every CSV using the UN Comtrade Latin-1 encoding.
2. Retained UK monthly imports for HS 810820 and HS 810890 only.
3. Verified reporter, flow, second partner, customs procedure and transport mode settings.
4. Confirmed one World row for every HS code and month from January 2010 through June 2026: 396 rows in total.
5. Used the World-partner `primaryValue` as monthly import value in USD.
6. Used World net weight when positive. For 37 rows with missing or zero World weight, used the sum of individual partner-country net weights and retained an explicit reconstruction flag.
7. Reconciled World import value against country-row value. Maximum absolute difference: 0.00000007%.
8. Calculated monthly import unit value as `trade_value_usd / net_weight_kg`.
9. Created price log changes over 1, 2, 3, 6 and 12 months.
10. Created quantity log changes over 1, 3 and 12 months.
11. Defined main suppliers using the specification rule: at least 5% of combined UK titanium import weight during 2018-2020.
12. Calculated monthly weight shares for each main supplier, an `other` share, supplier HHI, top supplier and top-supplier share separately for each HS code.
13. Added calendar seasonality, the January 2021 EU collection-change indicator, source extraction date and a conservative availability date of month-end plus 42 days.
14. Created direct H3, H6 and H12 future-price labels and log-change targets. These are direct targets, not recursive H1 forecasts. Each target also has an assumed availability date so rolling training can exclude labels not yet published at the forecast date.
15. Assigned forecast-origin partitions: model selection/tuning through January 2022, disruption test from February 2022 through December 2023, and post-disruption monitoring from January 2024.
16. Added no-change baseline value `0.0` in log-change space.

## Main Suppliers

- USA: 43.54% of combined 2018-2020 titanium import weight
- Japan: 17.79% of combined 2018-2020 titanium import weight
- France: 17.00% of combined 2018-2020 titanium import weight
- Russian Federation: 7.10% of combined 2018-2020 titanium import weight
- China: 5.87% of combined 2018-2020 titanium import weight

All remaining countries are grouped into `supplier_share_other`. Main-supplier membership is fixed from 2018-2020 and is not recomputed during later test periods.

## Direct Targets

- `target_log_change_usd_per_kg_h3 = ln(price[t+3]) - ln(price[t])`
- `target_log_change_usd_per_kg_h6 = ln(price[t+6]) - ln(price[t])`
- `target_log_change_usd_per_kg_h12 = ln(price[t+12]) - ln(price[t])`

The accompanying target month and future price columns are included for auditing. Target columns must never be supplied to the model as input features.

During rolling model fitting, include a historical row only when its `target_data_available_date_assumed_h*` is no later than the forecast date being simulated. This prevents future target information from entering a training window.

The current targets are denominated in USD/kg because UN Comtrade provides USD value. The Model A specification requires GBP/kg. After monthly historical FX is joined, recalculate the unit value, lag changes and direct targets from GBP/kg rather than simply renaming these columns.

## Quality Controls

- World values and partner values reconcile exactly within floating-point precision.
- 37 monthly weights are reconstructed and identified by `weight_reconstructed_flag=1`.
- 13 HS-month rows contain at least one suspicious partner-level weight/value combination.
- Robust unit-value flags were fitted only on data available through January 2022, using log unit value and a median absolute deviation threshold of absolute robust-z greater than 5.
- Flagged national unit values are preserved, not silently corrected:

- 2020-06, HS 810820: USD 4.7185/kg
- 2011-07, HS 810890: USD 0.7178/kg
- 2014-08, HS 810890: USD 178.6929/kg
- 2015-08, HS 810890: USD 170.6090/kg
- 2018-11, HS 810890: USD 6.1136/kg
- 2021-06, HS 810890: USD 19.9701/kg

- `trade_core_eligible_h3`, `trade_core_eligible_h6` and `trade_core_eligible_h12` require complete lagged trade features, an available direct target and no flagged unit-value anomaly at the origin or target month.
- Eligibility does not mean the row is ready for the final Model A fit; external features and GBP conversion are still required.

## Eligible Trade-Core Rows

### H3
- HS 810820, disruption_test_2022_2023: 23 trade-core eligible rows
- HS 810820, model_selection_and_tuning: 131 trade-core eligible rows
- HS 810820, post_disruption_monitoring: 27 trade-core eligible rows
- HS 810890, disruption_test_2022_2023: 23 trade-core eligible rows
- HS 810890, model_selection_and_tuning: 123 trade-core eligible rows
- HS 810890, post_disruption_monitoring: 27 trade-core eligible rows
### H6
- HS 810820, disruption_test_2022_2023: 23 trade-core eligible rows
- HS 810820, model_selection_and_tuning: 131 trade-core eligible rows
- HS 810820, post_disruption_monitoring: 24 trade-core eligible rows
- HS 810890, disruption_test_2022_2023: 23 trade-core eligible rows
- HS 810890, model_selection_and_tuning: 123 trade-core eligible rows
- HS 810890, post_disruption_monitoring: 24 trade-core eligible rows
### H12
- HS 810820, disruption_test_2022_2023: 23 trade-core eligible rows
- HS 810820, model_selection_and_tuning: 131 trade-core eligible rows
- HS 810820, post_disruption_monitoring: 18 trade-core eligible rows
- HS 810890, disruption_test_2022_2023: 23 trade-core eligible rows
- HS 810890, model_selection_and_tuning: 125 trade-core eligible rows
- HS 810890, post_disruption_monitoring: 18 trade-core eligible rows

## Model Input Columns

Use these supplied feature groups:

- Price history: `price_log_change_1m`, `2m`, `3m`, `6m`, `12m`.
- Quantity history: `quantity_log_change_1m`, `3m`, `12m`.
- Supplier structure: five main-country shares, `supplier_share_other`, `supplier_hhi_weight`, and `top_supplier_share_weight`.
- Time controls: `month_sin`, `month_cos`, `eu_collection_change_flag`.
- Quality controls: reconstruction, anomaly and eligibility flags. Do not use these as price signals without an explicit modelling decision.

Identifiers, source fields, future targets, future prices and target-quality flags are audit fields, not predictive inputs.

## Required Before Final Model Training

1. Join a monthly historical USD-to-GBP exchange-rate series using only information available at each forecast date.
2. Recalculate import unit value and H3/H6/H12 targets in GBP/kg.
3. Join monthly World Bank nickel and aluminium price changes.
4. Join annual USGS titanium sponge price, making each annual observation available only from January of the following year.
5. Confirm the assumed 42-day trade-publication lag against the actual release calendar used in production.
6. Review flagged observations, especially HS 810890 in July 2011, before deciding whether to retain, correct from authoritative evidence or exclude them.
7. Fit separate models for HS 810820 and HS 810890, or use an explicitly justified pooled model with HS identity and interactions. Do not average the products together.
8. Tune only on forecast origins through January 2022. Freeze settings before the February 2022-December 2023 disruption test.

## Full Column List

- `date`
- `period`
- `year`
- `month`
- `hs_code`
- `hs_product`
- `source_product_description`
- `classification_code`
- `source_file`
- `source_extraction_date`
- `month_end_date`
- `trade_data_available_date_assumed`
- `evaluation_partition`
- `trade_value_usd`
- `world_net_weight_kg_original`
- `partner_weight_sum_kg`
- `net_weight_kg`
- `weight_source`
- `weight_reconstructed_flag`
- `world_weight_estimated_by_un_flag`
- `partner_value_sum_usd`
- `world_value_partner_difference_pct`
- `partner_count`
- `import_unit_value_usd_per_kg`
- `log_unit_value`
- `unit_value_robust_z`
- `unit_value_outlier_flag`
- `partner_anomaly_count`
- `partner_anomaly_max_weight_multiple`
- `data_quality_status`
- `price_log_change_1m`
- `price_log_change_2m`
- `price_log_change_3m`
- `price_log_change_6m`
- `price_log_change_12m`
- `quantity_log_change_1m`
- `quantity_log_change_3m`
- `quantity_log_change_12m`
- `supplier_share_usa`
- `supplier_share_japan`
- `supplier_share_france`
- `supplier_share_russia`
- `supplier_share_china`
- `supplier_share_other`
- `supplier_share_sum_check`
- `supplier_hhi_weight`
- `top_supplier_name`
- `top_supplier_share_weight`
- `month_sin`
- `month_cos`
- `eu_collection_change_flag`
- `latest_year_partial_flag`
- `no_change_baseline_log_change`
- `target_month_h3`
- `target_data_available_date_assumed_h3`
- `target_unit_value_usd_per_kg_h3`
- `target_log_change_usd_per_kg_h3`
- `target_quality_flag_h3`
- `trade_core_eligible_h3`
- `target_month_h6`
- `target_data_available_date_assumed_h6`
- `target_unit_value_usd_per_kg_h6`
- `target_log_change_usd_per_kg_h6`
- `target_quality_flag_h6`
- `trade_core_eligible_h6`
- `target_month_h12`
- `target_data_available_date_assumed_h12`
- `target_unit_value_usd_per_kg_h12`
- `target_log_change_usd_per_kg_h12`
- `target_quality_flag_h12`
- `trade_core_eligible_h12`
