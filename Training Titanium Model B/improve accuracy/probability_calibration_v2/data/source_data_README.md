# Model B Training Data

`titanium_model_b_country_month_training_2010_2026.csv` contains one row per
month and supplier group for USA, Japan, France, Russian Federation, China and
Other countries.

The dataset contains 1,188 rows from January 2010 through June 2026. Country
weights are reconstructed from each HS row's fixed supplier share and net
weight, then aggregated across HS 810820 and 810890. The sum of country shares
is checked against one for every month.

`model_b_fixed_class_thresholds.json` records the frozen percentage-point share
change boundaries derived from pre-test H3 observations ending June 2021.

Future target columns are audit/training labels and must never be passed to the
model as input features.
