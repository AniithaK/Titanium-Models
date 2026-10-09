"""Build a CSV dictionary covering every prepared Model C field."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'Train'
DATA = TRAIN / 'model_c_freight_training_2005_2026.csv'
OUTPUT = TRAIN / 'model_c_training_column_dictionary.csv'
manifest = json.loads((TRAIN / 'model_c_preparation_manifest.json').read_text())
features = set(manifest['feature_columns'])
definitions = {}


def define(field, category, origin, source, description, calculation, units, purpose, note=''):
    definitions[field] = dict(field_name=field,category=category,origin_type=origin,source=source,
        description=description,calculation=calculation,units_or_values=units,
        used_as_model_input='Yes' if field in features else 'No',usage_or_purpose=purpose,
        assumption_or_limitation=note)


FREIGHT = 'BLS/FRED PCU483111483111; Data/bls_deep_sea_freight_ppi_monthly_1988-06_to_2026-08.csv'
BRENT = 'World Bank Pink Sheet; Monthly Prices / Crude oil, Brent; Data/world_bank_pink_sheet_monthly_1960-01_to_2026-09.xlsx'
POLICY = 'Preparation policy: observations assumed usable at the end of the following calendar month'
RULE = 'User-agreed split and preparation code'

define('date','Month identifier','Original observation date, normalized',FREIGHT,
    'Reference month described by the row; dates are stored as the first day of the month.',
    'Read observation_date from freight CSV; join Brent by the same calendar month.',
    'YYYY-MM-DD','Join key, chronology and lag/target alignment',
    'This is the observation month, not a verified publication or forecast-issue date.')
for field,description,formula in [
    ('period','Readable year-month identifier.','Format date as YYYY-MM'),
    ('year','Calendar year of the observation.','Year component of date'),
    ('month','Calendar month number of the observation.','Month component of date'),
]:
    define(field,'Calendar metadata','Derived from date','date',description,formula,
           'YYYY-MM' if field=='period' else 'Integer','Sorting, display and split controls; not a predictor')
define('freight_index','Source observation','Original source value, renamed',FREIGHT,
    'Monthly Deep Sea Freight Transportation producer-price index.',
    'Copy PCU483111483111; rename to freight_index. No interpolation or scaling.',
    'Index points; June 1988=100; not seasonally adjusted','Basis for freight features and targets',
    'Not a GBP/USD shipping quote. Current historical download may contain later revisions.')
define('brent_usd_per_barrel','Source observation','Original source value, extracted',BRENT,
    'Monthly Brent crude-oil price; potential fuel-cost signal for freight.',
    'Extract Crude oil, Brent from Monthly Prices; match the observation month.',
    'USD per barrel','Basis for monthly Brent-change predictor',
    'No currency conversion. Downloaded historical snapshot; original release vintages are unverified.')
for lag in [1,2,3,6,12]:
    define(f'freight_log_change_{lag}m','Prediction input','Derived historical feature','freight_index',
        f'Freight-index movement over the previous {lag} months.',
        f'ln(freight_index[t] / freight_index[t-{lag}])','Natural-log ratio',
        'Model input representing historical freight momentum',
        'Uses current and earlier observations only. Exact relative change is exp(value)-1. 2004 supplies warm-up history.')
define('brent_log_change_1m','Prediction input','Derived historical feature','brent_usd_per_barrel',
    'Movement in Brent crude price since the previous month.',
    'ln(brent_usd_per_barrel[t] / brent_usd_per_barrel[t-1])','Natural-log ratio',
    'Model input representing recent oil/fuel-price movement',
    'Oil can be associated with freight costs, but this feature alone does not prove a causal effect.')
for w in [6,12]:
    define(f'freight_volatility_{w}m','Prediction input','Derived rolling feature','freight_log_change_1m',
        f'Variability of monthly freight changes over the latest {w} observations, including the current month.',
        f'Population standard deviation of {w} monthly log changes, ddof=0; full window required.',
        'Monthly log-return standard deviation','Model input representing uncertainty in recent freight movements',
        'Not annualized. Measures return variability rather than dispersion of raw index levels.')

for field,description,formula in [
    ('freight_available_date_assumed','Assumed date on which this month\'s freight observation becomes usable.','End of calendar month following date'),
    ('brent_available_date_assumed','Assumed date on which this month\'s Brent observation becomes usable.','End of calendar month following date'),
    ('feature_available_date_assumed','Assumed date when both source observations are available.','Maximum of freight_available_date_assumed and brent_available_date_assumed'),
    ('forecast_date_assumed','Prototype issue date when the row\'s input data is treated as usable.','Copy feature_available_date_assumed'),
]:
    define(field,'Timing control','Assumed or derived assumption',POLICY,description,formula,
        'YYYY-MM-DD','Historical row/target availability control',
        'Explicit prototype convention, not a verified historical release date. Actual publication delays may differ.')
for field,source in [('freight_release_date_verified',FREIGHT),('brent_release_date_verified',BRENT)]:
    define(field,'Unresolved release metadata','Unfilled verification field',source,
        'Place to store the verified historical publication date of this observation.',
        'Left blank because historical release dates were not supplied.','Date or blank',
        'Would support verified historical information timing',
        'Blank means unverified; it must not be interpreted as zero delay or already published.')
for field,description in [
    ('historical_release_dates_verified','Whether actual historical release dates have been confirmed for this row.'),
    ('historical_data_vintages_verified','Whether the values match the versions available at the historical forecast date.'),
]:
    define(field,'Verification metadata','Assigned verification status','Source metadata audit',description,
        'False for every row in this dataset.','True/False','Distinguishes a research snapshot from verified historical inputs',
        'False is an explicit missing-evidence status, not a prediction that the source observations are numerically wrong.')
define('availability_policy','Method metadata','Recorded assumption',POLICY,
    'Text describing the assumed publication timing.',
    'Record reference-month end plus one full calendar month.','Text','Makes prototype timing assumptions visible')
define('source_value_basis','Method metadata','Recorded source-status description','Downloaded source snapshots',
    'States that original historical data revisions have not been reconstructed.',
    'Record downloaded historical snapshot; original historical vintages not verified.',
    'Text','Explains the historical revision limitation')

for h in [3,6,12]:
    define(f'target_month_h{h}','Future target metadata','Derived calendar field','date',
        f'Future reference month whose freight movement is predicted at H{h}.',
        f'date + {h} calendar months','YYYY-MM-DD','Align actual and predicted outcomes by horizon',
        'Kept even when the future actual is unknown. Horizon starts from observation month t, not the later assumed issue date.')
    define(f'target_freight_index_h{h}','Future actual outcome','Extracted future source value','freight_index',
        f'Observed freight-index level {h} months after the origin reference month.',
        f'Lookup freight_index[t+{h}]','Index points','Ground-truth value used to construct the learning target',
        f'Never a predictor input. Last {h} rows are blank because outcomes are unobserved.')
    define(f'target_freight_log_change_h{h}','Supervised learning target','Derived from future outcome','freight_index and target_freight_index',
        f'Actual freight log movement from t to t+{h}; this is the target learned by quantile models.',
        f'ln(target_freight_index_h{h} / freight_index)','Natural-log ratio',
        'Training target for P05/P10/P25/P50/P75/P90/P95 forecasts',
        'Future outcome. Must not enter the input feature matrix; remains blank if unobserved.')
    define(f'target_freight_pct_change_h{h}','Outcome for interpretation','Derived from future target',f'target_freight_log_change_h{h}',
        f'Actual relative freight-index change over {h} months, expressed in percent.',
        f'100 * (exp(target_freight_log_change_h{h}) - 1)','Percent; 10 means 10%',
        'Human-readable outcome and later forecast-comparison tables',
        'Not an input feature. The primary supervised target is the log change.')
    define(f'target_available_date_assumed_h{h}','Future timing control','Derived from assumed policy',f'target_month_h{h} and availability policy',
        'Scheduled prototype date when the future target would become available.',
        f'End of calendar month following target_month_h{h}','YYYY-MM-DD',
        'Prevents using a target before its assumed release date',
        f'An assumed schedule, not proof that a target exists. Also require target_observed_h{h}.')
    define(f'target_observed_h{h}','Outcome availability flag','Derived status',f'target_freight_index_h{h}',
        'Whether the future actual freight observation exists in the downloaded data.',
        f'target_freight_index_h{h} is nonmissing','True/False','Distinguishes completed labels from future forecast rows',
        'Observed in the current snapshot does not establish availability at an earlier forecast date.')

define('origin_period_group','Split control','Assigned from agreed rules',RULE,
    'Initial training, validation, disruption test or post-disruption group by reference month.',
    '2005-2014 training; 2015-2019 validation; 2020-2021 disruption; 2022 onward post-disruption.',
    'Period category','Main agreed split',
    'Reference month differs from actual issue month. This column alone does not enforce target timing.')
define('forecast_issue_period_group_assumed','Split control','Derived from assumed forecast date',RULE+'; forecast_date_assumed',
    'Same period groups applied to the month when the forecast is assumed issued.',
    'Apply period boundaries to the calendar month of forecast_date_assumed.',
    'Period category','Supports evaluation by issue month rather than observation month',
    'Timing is assumed. Boundary rows can belong to a different group from origin_period_group.')
define('optional_early_backtest_origin','Split control','Assigned calendar flag',RULE,
    'Marks 2012-2014 origins for an optional early chronological backtest.',
    'date between January 2012 and December 2014 inclusive.','True/False',
    'Additional diagnostic addressing the specification\'s 2012-2019 selection period',
    'These rows remain initial-training history for the main 2015-2019 validation.')
define('feature_complete','Training control','Derived completeness check','Eight model feature columns',
    'Whether all eight required predictor values are present.',
    'All eight features are nonmissing.','True/False','Screens rows for numerical feature completeness',
    'Does not prove verified publication timing or the existence of a future target.')
define('h12_reporting_only','Horizon policy','Assigned specification rule','Model C specification',
    'States that H12 outputs are reported but do not determine the best model.',
    'True for every row; the rule applies to H12.','True/False','Keeps H3/H6 as acceptance and selection horizons')
define('model_selection_freeze_date','Split control','Assigned cutoff',RULE,
    'Date before which selection input observations and labels must be available.',
    'Fixed 2020-01-01.','YYYY-MM-DD','Prevents calibration/model tuning on the COVID test outcomes',
    'Availability comparisons use strictly earlier than this cutoff.')
define('initial_validation_asof_date_assumed','Split timing control','Derived from assumed dates','forecast_date_assumed of validation rows',
    'Assumed issue date for the first main-validation origin.',
    'Minimum forecast_date_assumed among origin_period_group=validation.','YYYY-MM-DD',
    'Limits which initial-training targets may enter the first validation fit',
    'Under the current policy this is 2015-02-28, for the January 2015 reference month.')

for h in [3,6,12]:
    define(f'research_training_eligible_h{h}','Training control','Derived eligibility flag','feature_complete and target_observed',
        'Whether complete features and an observed horizon target exist.',
        f'feature_complete AND target_observed_h{h}','True/False','General research row eligibility',
        'Not sufficient alone: training code must also restrict origins and target availability for each forecast.')
    define(f'initial_training_fit_eligible_h{h}','Training control','Derived eligibility flag','Research eligibility; split; assumed dates',
        f'Whether the row can enter the first validation fit at H{h}.',
        f'Research eligible AND initial_training group AND target_available_date_assumed_h{h} <= initial_validation_asof_date_assumed.',
        'True/False','Initial supervised training selection','Depends on the recorded prototype timing assumptions.')
    define(f'validation_selection_eligible_h{h}','Validation control','Derived eligibility flag','Research eligibility; validation split; freeze date',
        f'Whether an H{h} validation row and its outcome can be used for pre-test model selection.',
        f'Research eligible AND validation group AND forecast_date_assumed < 2020-01-01 AND target_available_date_assumed_h{h} < 2020-01-01.',
        'True/False','Excludes late validation origins with outcomes unavailable before the disruption test',
        'Both timing and origin restrictions apply. Late-2019 rows are retained but some are ineligible.')
    define(f'origin_period_test_eligible_h{h}','Test control','Derived eligibility flag','Research eligibility and origin_period_group',
        'Whether an observed target exists for a 2020-2021 reference-month test row.',
        'Research eligible AND origin_period_group=disruption_test.','True/False',
        'Reference-month disruption evaluation','This checks reference months, not actual publication/forecast issue dates.')
    define(f'forecast_issue_test_eligible_assumed_h{h}','Test control','Derived eligibility flag','Research eligibility and forecast_issue_period_group_assumed',
        'Whether an observed target exists for a forecast assumed issued during 2020-2021.',
        'Research eligible AND forecast_issue_period_group_assumed=disruption_test.','True/False',
        'Issue-month disruption evaluation','Requires explicitly choosing the issue-month interpretation; dates remain assumed.')
    define(f'strict_point_in_time_verified_h{h}','Verification control','Derived evidence flag','Research eligibility and historical verification flags',
        'Whether inputs and outcome eligibility have verified historical dates and data versions.',
        'Research eligible AND historical_release_dates_verified AND historical_data_vintages_verified.',
        'True/False','Distinguishes numerically complete research data from verified point-in-time evidence',
        'False for all current rows because release dates and original revision vintages are not verified.')


def main():
    before = hashlib.sha256(DATA.read_bytes()).hexdigest()
    with DATA.open(encoding='utf-8-sig',newline='') as stream:
        columns = next(csv.reader(stream))
    if set(definitions) != set(columns):
        raise ValueError(f'Dictionary mismatch: missing={set(columns)-set(definitions)}, extra={set(definitions)-set(columns)}')
    rows = [dict(column_number=i,**definitions[name]) for i,name in enumerate(columns,1)]
    headers = ['column_number','field_name','category','origin_type','source','description',
               'calculation','units_or_values','used_as_model_input','usage_or_purpose','assumption_or_limitation']
    with OUTPUT.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with OUTPUT.open(encoding='utf-8-sig',newline='') as stream:
        exported = list(csv.DictReader(stream))
    assert [r['field_name'] for r in exported] == columns
    assert {r['field_name'] for r in exported if r['used_as_model_input']=='Yes'} == features
    assert all(r['source'] and r['description'] and r['calculation'] and r['origin_type'] for r in exported)
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == before
    print(f'Created: {OUTPUT}')
    print(f'Documented {len(exported)} fields; marked {len(features)} model inputs; training data unchanged.')


if __name__ == '__main__':
    main()
