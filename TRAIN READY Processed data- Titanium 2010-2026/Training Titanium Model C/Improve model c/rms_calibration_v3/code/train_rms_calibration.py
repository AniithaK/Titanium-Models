"""Extend V2 calibration with observed return magnitudes; reuse its raw forecasts."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import sys

import numpy as np
import pandas as pd

RUN=Path(__file__).resolve().parents[1]
PARENT=RUN.parent
spec=importlib.util.spec_from_file_location('model_c_rms_experiment',PARENT/'code/train_improved_model_c.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.RUN=RUN
m.SCALE_MODES+=['rms_6m','rms_12m','hybrid_rms']
original_scales=m.scales


def rms_scales(bundle,origins):
    values=original_scales(bundle,origins)
    h=bundle['horizon_months']
    # RMS^2 = variance + squared mean return; both come from existing historical features.
    values['rms_6m']=np.sqrt(values['volatility_6m']**2+h*(origins.freight_log_change_6m.to_numpy()/6)**2)
    values['rms_12m']=np.sqrt(values['volatility_12m']**2+h*(origins.freight_log_change_12m.to_numpy()/12)**2)
    values['hybrid_rms']=np.maximum(values['rms_6m'],values['rms_12m'])
    return values


m.scales=rms_scales


def main():
    for name in ['models','results','predictions','visualizations']:(RUN/name).mkdir(parents=True,exist_ok=True)
    if (RUN/'results/run_manifest.json').exists():raise FileExistsError('RMS experiment already completed')
    data=m.load_data()
    raw=pd.read_csv(PARENT/'predictions/raw_out_of_sample_forecasts.csv')
    features=data.set_index(data.date.dt.strftime('%Y-%m'))
    by_origin=features.loc[raw.origin_month]
    h=raw.horizon_months.to_numpy()
    raw['scale_rms_6m']=np.sqrt(raw.scale_volatility_6m.to_numpy()**2+h*(by_origin.freight_log_change_6m.to_numpy()/6)**2)
    raw['scale_rms_12m']=np.sqrt(raw.scale_volatility_12m.to_numpy()**2+h*(by_origin.freight_log_change_12m.to_numpy()/12)**2)
    raw['scale_hybrid_rms']=np.maximum(raw.scale_rms_6m,raw.scale_rms_12m)
    raw.to_csv(RUN/'predictions/raw_out_of_sample_forecasts.csv',index=False)
    configs,selections=m.select_validation(raw,data)
    records,matched,metrics=m.evaluate(raw,data,configs)
    records.to_csv(RUN/'predictions/all_predictions.csv',index=False)
    matched.to_csv(RUN/'predictions/matched_baseline_predictions.csv',index=False)
    for period,group in records.groupby('period'):group.to_csv(RUN/f'predictions/{period}_actual_vs_predicted.csv',index=False)
    recovery,summary=m.original_module().recovery_assessment(records)
    recovery.to_csv(RUN/'results/recovery_trajectory.csv',index=False);summary.to_csv(RUN/'results/recovery_summary.csv',index=False)
    metrics=metrics.merge(summary[['model','horizon_months','recovery_within_six_months']],on=['model','horizon_months'])
    metrics['full_disruption_gate_pass']=metrics.probability_gates_pass & metrics.recovery_within_six_months
    metrics.to_csv(RUN/'results/evaluation_metrics.csv',index=False)
    issue=records[pd.to_datetime(records.forecast_date_assumed).between('2020-01-01','2021-12-31')].copy()
    issue['period']='specification_issue_date_test';issue.to_csv(RUN/'predictions/specification_issue_date_test.csv',index=False)
    comparison=m.compare_original(metrics,selections);comparison.to_csv(RUN/'results/previous_vs_improved_selected.csv',index=False)
    forecasts,influence=m.save_models(data,raw,configs,selections)
    forecasts.to_csv(RUN/'predictions/latest_forecasts.csv',index=False);influence.to_csv(RUN/'results/feature_influence.csv',index=False)
    m.save_json(RUN/'results/model_c_response.json',dict(model='Freight movement',version='rms_calibration_v3',
        status='Research experiment with reused test periods and assumed publication dates',h12_reporting_only=True,
        quantile_levels=m.QUANTILES.tolist(),forecasts=forecasts[forecasts.selected_for_horizon].to_dict('records')))
    m.save_json(RUN/'results/run_manifest.json',dict(source_csv=str(m.SOURCE),source_sha256=hashlib.sha256(m.SOURCE.read_bytes()).hexdigest(),
        features=m.FEATURES,models=m.LABELS,seed=m.SEED,training_window_months=m.TRAINING_WINDOW,
        recency_half_life_months=m.RECENCY_HALF_LIFE,calibration_windows=m.WINDOWS,calibration_scale_options=m.SCALE_MODES,
        selections=selections,raw_forecast_source=str(PARENT/'predictions/raw_out_of_sample_forecasts.csv'),
        raw_forecasts_unchanged=True,
        rms_definition='sqrt(floored historical return variance + squared historical mean monthly return) * sqrt(h); mean from 6/12-month existing log change divided by window',
        additional_predictor_columns=False,h12_selection_excluded=True,
        development_note='RMS candidate added after seeing V2 failures. Its selection uses pre-2020 validation only; reused test results are development evidence, not an untouched final holdout.',
        limitations=['Availability dates assumed','Historical revision vintages unverified','FBX comparison absent','Test periods previously examined','Time-series nominal coverage not guaranteed']))
    m.render()
    print(comparison.to_string(index=False),flush=True)
    print(metrics[metrics.period.eq('disruption_test')&metrics.horizon_months.isin([3,6])][['model','horizon_months','coverage_90','mean_interval_score','mis_ratio_vs_matched_baseline','mape_freight_index','full_disruption_gate_pass']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    m.render() if args.render_only else main()
