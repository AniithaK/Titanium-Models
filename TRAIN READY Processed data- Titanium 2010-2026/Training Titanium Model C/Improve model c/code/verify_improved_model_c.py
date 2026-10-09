"""Independently check metrics, past-error calibration and saved model reloads."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('improved_c_verify',HERE/'train_improved_model_c.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
parser=argparse.ArgumentParser()
parser.add_argument('--run-dir',type=Path)
args=parser.parse_args()
if args.run_dir:
    m.RUN=args.run_dir.resolve()
    original_scales=m.scales
    def rms_scales(bundle,origins):
        values=original_scales(bundle,origins);h=bundle['horizon_months']
        values['rms_6m']=np.sqrt(values['volatility_6m']**2+h*(origins.freight_log_change_6m.to_numpy()/6)**2)
        values['rms_12m']=np.sqrt(values['volatility_12m']**2+h*(origins.freight_log_change_12m.to_numpy()/12)**2)
        values['hybrid_rms']=np.maximum(values['rms_6m'],values['rms_12m'])
        return values
    m.scales=rms_scales
data=m.load_data();raw=pd.read_csv(m.RUN/'predictions/raw_out_of_sample_forecasts.csv')
predictions=pd.read_csv(m.RUN/'predictions/all_predictions.csv')
matched=pd.read_csv(m.RUN/'predictions/matched_baseline_predictions.csv')
metrics=pd.read_csv(m.RUN/'results/evaluation_metrics.csv')
manifest=json.loads((m.RUN/'results/run_manifest.json').read_text())
assert hashlib.sha256(m.SOURCE.read_bytes()).hexdigest()==manifest['source_sha256']
assert not raw.duplicated(['model','horizon_months','origin_month']).any()
assert not predictions.duplicated(['model','horizon_months','origin_month']).any()
assert (pd.to_datetime(raw.max_training_target_available_date)<=pd.to_datetime(raw.model_fit_asof)).all()
assert (pd.to_datetime(raw.model_fit_asof)<=pd.to_datetime(raw.forecast_date_assumed)).all()
assert (pd.to_datetime(raw.model_fit_asof)<pd.to_datetime(raw.target_available_date_assumed)).all()
assert (pd.to_datetime(predictions.calibration_latest_target_available_date)<=pd.to_datetime(predictions.forecast_date_assumed)).all()
dates=data.set_index(data.date.dt.strftime('%Y-%m'))
verified=[]

def independent_quantiles(errors):
    errors=np.sort(np.asarray(errors));n=len(errors);values=[]
    for tau in m.QUANTILES:
        if tau==.5:values.append(float(np.median(errors)))
        else:
            rank=math.floor((n+1)*tau) if tau<.5 else math.ceil((n+1)*tau)
            values.append(float(errors[np.clip(rank,1,n)-1]))
    return np.asarray(values)

for (name,h),group in predictions.groupby(['model','horizon_months']):
    history=raw[(raw.model==name)&(raw.horizon_months==h)].sort_values('origin_month')
    observed=dates.loc[group.origin_month]
    assert np.allclose(group.actual_log_change,observed[f'target_freight_log_change_h{h}'])
    assert np.allclose(group.actual_freight_index,observed[f'target_freight_index_h{h}'])
    assert np.array_equal(group.target_month.to_numpy(),observed[f'target_month_h{h}'].dt.strftime('%Y-%m').to_numpy())
    assert group[group.period=='disruption_test'].shape[0]==24
    for _,record in group.iterrows():
        current=history[history.origin_month==record.origin_month].iloc[0]
        past=history[(history.origin_month<record.origin_month)&history.actual_log_change.notna()
                     &(history.target_available_date_assumed<=record.forecast_date_assumed)].tail(int(record.calibration_window))
        assert len(past)==record.calibration_rows
        residual=(past.actual_log_change.to_numpy()-past.raw_point_log_change.to_numpy())/past[f'scale_{record.scale_mode}'].to_numpy()
        q=current.raw_point_log_change+current[f'scale_{record.scale_mode}']*independent_quantiles(residual)
        if name=='no_change':q[:3]=np.minimum(q[:3],0);q[3]=0;q[4:]=np.maximum(q[4:],0)
        stored=np.array([record[f'{p}_log_change'] for p in m.QNAMES])
        assert np.allclose(q,stored,atol=1e-12)
        assert (np.diff(stored)>=-1e-12).all()
        assert np.allclose(record.origin_freight_index*np.exp(stored),[record[f'{p}_freight_index'] for p in m.QNAMES])
        if record.period=='validation':assert record.target_available_date_assumed<'2020-01-01'
    for period,part in group.groupby('period'):
        q=part[[f'{p}_log_change' for p in m.QNAMES]].to_numpy();y=part.actual_log_change.to_numpy()
        mis=(q[:,-1]-q[:,0])+20*(q[:,0]-y)*(y<q[:,0])+20*(y-q[:,-1])*(y>q[:,-1])
        coverage=np.mean((y>=q[:,0])&(y<=q[:,-1]))
        saved=metrics[(metrics.period==period)&metrics.model.eq(name)&metrics.horizon_months.eq(h)].iloc[0]
        assert abs(mis.mean()-saved.mean_interval_score)<1e-12 and abs(coverage-saved.coverage_90)<1e-12
        base=matched[(matched.period==period)&matched.compared_model.eq(name)&matched.horizon_months.eq(h)]
        assert set(base.scale_mode)==set(part.scale_mode) and set(base.calibration_window)==set(part.calibration_window)
        assert abs(m.metrics_for(base)['mean_interval_score']-saved.matched_baseline_mis)<1e-12
    verified.append(dict(model=name,horizon_months=int(h),records=len(group),past_only_calibration_verified=True,
                         quantile_order_verified=True,target_alignment_verified=True,metrics_reproduced=True))

future=pd.read_csv(m.RUN/'predictions/latest_forecasts.csv');origin=data.iloc[-1]
for h in m.HORIZONS:
    for name in m.MODELS:
        bundle=joblib.load(m.RUN/f'models/{name}_h{h}.joblib')
        centre=float(m.point_predict(bundle,origin[m.FEATURES].to_numpy(float)[None,:])[0])
        mode=bundle['calibration_configuration']['mode']
        scale=float(m.scales(bundle,data.tail(1))[mode][0])
        past=pd.DataFrame(bundle['calibration_history'])
        assert (past.origin_month<origin.date.strftime('%Y-%m')).all()
        assert (pd.to_datetime(past.target_available_date_assumed)<=origin.forecast_date_assumed).all()
        residual=(past.actual_log_change.to_numpy()-past.raw_point_log_change.to_numpy())/past[f'scale_{mode}'].to_numpy()
        q=centre+scale*independent_quantiles(residual)
        if name=='no_change':q[:3]=np.minimum(q[:3],0);q[3]=0;q[4:]=np.maximum(q[4:],0)
        saved=future[future.model.eq(name)&future.horizon_months.eq(h)].iloc[0]
        assert np.allclose(q,[saved[f'{p}_log_change'] for p in m.QNAMES],atol=1e-12)
    assert manifest['h12_selection_excluded']
report=dict(prediction_checks=verified,portable_models_reloaded=15,final_forecasts_reproduced=15,
            matched_baseline_calibration_verified=True,source_data_unchanged=True,
            limitations=manifest['limitations'])
m.save_json(m.RUN/'results/verification_report.json',report)
print(json.dumps(report,indent=2))
