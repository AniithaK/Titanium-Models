"""Independent artifact checks; no model selection or retraining."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('model_c_training', HERE / 'train_model_c.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
frame = m.load_data()
manifest = json.loads((m.RUN/'results/run_manifest.json').read_text())
assert hashlib.sha256(m.DATA_PATH.read_bytes()).hexdigest() == manifest['source_sha256']
assert manifest['h12_selection_excluded']
assert manifest['features'] == m.FEATURES and len(m.FEATURES)==8
predictions = pd.read_csv(m.RUN/'predictions/all_out_of_sample_predictions.csv')
metrics = pd.read_csv(m.RUN/'results/evaluation_metrics.csv')
assert not predictions.duplicated(['model','horizon_months','origin_month']).any()
assert predictions.forecast_produced.all()
assert not json.loads((m.RUN/'results/forecast_errors.json').read_text())
checks=[]

for (model,h),group in predictions.groupby(['model','horizon_months']):
    q=group[[f'{p}_log_change' for p in m.QNAMES]].to_numpy()
    levels=group[[f'{p}_freight_index' for p in m.QNAMES]].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-10).all()
    assert (levels>0).all() and (np.diff(levels,axis=1)>=-1e-8).all()
    assert np.allclose(levels,group.origin_freight_index.to_numpy()[:,None]*np.exp(q),atol=1e-8)
    reference=frame.set_index(frame.date.dt.strftime('%Y-%m')).loc[group.origin_month]
    assert np.allclose(group.actual_freight_index,reference[f'target_freight_index_h{h}'])
    assert np.allclose(group.actual_log_change,reference[f'target_freight_log_change_h{h}'])
    assert np.array_equal(group.target_month.to_numpy(),reference[f'target_month_h{h}'].dt.strftime('%Y-%m').to_numpy())
    assert (pd.to_datetime(group.max_training_target_available_date)<=pd.to_datetime(group.model_fit_asof)).all()
    assert (pd.to_datetime(group.model_fit_asof)<=pd.to_datetime(group.forecast_date_assumed)).all()
    assert (pd.to_datetime(group.max_training_target_available_date)<pd.to_datetime(group.target_available_date_assumed)).all()
    assert group.loc[group.period=='disruption_test'].shape[0]==24
    known=group[group.period=='validation']
    assert (pd.to_datetime(known.target_available_date_assumed)<pd.Timestamp('2020-01-01')).all()
    for period,part in group.groupby('period'):
        q=part[[f'{p}_log_change' for p in m.QNAMES]].to_numpy()
        y=part.actual_log_change.to_numpy()
        coverage=np.mean((y>=q[:,0])&(y<=q[:,-1]))
        score=(q[:,-1]-q[:,0])+20*(q[:,0]-y)*(y<q[:,0])+20*(y-q[:,-1])*(y>q[:,-1])
        residual=y[:,None]-q
        pinball=np.mean(np.maximum(m.QUANTILES*residual,(m.QUANTILES-1)*residual))
        saved=metrics[(metrics.period==period)&(metrics.model==model)&(metrics.horizon_months==h)].iloc[0]
        assert abs(saved.coverage_90-coverage)<1e-12
        assert abs(saved.mean_interval_score-score.mean())<1e-12
        assert abs(saved.pinball_loss-pinball)<1e-12
        baseline=metrics[(metrics.period==period)&(metrics.model=='no_change')&(metrics.horizon_months==h)].iloc[0].mean_interval_score
        assert abs(saved.baseline_mis-baseline)<1e-12
    checks.append(dict(model=model,horizon_months=int(h),records=len(group),quantile_order_verified=True,
                       actual_target_alignment_verified=True,metrics_reproduced=True,training_timing_verified=True))

latest=frame.iloc[-1]
future=pd.read_csv(m.RUN/'predictions/latest_forecasts_all_models.csv')
for h in m.HORIZONS:
    proper,calibration=m.partition_calibration(m.available_training(frame,latest,h),h)
    for name in m.MODELS:
        bundle=joblib.load(m.RUN/f'models/{name}/model_c_h{h}.joblib')
        assert bundle['features']==m.FEATURES
        assert bundle['proper_training_rows']==len(proper)
        assert bundle['calibration_rows']==len(calibration)==24
        assert set(proper.date).isdisjoint(set(calibration.date))
        raw=m.raw_predict(bundle,calibration[m.FEATURES].to_numpy())
        actual=calibration[f'target_freight_log_change_h{h}'].to_numpy()
        scores=np.maximum(raw[:,0]-actual,actual-raw[:,-1])
        order=math.ceil((len(scores)+1)*.9)
        expected_delta=max(0,float(np.sort(scores)[order-1]))
        assert abs(expected_delta-bundle['calibration_expansion'])<1e-12
        q,_=m.calibrated_predict(bundle,latest[m.FEATURES].to_numpy(float)[None,:])
        saved=future[(future.model==name)&(future.horizon_months==h)].iloc[0]
        assert np.allclose(q[0],[saved[f'{p}_log_change'] for p in m.QNAMES],atol=1e-12)
        if name=='no_change':assert q[0,3]==0

issue=pd.read_csv(m.RUN/'predictions/specification_issue_date_test_actual_vs_predicted.csv')
assert pd.to_datetime(issue.forecast_date_assumed).between('2020-01-01','2021-12-31').all()
assert (issue.groupby(['model','horizon_months']).size()==24).all()
curves=pd.read_csv(m.RUN/'results/train_validation_loss_curves.csv')
assert set(curves.model)==set(m.MODELS[:-1])
assert np.isfinite(curves[['train_pinball_loss','validation_pinball_loss']]).all().all()
assert (curves[['train_pinball_loss','validation_pinball_loss']]>=0).all().all()
result=dict(out_of_sample_checks=checks,portable_models_verified=15,
            final_forecasts_reproduced=len(future),calibration_scores_verified=True,
            same_calibration_method_for_baseline=True,source_data_unchanged=True,
            issue_date_test_count_verified=True,loss_curve_columns_verified=True,
            remaining_limitations=manifest['limitations'])
m.save_json(m.RUN/'results/verification_report.json',result)
print(json.dumps(result,indent=2))
