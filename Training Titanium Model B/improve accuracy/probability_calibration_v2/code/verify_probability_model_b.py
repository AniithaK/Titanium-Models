"""Check saved outputs independently, including model reload and chronology."""
from pathlib import Path
import hashlib
import json
import importlib.util
import sys
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('probability_model_b', HERE / 'train_probability_model_b.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
RUN = module.RUN
data, thresholds = module.load_data()
predictions = pd.read_csv(RUN / 'results/all_ground_truth_predictions.csv')
metrics = pd.read_csv(RUN / 'results/evaluation_metrics.csv')
audit = json.loads((RUN / 'results/source_audit.json').read_text())
assert audit['sha256'] == hashlib.sha256(module.DATA_PATH.read_bytes()).hexdigest()
assert not predictions.duplicated(['period', 'horizon_months', 'forecast_origin', 'country_key']).any()

checked = []
for (period,h),frame in predictions.groupby(['period','horizon_months']):
    probabilities = frame[[f'probability_{c}' for c in module.CLASSES]].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    actual = np.eye(4)[[module.CLASSES.index(c) for c in frame.actual_class]]
    brier = ((probabilities-actual)**2).sum(axis=1).mean()
    observed = metrics[(metrics.period==period) & (metrics.horizon_months==h)].iloc[0]
    assert abs(brier-observed.brier_score)<1e-12
    for field,stored in [('historical_probability_',observed.historical_frequency_brier),('last_change_probability_',observed.persistence_brier)]:
        values=frame[[field+c for c in module.CLASSES]].to_numpy()
        assert abs(((values-actual)**2).sum(axis=1).mean()-stored)<1e-12
    assert np.array_equal(np.array(module.CLASSES)[probabilities.argmax(1)],frame.predicted_class)
    origin = pd.to_datetime(frame.forecast_origin)
    assert np.array_equal((origin+pd.DateOffset(months=int(h))).to_numpy(),pd.to_datetime(frame.target_month).to_numpy())
    left=frame[['forecast_origin','country_key']].copy()
    left['forecast_origin']=pd.to_datetime(left.forecast_origin)
    source=data[['date','country_key','origin_data_available_date_assumed',f'target_class_h{h}']]
    merged=left.merge(source,left_on=['forecast_origin','country_key'],right_on=['date','country_key'],validate='one_to_one')
    assert (pd.to_datetime(frame.max_training_target_available_date).to_numpy()<=merged.origin_data_available_date_assumed.to_numpy()).all()
    assert np.array_equal(frame.actual_class.to_numpy(),merged[f'target_class_h{h}'].to_numpy())
    if period=='validation':assert pd.to_datetime(frame.target_month).max()<=pd.Timestamp('2021-06-01')
    checked.append(dict(period=period,horizon_months=int(h),rows=len(frame),metrics_reproduced=True,chronology_verified=True))

latest=data[data.date==data.date.max()]
reloaded=pd.concat([module.predict_bundle(joblib.load(RUN/f'models/model_b_h{h}_calibrated.joblib'),latest) for h in module.HORIZONS],ignore_index=True)
saved=pd.read_csv(RUN/'results/latest_forecasts.csv')
for field in ['country_key','target_month','predicted_class']:
    assert np.array_equal(saved[field].to_numpy(),reloaded[field].to_numpy())
for label in module.CLASSES:
    assert np.allclose(saved[f'probability_{label}'],reloaded[f'probability_{label}'],atol=1e-12)

# The structural rule follows mathematical bounds; it must never exclude an observed label.
choice=dict(temperature=1.0,frequency_weight=0.0,feasible_classes=True)
for h in module.HORIZONS:
    valid=data[data[f'target_class_h{h}'].notna()]
    probs=module.adjusted_probabilities(np.full((len(valid),4),0.25),np.full((len(valid),4),0.25),valid.country_share,choice,thresholds)
    actual_index=np.array([module.CLASSES.index(c) for c in valid[f'target_class_h{h}']])
    assert (probs[np.arange(len(valid)),actual_index]>0).all()
small=module.adjusted_probabilities(np.full((1,4),0.25),np.full((1,4),0.25),[0.08],choice,thresholds)
assert small[0,0]==0

report=dict(checks=checked,probability_sum_verified=True,model_reload_reproduces_saved_forecasts=True,
            future_target_alignment_verified=True,feasibility_never_excludes_observed_class=True,
            source_hash_unchanged=True,forecast_rows=len(saved))
module.dump_json(RUN/'results/verification_report.json',report)
print(json.dumps(report,indent=2))
