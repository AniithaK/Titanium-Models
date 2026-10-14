from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
DATA_DIR = RUN / 'data'
DATA_PATH = DATA_DIR / 'titanium_model_b_country_month_training_2010_2026.csv'
CLASSES = ['collapse', 'fall', 'stable', 'rise']
HORIZONS = [3, 6]
BLOCKS = [('2018-01-01', '2018-12-01'), ('2019-01-01', '2019-12-01'), ('2020-01-01', '2021-06-01')]
SEED = 42
CORE = json.loads((DATA_DIR / 'original_model_manifest.json').read_text())['features']
EXPANDED = json.loads((DATA_DIR / 'previous_experiment_manifest.json').read_text())['features']
HISTORY = [c for c in EXPANDED if c.startswith(('country_', 'share_', 'quantity_')) or c in ['log1p_country_quantity_kg', 'last_observed_3m_class', 'month_sin', 'month_cos', 'supplier_hhi_weight', 'top_supplier_share_weight']]
FEATURE_SETS = {'core': CORE, 'expanded': EXPANDED, 'history': HISTORY}
CANDIDATES = {
    'original_rf_core': dict(family='rf', features='core', n_estimators=140, max_depth=8, min_samples_leaf=4, max_features='sqrt', class_weight='balanced_subsample'),
    'original_extra_core': dict(family='extra', features='core', n_estimators=160, max_depth=10, min_samples_leaf=3, max_features='sqrt', class_weight='balanced'),
    'logistic_core': dict(family='logistic', features='core', C=0.1),
    'extra_history': dict(family='extra', features='history', n_estimators=160, max_depth=9, min_samples_leaf=6, max_features=0.75, class_weight=None),
    'rf_expanded': dict(family='rf', features='expanded', n_estimators=160, max_depth=8, min_samples_leaf=6, max_features=0.6, class_weight=None),
    'extra_expanded': dict(family='extra', features='expanded', n_estimators=180, max_depth=10, min_samples_leaf=6, max_features=0.7, class_weight=None),
    'hist_core': dict(family='hist', features='core', max_iter=130, max_leaf_nodes=7, min_samples_leaf=20, learning_rate=0.04, l2_regularization=3.0),
}
ENSEMBLES = {
    'original_forest_ensemble': ['original_rf_core', 'original_extra_core'],
    'expanded_forest_ensemble': ['rf_expanded', 'extra_expanded'],
}

def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding='utf-8')

def load_data():
    spec = importlib.util.spec_from_file_location('model_b_previous_feature_code', RUN / 'code/model_b_feature_engine.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DATA_PATH = DATA_PATH
    module.THRESHOLD_PATH = DATA_DIR / 'model_b_fixed_class_thresholds.json'
    data, thresholds = module.load_data()
    data = data.sort_values(['date', 'country_key']).reset_index(drop=True)
    assert not data.duplicated(['date', 'country_key']).any()
    assert np.allclose(data.groupby('date').country_share.sum(), 1, atol=1e-8)
    assert not any(c.startswith('target_') for c in set(CORE + EXPANDED + HISTORY))
    for h in HORIZONS:
        valid = data[f'target_class_h{h}'].notna()
        dates = data.loc[valid, 'date'] + pd.DateOffset(months=h)
        assert dates.equals(data.loc[valid, f'target_month_h{h}'])
        actual = data.loc[valid, f'target_share_change_h{h}'].to_numpy()
        predicted_labels = module.class_from_change(actual, thresholds)
        assert np.array_equal(predicted_labels, data.loc[valid, f'target_class_h{h}'].to_numpy())
    return data, thresholds

def train_rows(data, h, origins):
    cutoff = origins.origin_data_available_date_assumed.min()
    eligible = data[f'model_b_training_eligible_h{h}'].astype(str).str.lower().eq('true')
    frame = data.loc[eligible & (data.date < origins.date.min()) & (data[f'target_data_available_date_assumed_h{h}'] <= cutoff)].copy()
    assert frame[f'target_data_available_date_assumed_h{h}'].max() <= cutoff
    assert frame.date.max() < origins.date.min()
    if len(frame) < 150:
        raise ValueError('Insufficient chronological training history')
    return frame

def fit_model(name, train, h):
    config = CANDIDATES[name].copy()
    family, feature_set = config.pop('family'), config.pop('features')
    features = FEATURE_SETS[feature_set]
    categorical = [c for c in features if c in ['country_key', 'last_observed_3m_class']]
    numeric = [c for c in features if c not in categorical]
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if family == 'logistic':
        steps.append(('scale', StandardScaler()))
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline(steps), numeric),
        ('category', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
    ], sparse_threshold=0)
    if family == 'rf':
        estimator = RandomForestClassifier(**config, n_jobs=2, random_state=SEED)
    elif family == 'extra':
        estimator = ExtraTreesClassifier(**config, n_jobs=2, random_state=SEED)
    elif family == 'logistic':
        estimator = LogisticRegression(**config, max_iter=3000, random_state=SEED)
    else:
        estimator = HistGradientBoostingClassifier(**config, random_state=SEED)
    pipeline = Pipeline([('preprocessing', preprocessing), ('estimator', estimator)])
    pipeline.fit(train[features], train[f'target_class_h{h}'])
    return pipeline

def raw_probabilities(model, origins):
    raw = model.predict_proba(origins)
    aligned = np.zeros((len(origins), 4))
    for i, name in enumerate(model.named_steps['estimator'].classes_):
        aligned[:, CLASSES.index(name)] = raw[:, i]
    return aligned

def frequency_map(train, h):
    return {key: group[f'target_class_h{h}'].value_counts(normalize=True).reindex(CLASSES, fill_value=0).tolist()
            for key, group in train.groupby('country_key')}

def historical_probabilities(mapping, origins):
    return np.array([mapping[key] for key in origins.country_key])

def last_change_probabilities(origins):
    return np.eye(4)[[CLASSES.index(c) for c in origins.last_observed_3m_class]]

def adjusted_probabilities(raw, historical, shares, selection, thresholds):
    logits = np.log(np.clip(raw, 1e-12, 1)) / selection['temperature']
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs = (1-selection['frequency_weight']) * probs + selection['frequency_weight'] * historical
    if selection['feasible_classes']:
        # Future share is bounded by [0,1], so the change lies in [-share,1-share].
        minimum, maximum = -np.asarray(shares), 1-np.asarray(shares)
        possible = np.column_stack([
            minimum < thresholds['collapse_upper'],
            (minimum < thresholds['fall_upper']) & (maximum >= thresholds['collapse_upper']),
            (minimum <= thresholds['stable_upper']) & (maximum >= thresholds['fall_upper']),
            maximum > thresholds['stable_upper'],
        ])
        probs *= possible
        probs /= probs.sum(axis=1, keepdims=True)
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.isfinite(probs).all() and (probs >= 0).all()
    return probs

def reliability(actual, probabilities):
    labels = np.eye(4)[actual].ravel()
    values = probabilities.ravel()
    bins = np.minimum((values*10).astype(int), 9)
    records = []
    for index in range(10):
        selected = bins == index
        count = int(selected.sum())
        mean = float(values[selected].mean()) if count else None
        observed = float(labels[selected].mean()) if count else None
        records.append(dict(bin=index+1, lower=index/10, upper=(index+1)/10, count=count,
                            mean_probability=mean, observed_frequency=observed,
                            absolute_gap=abs(mean-observed) if count else None))
    return pd.DataFrame(records)

def score(origins, probabilities, historical, persistence, h):
    actual = np.array([CLASSES.index(c) for c in origins[f'target_class_h{h}']])
    observed = np.eye(4)[actual]
    bins = reliability(actual, probabilities)
    filled = bins[bins['count'] > 0]
    brier = float(np.mean(np.sum((probabilities-observed)**2, axis=1)))
    hist = float(np.mean(np.sum((historical-observed)**2, axis=1)))
    last = float(np.mean(np.sum((persistence-observed)**2, axis=1)))
    gap = float(filled.absolute_gap.max())
    record = dict(rows=len(origins), brier_score=brier, historical_frequency_brier=hist,
                  persistence_brier=last, max_calibration_gap=gap,
                  expected_calibration_error=float(np.average(filled.absolute_gap, weights=filled['count'])),
                  accuracy=float(np.mean(actual == probabilities.argmax(axis=1))),
                  balanced_accuracy=float(balanced_accuracy_score(actual, probabilities.argmax(axis=1))),
                  macro_f1=float(f1_score(actual, probabilities.argmax(axis=1), labels=range(4), average='macro', zero_division=0)),
                  log_loss=float(log_loss(actual, probabilities, labels=range(4))),
                  brier_skill_vs_frequency=1-brier/hist, beats_both_baselines=bool(brier<hist and brier<last),
                  calibration_pass=gap<=0.10, passes_acceptance=bool(brier<hist and brier<last and gap<=0.10))
    return record, bins

def validation(data, h):
    pieces, raws, historical, persistence = [], {name: [] for name in CANDIDATES}, [], []
    for fold, (start, end) in enumerate(BLOCKS, 1):
        origins = data[data.date.between(start, end) & data[f'target_class_h{h}'].notna() &
                       (data[f'target_month_h{h}'] <= pd.Timestamp('2021-06-01'))].copy()
        origins['fold'] = fold
        train = train_rows(data, h, origins)
        origins['training_rows'] = len(train)
        origins['max_training_target_available_date'] = train[f'target_data_available_date_assumed_h{h}'].max()
        for name in CANDIDATES:
            print(f'Validation H{h} fold {fold}: {name}, {len(train)} training rows', flush=True)
            model = fit_model(name, train, h)
            raws[name].append(raw_probabilities(model, origins))
        pieces.append(origins)
        historical.append(historical_probabilities(frequency_map(train, h), origins))
        persistence.append(last_change_probabilities(origins))
    origins = pd.concat(pieces, ignore_index=True)
    raw = {name: np.concatenate(values) for name, values in raws.items()}
    for name, members in ENSEMBLES.items():
        raw[name] = np.mean([raw[member] for member in members], axis=0)
    return origins, raw, np.concatenate(historical), np.concatenate(persistence)

def select_configuration(origins, candidates, historical, persistence, thresholds, h):
    comparisons=[]
    for name, raw in candidates.items():
        for temperature in [0.75, 1.0, 1.25, 1.5, 2.0]:
            for weight in [0.0, 0.15, 0.3, 0.5]:
                for feasible in [False, True]:
                    choice=dict(name=name, temperature=temperature, frequency_weight=weight, feasible_classes=feasible)
                    probabilities=adjusted_probabilities(raw,historical,origins.country_share,choice,thresholds)
                    metrics,_=score(origins,probabilities,historical,persistence,h)
                    comparisons.append(dict(horizon_months=h,**choice,**metrics))
    comparison=pd.DataFrame(comparisons)
    eligible=comparison[comparison.passes_acceptance]
    pool=eligible if len(eligible) else comparison[comparison.beats_both_baselines]
    if pool.empty:
        pool=comparison
    # Freeze selection on pre-July-2021 validation; tests never enter this ranking.
    best=pool.sort_values(['brier_score','expected_calibration_error','log_loss']).iloc[0]
    choice={key:best[key].item() if isinstance(best[key],np.generic) else best[key]
            for key in ['name','temperature','frequency_weight','feasible_classes']}
    choice['members']=ENSEMBLES.get(choice['name'],[choice['name']])
    choice['validation_pass']=bool(best.passes_acceptance)
    probabilities=adjusted_probabilities(candidates[choice['name']],historical,origins.country_share,choice,thresholds)
    print(f'SELECTED H{h}: {choice}; Brier={best.brier_score:.4f}; gap={best.max_calibration_gap:.4f}',flush=True)
    return choice,comparison,probabilities

def prediction_frame(origins, probabilities, historical, persistence, h, period, choice):
    frame=origins[['date','country_key','country','country_share',f'target_month_h{h}',
                   f'target_country_share_h{h}',f'target_share_change_h{h}',f'target_class_h{h}',
                   'training_rows','max_training_target_available_date']].copy()
    frame=frame.rename(columns={'date':'forecast_origin',f'target_month_h{h}':'target_month',
        f'target_country_share_h{h}':'actual_future_share',f'target_share_change_h{h}':'actual_share_change',
        f'target_class_h{h}':'actual_class'})
    frame['period']=period
    frame['horizon_months']=h
    frame['selected_model']=choice['name']
    frame['predicted_class']=np.array(CLASSES)[probabilities.argmax(axis=1)]
    frame['correct']=frame.actual_class==frame.predicted_class
    frame['confidence']=probabilities.max(axis=1)
    for i,c in enumerate(CLASSES):
        frame[f'probability_{c}']=probabilities[:,i]
        frame[f'historical_probability_{c}']=historical[:,i]
        frame[f'last_change_probability_{c}']=persistence[:,i]
    return frame

def evaluate(data,choice,thresholds,h):
    frames=[]
    origins_dates=sorted(data.loc[data.date >= '2021-07-01','date'].unique())
    for i,date in enumerate(origins_dates):
        origins=data[(data.date==date)&data[f'target_class_h{h}'].notna()].copy()
        if origins.empty:
            continue
        train=train_rows(data,h,origins)
        raws=[raw_probabilities(fit_model(name,train,h),origins) for name in choice['members']]
        historical=historical_probabilities(frequency_map(train,h),origins)
        persistence=last_change_probabilities(origins)
        probabilities=adjusted_probabilities(np.mean(raws,axis=0),historical,origins.country_share,choice,thresholds)
        origins['training_rows']=len(train)
        origins['max_training_target_available_date']=train[f'target_data_available_date_assumed_h{h}'].max()
        period='disruption_2021_2023' if date<=pd.Timestamp('2023-06-01') else 'post_disruption'
        frames.append(prediction_frame(origins,probabilities,historical,persistence,h,period,choice))
        if i%6==0:
            print(f'Test H{h}: {date:%Y-%m}, training={len(train)}',flush=True)
    return pd.concat(frames,ignore_index=True)

def metrics_for_frame(frame):
    h=int(frame.horizon_months.iloc[0])
    origins=frame.rename(columns={'actual_class':f'target_class_h{h}'})
    p=frame[[f'probability_{c}' for c in CLASSES]].to_numpy()
    hist=frame[[f'historical_probability_{c}' for c in CLASSES]].to_numpy()
    last=frame[[f'last_change_probability_{c}' for c in CLASSES]].to_numpy()
    return score(origins,p,hist,last,h)

def final_model(data,choice,thresholds,h):
    origins=data[data.date==data.date.max()].copy()
    train=train_rows(data,h,origins)
    pipelines=[{'candidate':name,'features':FEATURE_SETS[CANDIDATES[name]['features']],
                'pipeline':fit_model(name,train,h)} for name in choice['members']]
    bundle=dict(classes=CLASSES,horizon_months=h,configuration=choice,thresholds=thresholds,
                pipelines=pipelines,country_frequency=frequency_map(train,h),training_rows=len(train),
                source_cutoff=str(data.date.max().date()),
                source_sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())
    joblib.dump(bundle,RUN/f'models/model_b_h{h}_calibrated.joblib')
    return predict_bundle(bundle,origins)

def predict_bundle(bundle,origins):
    raw=np.mean([raw_probabilities(item['pipeline'],origins) for item in bundle['pipelines']],axis=0)
    historical=historical_probabilities(bundle['country_frequency'],origins)
    probs=adjusted_probabilities(raw,historical,origins.country_share,bundle['configuration'],bundle['thresholds'])
    h=bundle['horizon_months']
    rows=[]
    for i,(_,origin) in enumerate(origins.iterrows()):
        row=dict(forecast_origin=origin.date.strftime('%Y-%m'),target_month=(origin.date+pd.DateOffset(months=h)).strftime('%Y-%m'),
                 horizon_months=h,country_key=origin.country_key,country=origin.country,
                 current_share=float(origin.country_share),selected_model=bundle['configuration']['name'],
                 predicted_class=CLASSES[int(probs[i].argmax())],confidence=float(probs[i].max()))
        row.update({f'probability_{c}':float(probs[i,j]) for j,c in enumerate(CLASSES)})
        rows.append(row)
    return pd.DataFrame(rows)

def source_audit(data,thresholds):
    audit=dict(source_file=str(DATA_PATH),sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
               rows=len(data),supplier_groups=sorted(data.country.unique()),start=str(data.date.min().date()),
               end=str(data.date.max().date()),duplicate_rows=0,share_sums_verified=True,target_alignment_verified=True,
               assumption_notes=['FX availability: month end +7 days','ERUC availability: month end +31 days',
                   'Trade availability inherited as an assumption','Country weights reconstructed from upstream shares and HS weights'],
               outstanding_specification_gaps=['UN Comtrade rather than requested HMRC','No 2005-2009 history',
                   'Nickel/aluminium benchmark inputs absent under existing source restriction',
                   'Country membership hardcoded upstream; complete raw-partner selection requires audit'],
               thresholds=thresholds,prospective_holdout_required=True,
               test_interpretation='Historical periods were examined in prior experiments. New selection excludes test data, but these are reused historical tests, not a new prospective holdout.')
    # Reconcile reconstructed country weights to the prepared upstream data; this is not raw-source validation.
    enriched=pd.read_csv(DATA_DIR/'upstream_titanium_model_a_training_with_fx_eruc_2010_2026.csv')
    errs=[]
    for key in sorted(data.country_key.unique()):
        reconstructed=(enriched[f'supplier_share_{key}']*enriched.net_weight_kg).groupby(enriched.date).sum()
        actual=data[data.country_key==key].set_index(data.loc[data.country_key==key,'date'].dt.strftime('%Y-%m-%d')).country_quantity_kg
        error=float(np.abs(actual-reconstructed.reindex(actual.index)).max())
        errs.append(dict(country_key=key,max_difference_kg=error))
        assert np.allclose(actual,reconstructed.reindex(actual.index),rtol=1e-8,atol=.05)
    audit['upstream_reconstruction_reconciled']=True
    audit['reconstruction_differences']=errs
    russia=data[(data.country_key=='russia')&data.date.between('2021-07-01','2023-06-01')]
    audit['russia_disruption_collapse_labels']={f'h{h}':int(russia[f'target_class_h{h}'].eq('collapse').sum()) for h in HORIZONS}
    dump_json(RUN/'results/source_audit.json',audit)
    return audit

def render():
    metrics=pd.read_csv(RUN/'results/evaluation_metrics.csv')
    comparison=pd.read_csv(RUN/'results/previous_vs_new_metrics.csv')
    bins=pd.read_csv(RUN/'results/reliability_bins.csv')
    predictions=pd.read_csv(RUN/'results/all_ground_truth_predictions.csv')
    forecasts=pd.read_csv(RUN/'results/latest_forecasts.csv')
    original_choices=json.loads((DATA_DIR/'original_selected_models.json').read_text())
    earlier_choices=json.loads((DATA_DIR/'previous_experiment_manifest.json').read_text())['selections']
    for h in HORIZONS:
        row=(comparison.period=='validation')&(comparison.horizon_months==h)
        original=original_choices[f'h{h}']['validation_metrics']
        earlier=earlier_choices[f'h{h}']['validation_metrics']
        comparison.loc[row,'previous_brier']=original['brier_score']
        comparison.loc[row,'previous_accuracy']=original['accuracy']
        comparison.loc[row,'previous_calibration_gap']=original['max_calibration_gap']
        comparison.loc[row,'accuracy_experiment_brier']=earlier['brier_score']
        comparison.loc[row,'accuracy_experiment_accuracy']=earlier['accuracy']
    comparison['brier_change_vs_original']=comparison.brier_score-comparison.previous_brier
    comparison['accuracy_change_percentage_points']=(comparison.accuracy-comparison.previous_accuracy)*100
    comparison.to_csv(RUN/'results/previous_vs_new_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,h in zip(axes,HORIZONS):
        part=comparison[comparison.horizon_months==h].set_index('period').loc[['validation','disruption_2021_2023','post_disruption']]
        x=np.arange(len(part));width=.23
        for shift,col,label,color in [(-1,'previous_brier','Original model','#7C8C99'),(0,'accuracy_experiment_brier','Accuracy experiment','#BC7644'),(1,'brier_score','New probability model','#28635C')]:
            ax.bar(x+shift*width,part[col],width,label=label,color=color)
        ax.set_xticks(x,['Validation','Disruption','Post-disruption']);ax.set_title(f'H{h} probability error');ax.set_ylabel('Brier score (lower is better)')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(RUN/'visualizations/brier_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    for ax,(period,h) in zip(axes.ravel(),[(p,h) for p in ['disruption_2021_2023','post_disruption'] for h in HORIZONS]):
        part=bins[(bins.period==period)&(bins.horizon_months==h)&(bins['count']>0)]
        ax.plot([0,1],[0,1],color='#888888',linestyle='--');ax.plot(part.mean_probability,part.observed_frequency,'o-',color='#28635C')
        for _,row in part.iterrows():ax.annotate(f"n={int(row['count'])}",(row.mean_probability,row.observed_frequency),fontsize=7,xytext=(3,4),textcoords='offset points')
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Predicted probability');ax.set_ylabel('Observed frequency');ax.set_title(f'{period.replace("_"," ")} H{h}')
    fig.tight_layout();fig.savefig(RUN/'visualizations/probability_reliability.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,h in zip(axes,HORIZONS):
        part=predictions[(predictions.horizon_months==h)&predictions.country_key.eq('russia')&predictions.period.eq('disruption_2021_2023')]
        ax.plot(pd.to_datetime(part.forecast_origin),part.country_share*100,label='Origin share (%)',color='#28635C')
        ax.plot(pd.to_datetime(part.forecast_origin),part.actual_future_share*100,label='Future actual share (%)',color='#C58B32')
        ax.plot(pd.to_datetime(part.forecast_origin),part.probability_collapse*100,label='Collapse probability (%)',color='#A64239')
        ax.set_title(f'Russia H{h}: no collapse-labelled actual rows');ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(RUN/'visualizations/russia_risk_check.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,h in zip(axes,HORIZONS):
        part=predictions[(predictions.horizon_months==h)&predictions.period.eq('post_disruption')]
        cm=confusion_matrix(part.actual_class,part.predicted_class,labels=CLASSES)
        ax.imshow(cm,cmap='Blues');ax.set_xticks(range(4),CLASSES,rotation=30);ax.set_yticks(range(4),CLASSES)
        for i in range(4):
            for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#23364D')
        ax.set_title(f'Post-disruption H{h}');ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
    fig.tight_layout();fig.savefig(RUN/'visualizations/confusion_matrices.png',dpi=150);plt.close(fig)
    def image(name):
        return 'data:image/png;base64,'+base64.b64encode((RUN/'visualizations'/name).read_bytes()).decode()
    display=metrics.copy()
    for c in ['accuracy','balanced_accuracy','macro_f1']:display[c]=display[c].map(lambda v:f'{v:.1%}')
    for c in ['brier_score','historical_frequency_brier','persistence_brier','max_calibration_gap']:display[c]=display[c].map(lambda v:f'{v:.4f}')
    display['result']=np.where(display.passes_acceptance,'Pass probability gates','Fail probability gates')
    display['period']=display.period.map({'validation':'Validation','disruption_2021_2023':'Disruption','post_disruption':'Post-disruption'})
    display['horizon_months']=display.horizon_months.map(lambda h:f'H{h}')
    table=display[['period','horizon_months','accuracy','brier_score','historical_frequency_brier','persistence_brier','max_calibration_gap','result']].rename(columns={
        'period':'Evaluation','horizon_months':'Horizon','accuracy':'Correct classes','brier_score':'Model Brier',
        'historical_frequency_brier':'Frequency baseline','persistence_brier':'Last-change baseline',
        'max_calibration_gap':'Largest calibration gap','result':'Result'}).to_html(index=False,border=0)
    payload=predictions.where(pd.notna(predictions),None).to_dict('records')
    js=json.dumps(payload,allow_nan=False).replace('</','<\\/')
    page='''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Titanium Model B probability improvement</title><style>
    :root{--ink:#243C48;--accent:#28635C;--paper:#F8F7F2}*{box-sizing:border-box}body{margin:0;background:linear-gradient(120deg,#E9EFEC,#F8F7F2);color:var(--ink);font:16px Georgia,serif}main{max-width:1250px;margin:auto;padding:30px}h1{font-size:36px}h2{font-size:25px}section{background:var(--paper);border:1px solid #CCD5D1;padding:24px;margin:22px 0}p{line-height:1.6}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font:13px Verdana,sans-serif}th,td{text-align:left;padding:11px;border-bottom:1px solid #DEE5E1}th{background:#243C48;color:white;white-space:nowrap}img{width:100%;height:auto}select,button{font:14px Verdana,sans-serif;padding:9px;margin:5px;border:1px solid #9BAEA7;background:white;color:var(--ink)}.notice{border-left:4px solid #BB7439;padding:12px;background:#F2EADD}.cards{display:flex;gap:12px;flex-wrap:wrap}.cards div{padding:16px;border:1px solid #CCD5D1;flex:1;min-width:220px}.cards b{display:block;font:24px Georgia,serif;margin-top:10px}@media(max-width:650px){main{padding:12px}section{padding:16px}h1{font-size:28px}}
    </style></head><body><main><h1>Titanium supply risk: H3 and H6</h1><p>Model B probability improvement experiment. Forecasts describe each supplier group's change in UK titanium import-weight share.</p>'''
    test=metrics[metrics.period.ne('validation')]
    page+=f'<div class="cards"><div>Historical probability gates passed<b>{int(test.passes_acceptance.sum())}/4</b></div><div>Current source cutoff<b>June 2026</b></div><div>Required horizons<b>3 and 6 months</b></div></div>'
    page+='<section><h2>How this experiment works</h2><p>Models and calibration settings were selected using forecasts up to June 2021. Calibration adjusts probability sharpness and can blend with each country\'s historical class frequencies. A validated share-bound rule removes mathematically impossible classes where selected.</p><p class="notice">The disruption and post-disruption periods have been reviewed in previous experiments. These are reused historical evaluations. A fresh prospective holdout is required before a production reliability claim.</p></section>'
    page+='<section><h2>Probability acceptance results</h2><p>Lower Brier score is better. Acceptance requires beating both baselines and a maximum gap of 0.10 across all populated probability bins. Every bin is retained, including sparse bins.</p><div class="scroll">'+table+'</div></section>'
    page+='<section><h2>Previous models versus this experiment</h2><img alt="Brier comparison" src="'+image('brier_comparison.png')+'"></section>'
    page+='<section><h2>What improved and what remains</h2><p>Both horizons pass the validation probability gates. Post-disruption H3 Brier falls from 0.5835 to 0.5666; H6 falls from 0.5851 to 0.5827. All four test groups beat both Brier baselines. Calibration still fails disruption H3 and both post-disruption horizons. Keep these models as experimental; the original H3 model retains the better disruption calibration.</p><p>Post-disruption H3 misses the calibration limit in a bin with 64 observations. Post-disruption H6 has a larger error in a bin with five observations. Sparse bins are still included in the required acceptance test.</p></section>'
    page+='<section><h2>Probability reliability</h2><p>Points close to the diagonal indicate better reliability. Bin labels show the number of class-probability observations. Confidence is the largest class probability; it does not establish an individual prediction will be correct.</p><img alt="Reliability" src="'+image('probability_reliability.png')+'"></section>'
    page+='<section><h2>Actual outcomes and predictions</h2><label>Horizon <select id="h"><option value="3">H3</option><option value="6">H6</option></select></label><label>Period <select id="period"><option>post_disruption</option><option>disruption_2021_2023</option><option>validation</option></select></label><label>Supplier <select id="country"><option value="all">All suppliers</option></select></label><button id="download">Download filtered CSV</button><p id="counts"></p><div class="scroll"><table><thead><tr><th>Origin</th><th>Target</th><th>Supplier</th><th>Actual</th><th>Predicted</th><th>Correct</th><th>Collapse</th><th>Fall</th><th>Stable</th><th>Rise</th></tr></thead><tbody id="rows"></tbody></table></div></section>'
    page+='<section><h2>Russia disruption check</h2><p>The prepared H3/H6 targets contain no Russia collapse-class event in July 2021-June 2023. This requirement remains unresolved. The frozen class boundaries are retained; source reconstruction and label design need further raw-data review.</p><img alt="Russia check" src="'+image('russia_risk_check.png')+'"></section>'
    page+='<section><h2>Classification diagnostics</h2><img alt="Confusion matrices" src="'+image('confusion_matrices.png')+'"></section>'
    page+='<section><h2>Latest country probabilities</h2><p>These forecasts originate in June 2026, the last observation in the supplied data. A system-date change does not provide newer input observations.</p><div class="scroll">'+forecasts.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')+'</div></section>'
    page+='''<section><h2>Data and remaining requirements</h2><p>Uses the existing UN Comtrade, Bank of England FX and ONS ERUC data. Supplier quantities are reconstructed from upstream shares and HS weights. Publication timing is assumed. HMRC data, 2005-2009 history and required nickel/aluminium benchmark inputs remain absent. The dataset and class boundaries are recorded in the source audit.</p></section><script>
    const data=PAYLOAD;const qs=id=>document.getElementById(id);const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    [...new Set(data.map(r=>r.country))].sort().forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;qs('country').appendChild(o)});
    function filtered(){return data.filter(r=>r.horizon_months===Number(qs('h').value)&&r.period===qs('period').value&&(qs('country').value==='all'||r.country===qs('country').value))}
    function draw(){const rows=filtered();qs('counts').textContent=rows.length+' predictions; '+(rows.length?100*rows.filter(r=>r.correct).length/rows.length:0).toFixed(1)+'% correct.';qs('rows').innerHTML=rows.map(r=>'<tr>'+[r.forecast_origin.slice(0,7),r.target_month.slice(0,7),r.country,r.actual_class,r.predicted_class,r.correct?'Yes':'No',...['collapse','fall','stable','rise'].map(c=>(r['probability_'+c]*100).toFixed(1)+'%')].map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')}
    ['h','period','country'].forEach(id=>qs(id).addEventListener('change',draw));draw();
    qs('download').addEventListener('click',()=>{const rows=filtered();if(!rows.length)return;const keys=Object.keys(rows[0]);const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';const csv=[keys.map(quote).join(','),...rows.map(r=>keys.map(k=>quote(r[k])).join(','))].join('\\r\\n');const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='model_b_filtered_predictions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)});
    </script></main></body></html>'''.replace('PAYLOAD',js)
    (RUN/'visualizations/model_b_probability_improvement_dashboard.html').write_text(page,encoding='utf-8')

def run_training():
    for folder in ['results','models','visualizations','documentation']:
        (RUN/folder).mkdir(parents=True,exist_ok=True)
    data,thresholds=load_data()
    audit=source_audit(data,thresholds)
    selections={};validation_frames=[];comparisons=[]
    for h in HORIZONS:
        origins,raw,historical,persistence=validation(data,h)
        choice,comparison,probs=select_configuration(origins,raw,historical,persistence,thresholds,h)
        selections[h]=choice;comparisons.append(comparison)
        validation_frames.append(prediction_frame(origins,probs,historical,persistence,h,'validation',choice))
    dump_json(RUN/'results/selected_configurations.json',selections)
    pd.concat(comparisons).to_csv(RUN/'results/validation_configuration_comparison.csv',index=False)
    frames=validation_frames+[evaluate(data,selections[h],thresholds,h) for h in HORIZONS]
    predictions=pd.concat(frames,ignore_index=True).sort_values(['period','horizon_months','forecast_origin','country_key'])
    predictions.to_csv(RUN/'results/all_ground_truth_predictions.csv',index=False)
    records=[];bins=[];by_country=[];by_class=[]
    for (period,h),group in predictions.groupby(['period','horizon_months']):
        record,table=metrics_for_frame(group)
        records.append(dict(period=period,horizon_months=int(h),selected_model=selections[int(h)]['name'],**record))
        bins.append(table.assign(period=period,horizon_months=h))
        for country,part in group.groupby('country'):
            item,_=metrics_for_frame(part);by_country.append(dict(period=period,horizon_months=int(h),country=country,**item))
        for label in CLASSES:
            subset=group[group.actual_class==label]
            by_class.append(dict(period=period,horizon_months=int(h),actual_class=label,rows=len(subset),
                recall=float(subset.correct.mean()) if len(subset) else None))
    metrics=pd.DataFrame(records)
    metrics.to_csv(RUN/'results/evaluation_metrics.csv',index=False)
    pd.concat(bins).to_csv(RUN/'results/reliability_bins.csv',index=False)
    pd.DataFrame(by_country).to_csv(RUN/'results/country_metrics.csv',index=False)
    pd.DataFrame(by_class).to_csv(RUN/'results/class_recall.csv',index=False)
    for period,part in predictions.groupby('period'):part.to_csv(RUN/f'results/{period}_predictions.csv',index=False)
    original=pd.read_csv(DATA_DIR/'original_evaluation_metrics.csv').rename(columns={'brier_score':'previous_brier','max_calibration_gap':'previous_calibration_gap','accuracy':'previous_accuracy'})
    earlier=pd.read_csv(DATA_DIR/'previous_improved_evaluation_metrics.csv').rename(columns={'brier_score':'accuracy_experiment_brier','accuracy':'accuracy_experiment_accuracy'})
    comparison=metrics.merge(original[['period','horizon_months','previous_brier','previous_accuracy','previous_calibration_gap']],on=['period','horizon_months'],how='left')
    comparison=comparison.merge(earlier[['period','horizon_months','accuracy_experiment_brier','accuracy_experiment_accuracy']],on=['period','horizon_months'],how='left')
    comparison['brier_change_vs_original']=comparison.brier_score-comparison.previous_brier
    comparison['accuracy_change_percentage_points']=(comparison.accuracy-comparison.previous_accuracy)*100
    comparison.to_csv(RUN/'results/previous_vs_new_metrics.csv',index=False)
    forecasts=pd.concat([final_model(data,selections[h],thresholds,h) for h in HORIZONS],ignore_index=True)
    forecasts.to_csv(RUN/'results/latest_forecasts.csv',index=False)
    test=metrics[metrics.period!='validation']
    status='passes historical probability gates' if test.passes_acceptance.all() else 'research model: some probability gates fail'
    dump_json(RUN/'results/model_b_response.json',dict(model='Titanium supply shift by country',version='probability_calibration_v2',status=status,
         source_cutoff='2026-06',source_assumptions=audit['assumption_notes'],remaining_gaps=audit['outstanding_specification_gaps'],
         prospective_holdout_required=True,forecasts=forecasts.to_dict('records')))
    dump_json(RUN/'results/model_manifest.json',dict(configurations=selections,feature_sets=FEATURE_SETS,classes=CLASSES,thresholds=thresholds,
         candidates=CANDIDATES,source_sha256=audit['sha256'],selection_rule='Among validation configurations that pass both Brier baselines and calibration <=0.10, lowest Brier; otherwise lowest Brier beating baselines.',
         selection_period='2018-01 through 2021-06; targets through 2021-06',random_seed=SEED,test_status=status))
    render()
    print(metrics[['period','horizon_months','brier_score','accuracy','max_calibration_gap','passes_acceptance']].to_string(index=False),flush=True)
    print(f'Dashboard: {RUN}/visualizations/model_b_probability_improvement_dashboard.html',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--predict-only',action='store_true')
    args=parser.parse_args()
    if args.render_only:render()
    elif args.predict_only:
        data,_=load_data();latest=data[data.date==data.date.max()]
        out=pd.concat([predict_bundle(joblib.load(RUN/f'models/model_b_h{h}_calibrated.joblib'),latest) for h in HORIZONS])
        print(out.to_string(index=False))
    else:run_training()
