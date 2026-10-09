from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent
DATA_PATH = ROOT / 'Train/model_c_freight_training_2005_2026.csv'
PREPARATION = json.loads((ROOT / 'Train/model_c_preparation_manifest.json').read_text())
FEATURES = PREPARATION['feature_columns']
QUANTILES = np.array([.05, .10, .25, .50, .75, .90, .95])
QNAMES = ['p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95']
HORIZONS = [3, 6, 12]
MODELS = ['ridge', 'random_forest', 'extra_trees', 'gradient_boosting', 'no_change']
LABELS = {'ridge': 'Ridge + residual quantiles', 'random_forest': 'Quantile Random Forest',
          'extra_trees': 'Quantile Extra Trees', 'gradient_boosting': 'Gradient Boosting quantiles',
          'no_change': 'No-change baseline'}
PARAMETERS = {'ridge': {'alpha': 1.0},
              'random_forest': {'n_estimators': 120, 'max_depth': 7, 'min_samples_leaf': 6, 'max_features': .8},
              'extra_trees': {'n_estimators': 120, 'max_depth': 8, 'min_samples_leaf': 6, 'max_features': .8},
              'gradient_boosting': {'n_estimators': 80, 'max_depth': 2, 'min_samples_leaf': 8, 'learning_rate': .05}}
SEED = 42
CALIBRATION_ROWS = 24
ALPHA = .10
ERRORS = []


def json_clean(value):
    if isinstance(value, dict): return {str(k):json_clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray): return json_clean(value.tolist())
    if isinstance(value, np.generic): return json_clean(value.item())
    if isinstance(value, (pd.Timestamp, Path)): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(json_clean(value), indent=2, allow_nan=False), encoding='utf-8')


def load_data():
    frame = pd.read_csv(DATA_PATH)
    for c in frame:
        if c == 'date' or c.endswith('_date') or '_date_assumed' in c or c.startswith('target_month_'):
            frame[c] = pd.to_datetime(frame[c], errors='coerce')
    frame = frame.sort_values('date').reset_index(drop=True)
    assert len(FEATURES) == 8 and not any(c.startswith('target_') or 'assumed' in c for c in FEATURES)
    assert frame[FEATURES].notna().all().all()
    assert not frame.date.duplicated().any()
    assert pd.PeriodIndex(frame.date, freq='M').equals(pd.period_range(frame.date.min(), frame.date.max(), freq='M'))
    return frame


def available_training(frame, origin, h):
    train = frame.loc[(frame.date < origin.date) & frame[f'target_freight_log_change_h{h}'].notna()
                      & (frame[f'target_available_date_assumed_h{h}'] <= origin.forecast_date_assumed)].copy()
    assert train.date.max() < origin.date
    assert train[f'target_available_date_assumed_h{h}'].max() <= origin.forecast_date_assumed
    return train


def partition_calibration(train, h):
    calibration = train.tail(CALIBRATION_ROWS).copy()
    first = calibration.iloc[0]
    proper = train.loc[(train.date < first.date)
                       & (train[f'target_available_date_assumed_h{h}'] <= first.forecast_date_assumed)].copy()
    if len(proper) < 40 or len(calibration) < CALIBRATION_ROWS:
        raise ValueError('Insufficient proper training/calibration history')
    assert set(proper.date).isdisjoint(set(calibration.date))
    return proper, calibration


def fit_core(name, proper, h, override=None):
    x = proper[FEATURES].to_numpy(float)
    y = proper[f'target_freight_log_change_h{h}'].to_numpy(float)
    params = {**PARAMETERS.get(name, {}), **(override or {})}
    bundle = dict(model=name,features=FEATURES,horizon_months=h,parameters=params,quantiles=QUANTILES.tolist())
    if name == 'no_change':
        raw = np.quantile(y,QUANTILES)
        raw[:3] = np.minimum(raw[:3],0)
        raw[3] = 0.0
        raw[4:] = np.maximum(raw[4:],0)
        bundle['empirical_error_quantiles'] = raw
    elif name == 'ridge':
        estimator = make_pipeline(StandardScaler(), Ridge(**params))
        estimator.fit(x,y)
        bundle.update(estimator=estimator, empirical_error_quantiles=np.quantile(y-estimator.predict(x),QUANTILES))
    elif name in ['random_forest','extra_trees']:
        klass = RandomForestRegressor if name == 'random_forest' else ExtraTreesRegressor
        estimator = klass(**params,random_state=SEED,n_jobs=2)
        estimator.fit(x,y)
        leaves = estimator.apply(x)
        # Retain historical outcomes by leaf. Each tree contributes total weight one.
        leaf_values = [{int(leaf):y[leaves[:,j]==leaf] for leaf in np.unique(leaves[:,j])}
                       for j in range(leaves.shape[1])]
        bundle.update(estimator=estimator,leaf_values=leaf_values)
    else:
        estimators = []
        for q in QUANTILES:
            estimator = GradientBoostingRegressor(loss='quantile',alpha=float(q),random_state=SEED,**params)
            estimator.fit(x,y)
            estimators.append(estimator)
        bundle['estimators'] = estimators
    return bundle


def weighted_quantiles(values, weights):
    order = np.argsort(values)
    values,weights = values[order],weights[order]
    cdf = (np.cumsum(weights)-.5*weights)/weights.sum()
    return np.interp(QUANTILES,cdf,values)


def raw_predict(bundle, x):
    x = np.asarray(x,float)
    name = bundle['model']
    if name == 'no_change':
        q = np.tile(bundle['empirical_error_quantiles'],(len(x),1))
    elif name == 'ridge':
        q = bundle['estimator'].predict(x)[:,None] + bundle['empirical_error_quantiles'][None,:]
    elif name in ['random_forest','extra_trees']:
        leaves = bundle['estimator'].apply(x)
        result = []
        for row in leaves:
            parts = [bundle['leaf_values'][j][int(leaf)] for j,leaf in enumerate(row)]
            values = np.concatenate(parts)
            weights = np.concatenate([np.full(len(part),1/len(part)) for part in parts])
            result.append(weighted_quantiles(values,weights))
        q = np.array(result)
    else:
        q = np.column_stack([m.predict(x) for m in bundle['estimators']])
    return np.sort(q,axis=1)


def calibrated_predict(bundle, x):
    raw = raw_predict(bundle,x)
    offsets = (QUANTILES-.5)/.45*bundle['calibration_expansion']
    q = raw+offsets[None,:]
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all()
    return q,raw


def fit_calibrated(frame, name, origin, h):
    train = available_training(frame,origin,h)
    proper,calibration = partition_calibration(train,h)
    bundle = fit_core(name,proper,h)
    raw = raw_predict(bundle,calibration[FEATURES].to_numpy())
    actual = calibration[f'target_freight_log_change_h{h}'].to_numpy()
    nonconformity = np.maximum(raw[:,0]-actual,actual-raw[:,-1])
    order = int(math.ceil((len(nonconformity)+1)*(1-ALPHA)))
    delta = max(0.,float(np.sort(nonconformity)[min(order,len(nonconformity))-1]))
    bundle.update(calibration_expansion=delta,calibration_rows=len(calibration),
        proper_training_rows=len(proper),available_training_rows=len(train),
        proper_training_start=proper.date.min().strftime('%Y-%m'),proper_training_end=proper.date.max().strftime('%Y-%m'),
        calibration_origin_start=calibration.date.min().strftime('%Y-%m'),calibration_origin_end=calibration.date.max().strftime('%Y-%m'),
        model_fit_asof=origin.forecast_date_assumed.strftime('%Y-%m-%d'),
        fit_origin_reference_month=origin.date.strftime('%Y-%m'),
        max_training_target_available_date=train[f'target_available_date_assumed_h{h}'].max().strftime('%Y-%m-%d'),
        calibration_method='Chronological split CQR: latest 24 known labels held out; earlier core labels purged by availability; finite-sample 90% score order; nonnegative symmetric expansion.',
        timing_status='Assumed publication dates; historical revision vintages unverified')
    return bundle


def record(origin,h,name,period,bundle,q,raw):
    actual = float(origin[f'target_freight_log_change_h{h}'])
    row=dict(model=name,model_label=LABELS[name],horizon_months=h,period=period,
        origin_month=origin.date.strftime('%Y-%m'),forecast_date_assumed=origin.forecast_date_assumed.strftime('%Y-%m-%d'),
        target_month=origin[f'target_month_h{h}'].strftime('%Y-%m'),
        target_available_date_assumed=origin[f'target_available_date_assumed_h{h}'].strftime('%Y-%m-%d'),
        origin_freight_index=float(origin.freight_index),actual_freight_index=float(origin[f'target_freight_index_h{h}']),
        actual_log_change=actual,actual_pct_change=float(np.expm1(actual)*100),
        proper_training_rows=bundle['proper_training_rows'],calibration_rows=bundle['calibration_rows'],
        model_fit_asof=bundle['model_fit_asof'],max_training_target_available_date=bundle['max_training_target_available_date'],
        calibration_expansion=bundle['calibration_expansion'],forecast_produced=True)
    for i,p in enumerate(QNAMES):
        row[f'{p}_log_change']=float(q[i]);row[f'{p}_freight_index']=float(origin.freight_index*np.exp(q[i]))
        row[f'{p}_pct_change']=float(np.expm1(q[i])*100);row[f'raw_{p}_log_change']=float(raw[i])
    row['inside_90_interval']=bool(q[0]<=actual<=q[-1])
    row['predicted_freight_index']=row['p50_freight_index']
    row['error_freight_index']=row['predicted_freight_index']-row['actual_freight_index']
    row['absolute_pct_error']=abs(row['error_freight_index'])/row['actual_freight_index']*100
    return row


def missing_record(origin,h,name,period,error):
    ERRORS.append(dict(model=name,horizon_months=h,origin_month=origin.date.strftime('%Y-%m'),period=period,error=str(error)))
    row=dict(model=name,model_label=LABELS[name],horizon_months=h,period=period,
        origin_month=origin.date.strftime('%Y-%m'),forecast_date_assumed=origin.forecast_date_assumed.strftime('%Y-%m-%d'),
        target_month=origin[f'target_month_h{h}'].strftime('%Y-%m'),
        target_available_date_assumed=origin[f'target_available_date_assumed_h{h}'].strftime('%Y-%m-%d'),
        origin_freight_index=float(origin.freight_index),actual_freight_index=float(origin[f'target_freight_index_h{h}']),
        actual_log_change=float(origin[f'target_freight_log_change_h{h}']),forecast_produced=False,inside_90_interval=False)
    for p in QNAMES:row.update({f'{p}_log_change':np.nan,f'{p}_freight_index':np.nan,f'{p}_pct_change':np.nan})
    return row


def pinball(actual, q):
    residual=np.asarray(actual)[:,None]-q
    return float(np.mean(np.maximum(QUANTILES[None,:]*residual,(QUANTILES[None,:]-1)*residual)))


def metric_rows(records):
    rows=[]
    for (period,name,h),group in records.groupby(['period','model','horizon_months']):
        q=group[[f'{p}_log_change' for p in QNAMES]].to_numpy()
        y=group.actual_log_change.to_numpy()
        valid=np.isfinite(q).all(axis=1)
        if valid.any():
            a=y[valid];v=q[valid];width=v[:,-1]-v[:,0]
            mis=width+2/ALPHA*(v[:,0]-a)*(a<v[:,0])+2/ALPHA*(a-v[:,-1])*(a>v[:,-1])
            index_actual=group.loc[valid,'actual_freight_index'].to_numpy()
            index_pred=group.loc[valid,'p50_freight_index'].to_numpy()
            item=dict(pinball_loss=pinball(a,v),mae_log_change=float(np.mean(abs(a-v[:,3]))),
                rmse_log_change=float(np.sqrt(np.mean((a-v[:,3])**2))),
                mae_freight_index=float(np.mean(abs(index_actual-index_pred))),
                rmse_freight_index=float(np.sqrt(np.mean((index_actual-index_pred)**2))),
                mape_freight_index=float(np.mean(abs(index_actual-index_pred)/index_actual)*100),
                smape_freight_index=float(np.mean(200*abs(index_actual-index_pred)/(abs(index_actual)+abs(index_pred)))),
                interval_width_log_change=float(width.mean()),mis_available_forecasts=float(mis.mean()))
        else:item={k:np.nan for k in ['pinball_loss','mae_log_change','rmse_log_change','mae_freight_index','rmse_freight_index','mape_freight_index','smape_freight_index','interval_width_log_change','mis_available_forecasts']}
        missing=int((~valid).sum())
        coverage=float(np.sum(valid & (y>=q[:,0]) & (y<=q[:,-1]))/len(group))
        rows.append(dict(period=period,model=name,model_label=LABELS[name],horizon_months=int(h),
            expected_forecasts=len(group),produced_forecasts=int(valid.sum()),missing_forecasts=missing,
            coverage_90=coverage,coverage_pass=coverage>=.90,
            mean_interval_score=item['mis_available_forecasts'] if missing==0 else np.nan,**item))
    result=pd.DataFrame(rows)
    base=result[result.model=='no_change'][['period','horizon_months','mean_interval_score']].rename(columns={'mean_interval_score':'baseline_mis'})
    result=result.merge(base,on=['period','horizon_months'],how='left')
    result['mis_ratio_vs_baseline']=result.mean_interval_score/result.baseline_mis
    result['beats_baseline_mis']=result.mean_interval_score<result.baseline_mis
    result['probability_gates_pass']=result.coverage_pass & result.beats_baseline_mis & result.missing_forecasts.eq(0)
    result['h12_reporting_only']=result.horizon_months.eq(12)
    return result


def validation_predictions(frame):
    rows=[]
    for h in HORIZONS:
        origins=frame[frame[f'validation_selection_eligible_h{h}'].astype(str).str.lower().eq('true')]
        for name in MODELS:
            for year,block in origins.groupby(origins.date.dt.year):
                first=block.iloc[0]
                print(f'Validation {LABELS[name]} H{h}, {year}: {len(block)} forecasts',flush=True)
                try:
                    bundle=fit_calibrated(frame,name,first,h)
                    q,raw=calibrated_predict(bundle,block[FEATURES].to_numpy())
                    rows.extend(record(origin,h,name,'validation',bundle,q[i],raw[i]) for i,(_,origin) in enumerate(block.iterrows()))
                except Exception as error:
                    rows.extend(missing_record(origin,h,name,'validation',error) for _,origin in block.iterrows())
    return pd.DataFrame(rows)


def select_models(metrics):
    selections={};global_scores=[]
    for h in [3,6]:
        group=metrics[(metrics.period=='validation')&(metrics.horizon_months==h)&metrics.model.ne('no_change')]
        eligible=group[group.probability_gates_pass]
        best=group.sort_values(['mean_interval_score','pinball_loss']).iloc[0]
        selected=eligible.sort_values(['mean_interval_score','pinball_loss']).iloc[0].model if len(eligible) else 'no_change'
        selections[str(h)]=dict(selected_model=selected,best_ml_candidate=best.model,
            any_ml_validation_acceptance_pass=bool(len(eligible)),validation_only_selection=True)
    for name,group in metrics[(metrics.period=='validation')&metrics.horizon_months.isin([3,6])&metrics.model.ne('no_change')].groupby('model'):
        global_scores.append((float(group.mis_ratio_vs_baseline.mean()),name))
    best_global=min(global_scores)[1]
    selections['12']=dict(selected_model=best_global if any(v['selected_model']!='no_change' for v in selections.values()) else 'no_change',
        best_ml_candidate=best_global,h12_reporting_only=True,selection_basis='Family selected from H3/H6 validation MIS ratios only; H12 metrics excluded.')
    print('Frozen selection:',json.dumps(selections),flush=True)
    save_json(RUN/'results/selected_models.json',selections)
    return selections


def later_predictions(frame):
    rows=[]
    for h in HORIZONS:
        origins=frame[(frame.date>='2015-01-01')&frame[f'target_observed_h{h}'].astype(str).str.lower().eq('true')].copy()
        used=origins[f'validation_selection_eligible_h{h}'].astype(str).str.lower().eq('true')
        origins=origins[~used]
        for name in MODELS:
            for i,(_,origin) in enumerate(origins.iterrows()):
                period='pretest_bridge' if origin.date<pd.Timestamp('2020-01-01') else 'disruption_test' if origin.date<=pd.Timestamp('2021-12-01') else 'post_disruption'
                try:
                    bundle=fit_calibrated(frame,name,origin,h)
                    q,raw=calibrated_predict(bundle,origin[FEATURES].to_numpy(float)[None,:])
                    rows.append(record(origin,h,name,period,bundle,q[0],raw[0]))
                except Exception as error:rows.append(missing_record(origin,h,name,period,error))
                if i%12==0:print(f'{period} {name} H{h}: {origin.date:%Y-%m}',flush=True)
            pd.DataFrame(rows).to_csv(RUN/'predictions/rolling_predictions_checkpoint.csv',index=False)
    return pd.DataFrame(rows)


def recovery_assessment(records):
    trajectories=[];summaries=[]
    for (name,h),group in records.groupby(['model','horizon_months']):
        group=group.copy()
        group['target_available']=pd.to_datetime(group.target_available_date_assumed)
        group['issued']=pd.to_datetime(group.forecast_date_assumed)
        group['target_date']=pd.to_datetime(group.target_month)
        first_issue=pd.Timestamp('2020-01-31')
        deadline=first_issue+pd.DateOffset(months=6)
        first_recovered=None
        for asof in pd.date_range('2020-01-31','2021-12-31',freq='ME'):
            known=group[(group.target_available<=asof)&(group.issued<=asof)].sort_values('target_date').tail(6)
            count=len(known);coverage=float(known.inside_90_interval.sum()/count) if count else np.nan
            impacted=int((known.target_date>=pd.Timestamp('2020-01-01')).sum())
            complete_window=count==6 and impacted==6
            trajectories.append(dict(model=name,horizon_months=int(h),evaluation_asof=asof.strftime('%Y-%m-%d'),
                completed_forecasts=count,disruption_target_months_in_window=impacted,coverage_6_completed=coverage,
                all_six_targets_post_disruption=complete_window))
            if complete_window and coverage>=.90 and first_recovered is None:first_recovered=asof
        summaries.append(dict(model=name,horizon_months=int(h),first_fully_disruption_window_recovered_asof=first_recovered,
            recovery_deadline=deadline,recovery_within_six_months=first_recovered is not None and first_recovered<=deadline,
            definition='Six latest completed target months as known at month-end; all six target months in/after Jan 2020; recovery deadline six months after Jan 2020 issue month-end.',
            h12_reporting_only=int(h)==12))
    return pd.DataFrame(trajectories),pd.DataFrame(summaries)


def diagnostic_curves(frame):
    rows=[];training_records=[]
    for h in HORIZONS:
        val=frame[frame[f'validation_selection_eligible_h{h}'].astype(str).str.lower().eq('true')&frame.date.between('2015-01-01','2017-12-01')]
        origin=val.iloc[0]
        initial=available_training(frame,origin,h)
        proper,cal=partition_calibration(initial,h)
        xv=val[FEATURES].to_numpy();yv=val[f'target_freight_log_change_h{h}'].to_numpy()
        xt=proper[FEATURES].to_numpy();yt=proper[f'target_freight_log_change_h{h}'].to_numpy()
        for name in MODELS:
            print(f'Diagnostics {name} H{h}',flush=True)
            calibrated=fit_calibrated(frame,name,origin,h)
            trainq,raw=calibrated_predict(calibrated,xt)
            training_records.extend(record(o,h,name,'in_sample_diagnostic',calibrated,trainq[i],raw[i]) for i,(_,o) in enumerate(proper.iterrows()))
            if name=='no_change':continue
            if name=='gradient_boosting':
                bundle=fit_core(name,proper,h)
                train_stages=[list(m.staged_predict(xt)) for m in bundle['estimators']]
                val_stages=[list(m.staged_predict(xv)) for m in bundle['estimators']]
                for stage in range(PARAMETERS[name]['n_estimators']):
                    tq=np.sort(np.column_stack([s[stage] for s in train_stages]),axis=1)
                    vq=np.sort(np.column_stack([s[stage] for s in val_stages]),axis=1)
                    rows.append(dict(model=name,horizon_months=h,step=stage+1,step_type='Boosting stages',training_rows=len(proper),validation_rows=len(val),train_pinball_loss=pinball(yt,tq),validation_pinball_loss=pinball(yv,vq)))
            elif name=='ridge':
                for n in sorted(set([40,55,70,85,len(proper)])):
                    if n>len(proper):continue
                    subset=proper.head(n);bundle=fit_core(name,subset,h)
                    rows.append(dict(model=name,horizon_months=h,step=n,step_type='Chronological training rows',training_rows=n,validation_rows=len(val),train_pinball_loss=pinball(subset[f'target_freight_log_change_h{h}'],raw_predict(bundle,subset[FEATURES])),validation_pinball_loss=pinball(yv,raw_predict(bundle,xv))))
            else:
                for n in [20,40,60,80,100,120]:
                    bundle=fit_core(name,proper,h,override={'n_estimators':n})
                    rows.append(dict(model=name,horizon_months=h,step=n,step_type='Number of trees',training_rows=len(proper),validation_rows=len(val),train_pinball_loss=pinball(yt,raw_predict(bundle,xt)),validation_pinball_loss=pinball(yv,raw_predict(bundle,xv))))
    return pd.DataFrame(rows),pd.DataFrame(training_records)


def save_final_models(frame,selections):
    origin=frame.iloc[-1];forecasts=[];importance=[]
    for h in HORIZONS:
        for name in MODELS:
            bundle=fit_calibrated(frame,name,origin,h)
            bundle['data_sha256']=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
            folder=RUN/'models'/name;folder.mkdir(exist_ok=True)
            joblib.dump(bundle,folder/f'model_c_h{h}.joblib',compress=3)
            q,_=calibrated_predict(bundle,origin[FEATURES].to_numpy(float)[None,:])
            row=dict(model=name,model_label=LABELS[name],horizon_months=h,origin_month=origin.date.strftime('%Y-%m'),
                forecast_date_assumed=origin.forecast_date_assumed.strftime('%Y-%m-%d'),target_month=origin[f'target_month_h{h}'].strftime('%Y-%m'),
                origin_freight_index=float(origin.freight_index),selected_for_horizon=selections[str(h)]['selected_model']==name,
                h12_reporting_only=h==12)
            for i,p in enumerate(QNAMES):
                row[f'{p}_log_change']=float(q[0,i]);row[f'{p}_pct_change']=float(np.expm1(q[0,i])*100);row[f'{p}_freight_index']=float(origin.freight_index*np.exp(q[0,i]))
            forecasts.append(row)
            if name in ['random_forest','extra_trees']:values=bundle['estimator'].feature_importances_;method='Impurity importance'
            elif name=='gradient_boosting':values=bundle['estimators'][3].feature_importances_;method='P50 model impurity importance'
            elif name=='ridge':values=np.abs(bundle['estimator'].named_steps['ridge'].coef_);method='Absolute standardized coefficient; unsigned'
            else:continue
            importance.extend(dict(model=name,horizon_months=h,feature=f,influence=float(v),method=method) for f,v in zip(FEATURES,values))
    return pd.DataFrame(forecasts),pd.DataFrame(importance)


def plot_outputs(predictions,metrics,curves,recovery):
    from plot_feature_importance import create_feature_plots
    create_feature_plots()
    for period in ['validation','disruption_test','post_disruption']:
        for h in HORIZONS:
            fig,axes=plt.subplots(3,2,figsize=(14,12));axes=axes.ravel()
            for ax,name in zip(axes,MODELS):
                part=predictions[(predictions.period==period)&(predictions.horizon_months==h)&(predictions.model==name)].sort_values('target_month')
                x=pd.to_datetime(part.target_month)
                ax.fill_between(x,part.p05_freight_index.to_numpy(),part.p95_freight_index.to_numpy(),alpha=.18,color='#347D78',label='P05-P95 range')
                ax.plot(x,part.actual_freight_index,color='#223A46',linewidth=1.8,label='Actual index')
                ax.plot(x,part.p50_freight_index,color='#CC703A',linewidth=1.5,label='Predicted P50')
                ax.set_title(f'{LABELS[name]} | H{h}',fontsize=11);ax.set_ylabel('Freight index points');ax.tick_params(axis='x',rotation=25,labelsize=8);ax.grid(alpha=.2);ax.legend(fontsize=8)
            axes[-1].axis('off')
            fig.suptitle(f'{period.replace("_"," ").title()}: actual versus predicted target-month index',fontsize=15)
            fig.tight_layout();fig.savefig(RUN/f'visualizations/actual_vs_predicted_{period}_h{h}.png',dpi=140);plt.close(fig)
    for name in MODELS[:-1]:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,h in zip(axes,HORIZONS):
            part=curves[(curves.model==name)&(curves.horizon_months==h)]
            ax.plot(part.step,part.train_pinball_loss,label='In-sample training',color='#347D78')
            ax.plot(part.step,part.validation_pinball_loss,label='Fixed 2015-2017 validation diagnostic',color='#CC703A')
            ax.set_title(f'{LABELS[name]} H{h}',fontsize=11);ax.set_xlabel(part.step_type.iloc[0]);ax.set_ylabel('Raw mean pinball loss');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Training versus validation loss: uncalibrated diagnostics, not test results',fontsize=13)
        fig.tight_layout();fig.savefig(RUN/f'visualizations/train_validation_loss_{name}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for col,h in enumerate([3,6]):
        for row,period in enumerate(['validation','disruption_test']):
            ax=axes[row,col];part=metrics[(metrics.period==period)&(metrics.horizon_months==h)]
            ax.bar(part.model,part.mis_ratio_vs_baseline,color=['#CC703A' if n=='no_change' else '#347D78' for n in part.model])
            ax.axhline(1,linestyle='--',color='#223A46');ax.set_title(f'{period.replace("_"," ")} H{h}');ax.set_ylabel('MIS / baseline MIS (lower is better)');ax.tick_params(axis='x',rotation=30,labelsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(RUN/'visualizations/model_mis_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,h in zip(axes,[3,6]):
        for name in MODELS:
            part=recovery[recovery.model.eq(name)&recovery.horizon_months.eq(h)]
            ax.plot(pd.to_datetime(part.evaluation_asof),part.coverage_6_completed,label=LABELS[name],linewidth=1.3)
        ax.axhline(.90,linestyle='--',color='black');ax.set_ylim(0,1.05);ax.set_title(f'H{h}: last six completed predictions');ax.tick_params(axis='x',rotation=25);ax.set_ylabel('Coverage');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(RUN/'visualizations/disruption_recovery_coverage.png',dpi=150);plt.close(fig)


def dashboard(predictions,metrics,forecasts,selections):
    def img(name):return 'data:image/png;base64,'+base64.b64encode((RUN/'visualizations'/name).read_bytes()).decode()
    show=metrics[metrics.period.isin(['validation','disruption_test','post_disruption'])].copy()
    show['Coverage']=show.coverage_90.map(lambda v:f'{v:.1%}')
    show['MAPE']=show.mape_freight_index.map(lambda v:f'{v:.1f}%')
    show['MIS ratio']=show.mis_ratio_vs_baseline.map(lambda v:f'{v:.3f}')
    show['Acceptance']=np.where(show.model.eq('no_change'),'Reference baseline',np.where(show.h12_reporting_only,'H12 reporting only',np.where(show.probability_gates_pass,'Coverage + MIS pass','Coverage/MIS fail')))
    show['Full disruption gate']=np.where(show.period.eq('disruption_test')&~show.h12_reporting_only&show.model.ne('no_change'),np.where(show.full_disruption_gate_pass,'Pass, including recovery','Fail'),'Not assessed here')
    table=show[['period','model_label','horizon_months','Coverage','MIS ratio','MAPE','missing_forecasts','Acceptance','Full disruption gate']].rename(columns={'period':'Evaluation','model_label':'Model','horizon_months':'Horizon','missing_forecasts':'Missing'}).to_html(index=False,border=0)
    fields=['period','model','model_label','horizon_months','origin_month','target_month','actual_freight_index','p05_freight_index','p50_freight_index','p95_freight_index','actual_log_change','p50_log_change','inside_90_interval']
    payload=json.dumps(json_clean(predictions[fields].to_dict('records')),allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model C freight forecast evaluation</title><style>
    *{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#E7EFED,#F8F5EC);color:#233C48;font:16px Georgia,serif}main{max-width:1300px;margin:auto;padding:32px}h1{font-size:38px}h2{font-size:25px}section{padding:25px;background:#FFFCF6;border:1px solid #CED9D4;margin:24px 0}p{line-height:1.6}.notice{background:#F4E8D9;padding:14px;border-left:4px solid #BF7237}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font:13px Verdana,sans-serif}th,td{padding:11px;border-bottom:1px solid #DFE5E1;text-align:left}th{background:#233C48;color:white;white-space:nowrap}select,button{padding:9px;margin:5px;background:white;border:1px solid #A7B8B1;font:14px Verdana,sans-serif}img{width:100%;height:auto}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{border:1px solid #CED9D4;padding:17px;flex:1;min-width:240px}.card b{display:block;font-size:23px;margin:10px 0}svg{width:100%;height:auto;min-width:640px}@media(max-width:650px){main{padding:12px}section{padding:15px}h1{font-size:28px}}
    </style></head><body><main><h1>Model C: freight-cost movement forecasts</h1><p>Four algorithms and a no-change baseline. Direct H3/H6/H12 forecasts describe changes in the BLS freight index; values shown are index points.</p>'''
    page+='<section><h2>Selected from validation</h2><div class="cards">'+''.join(f'<div class="card">H{h}<b>{LABELS[selections[str(h)]["selected_model"]]}</b>Best ML candidate: {LABELS[selections[str(h)]["best_ml_candidate"]]}</div>' for h in HORIZONS)+'</div><p>Selections use 2015-2019 validation and its pre-2020 known outcomes. H12 does not determine model selection. A baseline selection means no ML candidate met both validation coverage and MIS requirements.</p></section>'
    page+='<section><h2>Data and evaluation</h2><p>Initial history: 2005-2014. Expanding annual validation fits: 2015-2019. Monthly disruption fits: 2020-2021 reference months. Later monitoring: 2022 onward. Every fit uses only targets assumed available then; the latest 24 eligible outcomes calibrate intervals and are held out from fitting the estimator.</p><p class="notice">Research evaluation using assumed publication dates and current historical snapshots. Historical release dates and original revision vintages remain unverified. The required FBX 2020-2021 comparison is still missing. Reference-month tests and the separate specification issue-date tests are saved with explicit labels.</p></section>'
    test=metrics[metrics.period.eq('disruption_test')]
    selected_h3=test[test.model.eq(selections['3']['selected_model'])&test.horizon_months.eq(3)].iloc[0]
    forest_h3=test[test.model.eq('random_forest')&test.horizon_months.eq(3)].iloc[0]
    max_h6=test[test.horizon_months.eq(6)&test.model.ne('no_change')].coverage_90.max()
    page+=f'<section><h2>What the results show</h2><p>The validation-selected H3 model, {LABELS[selections["3"]["selected_model"]]}, achieves {selected_h3.coverage_90:.1%} disruption coverage with MIS {selected_h3.mis_ratio_vs_baseline:.3f} times the baseline. Its test acceptance is reported independently of its validation selection.</p><p>Random Forest is the strongest H3 test candidate: {forest_h3.coverage_90:.1%} coverage, MIS ratio {forest_h3.mis_ratio_vs_baseline:.3f}, and {"passes" if forest_h3.full_disruption_gate_pass else "fails"} the recorded recovery gate. It was not selected by validation, so the test outcome does not replace the frozen recommendation.</p><p>At H6, the highest ML test coverage is {max_h6:.1%}, below the required 90%. No candidate meets the H6 coverage requirement. These results support further research rather than a production acceptance claim.</p></section>'
    page+='<section><h2>Evaluation metrics</h2><p>Coverage should be at least 90%. MIS ratio below 1 means better than the equally calibrated no-change baseline. Missing forecasts count as misses. MAPE measures median index error, while pinball and MIS evaluate the forecast distribution.</p><div class="scroll">'+table+'</div></section>'
    page+='<section><h2>Actual versus predicted</h2><label>Model <select id="model">'+''.join(f'<option value="{n}">{LABELS[n]}</option>' for n in MODELS)+'</select></label><label>Horizon <select id="h"><option value="3">H3</option><option value="6">H6</option><option value="12">H12</option></select></label><label>Period <select id="period"><option value="disruption_test">2020-2021 disruption</option><option value="validation">2015-2019 validation</option><option value="post_disruption">2022 onward</option></select></label><button id="download">Download displayed CSV</button><p id="count"></p><div class="scroll"><svg id="chart" viewBox="0 0 1000 380" role="img" aria-label="Actual and predicted freight index"></svg></div><div class="scroll"><table><thead><tr><th>Origin</th><th>Target</th><th>Actual index</th><th>P50 predicted</th><th>P05</th><th>P95</th><th>Actual inside range?</th></tr></thead><tbody id="rows"></tbody></table></div></section>'
    page+='<section><h2>Interval-score comparison</h2><img alt="Validation and test MIS" src="'+img('model_mis_comparison.png')+'"></section>'
    page+='<section><h2>Feature influence</h2><p>These graphs describe the final fitted models. Ridge uses absolute standardized coefficients, expressed as shares of their total magnitude. Forest models use tree impurity importance; Gradient Boosting shows its P50 estimator only. These measures do not establish causation or forecast accuracy, and correlated features can share influence. The no-change baseline has no learned feature importance.</p>'+''.join('<img alt="'+LABELS[n]+' feature influence" src="'+img(f'feature_importance_{n}.png')+'">' for n in MODELS[:-1])+'</section>'
    page+='<section><h2>Training and validation loss</h2><p>These are uncalibrated mean pinball-loss diagnostics on an initial chronological fit and a fixed 2015-2017 validation window. They do not use the disruption test. Ridge is plotted against training-set size, forests against tree count and Gradient Boosting against actual boosting stages. In-sample loss is diagnostic, not acceptance evidence.</p>'+''.join('<img alt="'+LABELS[n]+' loss" src="'+img(f'train_validation_loss_{n}.png')+'">' for n in MODELS[:-1])+'</section>'
    page+='<section><h2>Recovery of completed-outcome coverage</h2><p>Each point uses the latest six forecasts whose outcomes are assumed published by that date. Recovery credit requires all six target months to be January 2020 or later. With six observations, the 90% threshold requires six of six inside the interval.</p><img alt="Recovery coverage" src="'+img('disruption_recovery_coverage.png')+'"></section>'
    latest=forecasts[forecasts.selected_for_horizon].copy()
    latest['Median movement']=latest.p50_pct_change.map(lambda v:f'{v:+.1f}%')
    latest['90% movement range']=latest.apply(lambda r:f'{r.p05_pct_change:+.1f}% to {r.p95_pct_change:+.1f}%',axis=1)
    page+='<section><h2>Latest forecasts</h2><p>Based on the last supplied observation, August 2026. Its prototype issue date is September 2026. Targets are measured from that reference observation; these are freight-index movements, not route-specific charges in GBP.</p><div class="scroll">'+latest[['model_label','horizon_months','target_month','Median movement','90% movement range']].rename(columns={'model_label':'Model','horizon_months':'Horizon','target_month':'Target month'}).to_html(index=False,border=0)+'</div></section>'
    page+='''<script>const data=PAYLOAD;const el=id=>document.getElementById(id);const fmt=v=>v==null?'N/A':Number(v).toFixed(3);function filtered(){return data.filter(r=>r.model===el('model').value&&r.horizon_months===Number(el('h').value)&&r.period===el('period').value).sort((a,b)=>a.target_month.localeCompare(b.target_month))}
    function draw(){const rows=filtered();el('count').textContent=rows.length+' completed target records. Actual is dark blue, P50 is orange, and shading is the 90% interval.';el('rows').innerHTML=rows.map(r=>'<tr>'+[r.origin_month,r.target_month,fmt(r.actual_freight_index),fmt(r.p50_freight_index),fmt(r.p05_freight_index),fmt(r.p95_freight_index),r.inside_90_interval?'Yes':'No'].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');const valid=rows.filter(r=>r.p50_freight_index!=null);if(!valid.length){el('chart').innerHTML='';return;}const all=valid.flatMap(r=>[r.actual_freight_index,r.p05_freight_index,r.p95_freight_index]);let min=Math.min(...all),max=Math.max(...all);const pad=Math.max((max-min)*.1,1);min-=pad;max+=pad;const x=i=>75+i*870/Math.max(valid.length-1,1),y=v=>320-(v-min)*270/(max-min);const line=key=>valid.map((r,i)=>(i?'L':'M')+x(i)+','+y(r[key])).join(' ');const band=valid.map((r,i)=>x(i)+','+y(r.p05_freight_index)).concat([...valid].reverse().map((r,j)=>x(valid.length-1-j)+','+y(r.p95_freight_index))).join(' ');let svg='<polygon points="'+band+'" fill="#B9D5CF" opacity="0.65"/>';for(let k=0;k<5;k++){const v=min+(max-min)*k/4;svg+='<line x1="75" y1="'+y(v)+'" x2="945" y2="'+y(v)+'" stroke="#CCD6D2"/><text x="65" y="'+(y(v)+4)+'" text-anchor="end" font-size="12">'+v.toFixed(0)+'</text>';}svg+='<path d="'+line('actual_freight_index')+'" fill="none" stroke="#233C48" stroke-width="2"/><path d="'+line('p50_freight_index')+'" fill="none" stroke="#CC703A" stroke-width="2"/>';valid.forEach((r,i)=>{if(i%Math.max(1,Math.ceil(valid.length/7))===0||i===valid.length-1)svg+='<text x="'+x(i)+'" y="345" text-anchor="middle" font-size="12">'+r.target_month+'</text>';});el('chart').innerHTML=svg;}['model','h','period'].forEach(id=>el(id).addEventListener('change',draw));draw();
    el('download').addEventListener('click',()=>{const rows=filtered();if(!rows.length)return;const keys=Object.keys(rows[0]);const quote=v=>'"'+String(v??'').replace(/"/g,'""')+'"';const csv=[keys.map(quote).join(','),...rows.map(r=>keys.map(k=>quote(r[k])).join(','))].join('\\r\\n');const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='model_c_actual_vs_predicted.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)});</script></main></body></html>'''.replace('PAYLOAD',payload)
    (RUN/'visualizations/model_c_training_dashboard.html').write_text(page,encoding='utf-8')


def main():
    for folder in ['models','results','predictions','visualizations']:(RUN/folder).mkdir(parents=True,exist_ok=True)
    if (RUN/'results/run_manifest.json').exists():raise FileExistsError('Completed run already exists; use --render-only to rebuild visuals.')
    frame=load_data()
    validation=validation_predictions(frame)
    validation.to_csv(RUN/'predictions/validation_actual_vs_predicted.csv',index=False)
    validation_metrics=metric_rows(validation)
    validation_metrics.to_csv(RUN/'results/validation_metrics.csv',index=False)
    selections=select_models(validation_metrics)
    later=later_predictions(frame)
    predictions=pd.concat([validation,later],ignore_index=True).sort_values(['model','horizon_months','origin_month'])
    predictions.to_csv(RUN/'predictions/all_out_of_sample_predictions.csv',index=False)
    for period,group in predictions.groupby('period'):group.to_csv(RUN/f'predictions/{period}_actual_vs_predicted.csv',index=False)
    metrics=metric_rows(predictions)
    issue_records=predictions[pd.to_datetime(predictions.forecast_date_assumed).between('2020-01-01','2021-12-31')].copy()
    issue_records['period']='specification_issue_date_test'
    issue_records.to_csv(RUN/'predictions/specification_issue_date_test_actual_vs_predicted.csv',index=False)
    metrics=pd.concat([metrics,metric_rows(issue_records)],ignore_index=True)
    trajectory,recovered=recovery_assessment(predictions)
    trajectory.to_csv(RUN/'results/recovery_coverage_trajectory.csv',index=False)
    recovered.to_csv(RUN/'results/recovery_summary.csv',index=False)
    metrics=metrics.merge(recovered[['model','horizon_months','recovery_within_six_months']],on=['model','horizon_months'],how='left')
    metrics['full_disruption_gate_pass']=metrics.probability_gates_pass & metrics.recovery_within_six_months
    metrics.to_csv(RUN/'results/evaluation_metrics.csv',index=False)
    curves,train_records=diagnostic_curves(frame)
    curves.to_csv(RUN/'results/train_validation_loss_curves.csv',index=False)
    train_records.to_csv(RUN/'predictions/in_sample_training_diagnostic.csv',index=False)
    metric_rows(train_records).to_csv(RUN/'results/in_sample_diagnostic_metrics.csv',index=False)
    forecasts,importance=save_final_models(frame,selections)
    forecasts.to_csv(RUN/'predictions/latest_forecasts_all_models.csv',index=False)
    forecasts[forecasts.selected_for_horizon].to_csv(RUN/'predictions/latest_selected_forecasts.csv',index=False)
    importance.to_csv(RUN/'results/feature_influence.csv',index=False)
    save_json(RUN/'results/model_c_response.json',dict(model='Freight cost movement',source='BLS PCU483111483111',status='research evaluation; publication dates and vintages unverified',
        quantile_levels=QUANTILES.tolist(),h12_reporting_only=True,forecasts=forecasts[forecasts.selected_for_horizon].to_dict('records')))
    save_json(RUN/'results/forecast_errors.json',ERRORS)
    save_json(RUN/'results/run_manifest.json',dict(features=FEATURES,source_csv=str(DATA_PATH),source_sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        seed=SEED,model_parameters=PARAMETERS,calibration_rows=CALIBRATION_ROWS,quantile_levels=QUANTILES,
        selected_models=selections,validation_method='Annual expanding blocks 2015-2019 with targets available before 2020; settings fixed before test',
        test_method='Monthly expanding fits; reference-month 2020-2021 primary user split and separate issue-date 2020-2021 report',
        h12_selection_excluded=True,calibration_method='Chronological held-out CQR with availability purge; same method for model and no-change baseline',
        event_knowledge='COVID spike and its dates are known; event window comes from specification. No settings were selected using its outcomes.',
        limitations=['Publication timing assumed','Current historical data may contain revisions','FBX comparison not provided','Nonexchangeable overlapping time-series targets mean conformal nominal coverage is not guaranteed'],
        loss_curve_definition='Raw pinball diagnostics: Ridge versus training size; forests versus trees; Gradient Boosting versus actual stages; validation diagnostic 2015-2017; never test data'))
    plot_outputs(predictions,metrics,curves,trajectory)
    dashboard(predictions,metrics,forecasts,selections)
    print(metrics[(metrics.period=='disruption_test')&metrics.horizon_months.isin([3,6])][['model','horizon_months','coverage_90','mean_interval_score','mis_ratio_vs_baseline','mape_freight_index','full_disruption_gate_pass']].to_string(index=False),flush=True)
    print('Dashboard:',RUN/'visualizations/model_c_training_dashboard.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');parser.add_argument('--predict-only',action='store_true')
    args=parser.parse_args()
    if args.render_only:
        p=pd.read_csv(RUN/'predictions/all_out_of_sample_predictions.csv');m=pd.read_csv(RUN/'results/evaluation_metrics.csv');c=pd.read_csv(RUN/'results/train_validation_loss_curves.csv');r=pd.read_csv(RUN/'results/recovery_coverage_trajectory.csv')
        f=pd.read_csv(RUN/'predictions/latest_forecasts_all_models.csv');s=json.loads((RUN/'results/selected_models.json').read_text())
        plot_outputs(p,m,c,r);dashboard(p,m,f,s)
    elif args.predict_only:
        frame=load_data();origin=frame.iloc[-1];selection=json.loads((RUN/'results/selected_models.json').read_text())
        for h in HORIZONS:
            name=selection[str(h)]['selected_model'];bundle=joblib.load(RUN/f'models/{name}/model_c_h{h}.joblib')
            q,_=calibrated_predict(bundle,origin[FEATURES].to_numpy(float)[None,:])
            print(json.dumps(dict(model=name,horizon=h,target=origin[f'target_month_h{h}'].strftime('%Y-%m'),quantile_log_changes=q[0].tolist())))
    else:main()
