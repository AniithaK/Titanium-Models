from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent
SOURCE = ROOT / 'Train/model_c_freight_training_2005_2026.csv'
PREPARATION = json.loads((ROOT / 'Train/model_c_preparation_manifest.json').read_text())
FEATURES = PREPARATION['feature_columns']
HORIZONS = [3,6,12]
QUANTILES = np.array([.05,.10,.25,.50,.75,.90,.95])
QNAMES = ['p05','p10','p25','p50','p75','p90','p95']
MODELS = ['recent_ridge','linear_median','shallow_forest','regularized_boosting','no_change']
LABELS = {'recent_ridge':'Recent-weighted Ridge','linear_median':'Regularized linear median',
          'shallow_forest':'Shallow weighted Random Forest','regularized_boosting':'Regularized median boosting',
          'no_change':'Matched no-change baseline'}
WINDOWS = [24,36,48]
SCALE_MODES = ['none','volatility_6m','volatility_12m','hybrid_volatility']
SEED = 42
TRAINING_WINDOW = 120
RECENCY_HALF_LIFE = 60


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,(pd.Timestamp,Path)):return str(value)
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def save_json(path,data):path.write_text(json.dumps(clean(data),indent=2,allow_nan=False),encoding='utf-8')


def original_module():
    spec=importlib.util.spec_from_file_location('original_model_c_review',ROOT/'training_v1/code/train_model_c.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def load_data():
    data=original_module().load_data()
    assert len(FEATURES)==8 and not any('target' in c or 'assumed' in c for c in FEATURES)
    return data


def historical_training(data,origin,h):
    train=data[(data.date<origin.date)&data[f'target_freight_log_change_h{h}'].notna()
               &(data[f'target_available_date_assumed_h{h}']<=origin.forecast_date_assumed)].tail(TRAINING_WINDOW).copy()
    if len(train)<36:raise ValueError('Insufficient past observed labels')
    assert train[f'target_available_date_assumed_h{h}'].max()<=origin.forecast_date_assumed
    return train


def fit_point(name,train,h):
    x=train[FEATURES].to_numpy(float);y=train[f'target_freight_log_change_h{h}'].to_numpy(float)
    age=(train.date.max()-train.date).dt.days.to_numpy()/30.4375
    weights=np.power(.5,age/RECENCY_HALF_LIFE)
    bundle=dict(model=name,horizon_months=h,features=FEATURES,training_rows=len(train),
                training_start=train.date.min().strftime('%Y-%m'),training_end=train.date.max().strftime('%Y-%m'),
                max_training_target_available_date=train[f'target_available_date_assumed_h{h}'].max().strftime('%Y-%m-%d'),
                volatility_floor_6=float(train.freight_volatility_6m.quantile(.10)),
                volatility_floor_12=float(train.freight_volatility_12m.quantile(.10)))
    if name=='no_change':return bundle
    if name in ['recent_ridge','linear_median']:
        scaler=StandardScaler().fit(x,sample_weight=weights)
        estimator=Ridge(alpha=10.) if name=='recent_ridge' else QuantileRegressor(quantile=.5,alpha=.002,solver='highs')
        estimator.fit(scaler.transform(x),y,sample_weight=weights)
        bundle['estimator']=Pipeline([('scale',scaler),('model',estimator)])
    elif name=='shallow_forest':
        estimator=RandomForestRegressor(n_estimators=120,max_depth=4,min_samples_leaf=12,max_features=.7,n_jobs=2,random_state=SEED)
        estimator.fit(x,y,sample_weight=weights);bundle['estimator']=estimator
    else:
        estimator=GradientBoostingRegressor(loss='quantile',alpha=.5,n_estimators=30,max_depth=1,min_samples_leaf=15,learning_rate=.05,random_state=SEED)
        estimator.fit(x,y,sample_weight=weights);bundle['estimator']=estimator
    return bundle


def point_predict(bundle,x):
    return np.zeros(len(x)) if bundle['model']=='no_change' else bundle['estimator'].predict(np.asarray(x,float))


def scales(bundle,origins):
    h=bundle['horizon_months']
    v6=np.maximum(origins.freight_volatility_6m.to_numpy(),max(bundle['volatility_floor_6'],1e-8))
    v12=np.maximum(origins.freight_volatility_12m.to_numpy(),max(bundle['volatility_floor_12'],1e-8))
    return dict(none=np.ones(len(origins)),volatility_6m=v6*np.sqrt(h),volatility_12m=v12*np.sqrt(h),
                hybrid_volatility=np.maximum(v6,v12)*np.sqrt(h))


def build_raw_forecasts(data):
    records=[]
    for h in HORIZONS:
        origins=data[data.date>='2009-01-01']
        for name in MODELS:
            cached=None;last_block=None
            for i,(_,origin) in enumerate(origins.iterrows()):
                # Quarterly refits before 2020; monthly refits during and after tests.
                block=origin.date.to_period('Q') if origin.date.year<2020 else origin.date.to_period('M')
                if cached is None or block!=last_block:
                    train=historical_training(data,origin,h);cached=fit_point(name,train,h)
                    cached['fit_asof']=origin.forecast_date_assumed.strftime('%Y-%m-%d');last_block=block
                current=data[data.date==origin.date]
                centre=float(point_predict(cached,current[FEATURES].to_numpy())[0])
                scale=scales(cached,current)
                row=dict(model=name,horizon_months=h,origin_month=origin.date.strftime('%Y-%m'),
                    forecast_date_assumed=origin.forecast_date_assumed.strftime('%Y-%m-%d'),
                    target_month=origin[f'target_month_h{h}'].strftime('%Y-%m'),
                    target_available_date_assumed=origin[f'target_available_date_assumed_h{h}'].strftime('%Y-%m-%d'),
                    origin_freight_index=float(origin.freight_index),actual_log_change=origin[f'target_freight_log_change_h{h}'],
                    actual_freight_index=origin[f'target_freight_index_h{h}'],raw_point_log_change=centre,
                    training_rows=cached['training_rows'],model_fit_asof=cached['fit_asof'],
                    max_training_target_available_date=cached['max_training_target_available_date'])
                row.update({f'scale_{key}':float(value[0]) for key,value in scale.items()})
                records.append(row)
                if i%36==0:print(f'Raw rolling forecast {name} H{h} {origin.date:%Y-%m}',flush=True)
            pd.DataFrame(records).to_csv(RUN/'predictions/raw_forecasts_checkpoint.csv',index=False)
    raw=pd.DataFrame(records)
    raw.to_csv(RUN/'predictions/raw_out_of_sample_forecasts.csv',index=False)
    return raw


def ordered_error_quantiles(errors):
    errors=np.sort(np.asarray(errors,float));n=len(errors);out=[]
    for q in QUANTILES:
        if q==.5:out.append(float(np.median(errors)));continue
        order=math.floor((n+1)*q) if q<.5 else math.ceil((n+1)*q)
        out.append(float(errors[max(1,min(n,order))-1]))
    return np.asarray(out)


def predict_from_history(current,history,window,mode):
    pool=history[(history.origin_month<current.origin_month)&history.actual_log_change.notna()
                 &(history.target_available_date_assumed<=current.forecast_date_assumed)].tail(window)
    if len(pool)<window:raise ValueError('Missing required completed calibration forecasts')
    assert (pool.model_fit_asof<pool.target_available_date_assumed).all()
    errors=(pool.actual_log_change.to_numpy()-pool.raw_point_log_change.to_numpy())/pool[f'scale_{mode}'].to_numpy()
    quantiles=float(current.raw_point_log_change)+float(current[f'scale_{mode}'])*ordered_error_quantiles(errors)
    if current.model=='no_change':
        quantiles[:3]=np.minimum(quantiles[:3],0);quantiles[3]=0.;quantiles[4:]=np.maximum(quantiles[4:],0)
    assert np.isfinite(quantiles).all() and (np.diff(quantiles)>=-1e-12).all()
    return quantiles,pool


def replay(raw,data,name,h,window,mode,validation_only=False):
    history=raw[(raw.model==name)&(raw.horizon_months==h)].sort_values('origin_month')
    dates=data.set_index(data.date.dt.strftime('%Y-%m'))
    rows=[]
    for _,current in history[history.origin_month>='2015-01'].iterrows():
        source=dates.loc[current.origin_month]
        eligible=bool(source[f'validation_selection_eligible_h{h}'])
        if validation_only and not eligible:continue
        if pd.isna(current.actual_log_change):continue
        period='validation' if eligible else 'pretest_bridge' if current.origin_month<'2020-01' else 'disruption_test' if current.origin_month<='2021-12' else 'post_disruption'
        quantiles,pool=predict_from_history(current,history,window,mode)
        row=current.to_dict()
        row.update(period=period,model_label=LABELS[name],calibration_window=window,scale_mode=mode,
            calibration_rows=len(pool),calibration_latest_target_available_date=pool.target_available_date_assumed.max(),
            calibration_earliest_origin=pool.origin_month.min(),calibration_latest_origin=pool.origin_month.max(),
            inside_90_interval=bool(quantiles[0]<=current.actual_log_change<=quantiles[-1]),forecast_produced=True)
        for i,p in enumerate(QNAMES):
            row[f'{p}_log_change']=float(quantiles[i]);row[f'{p}_freight_index']=float(current.origin_freight_index*np.exp(quantiles[i]));row[f'{p}_pct_change']=float(np.expm1(quantiles[i])*100)
        row['predicted_freight_index']=row['p50_freight_index']
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_for(part):
    q=part[[f'{p}_log_change' for p in QNAMES]].to_numpy();y=part.actual_log_change.to_numpy()
    valid=np.isfinite(q).all(axis=1);residual=y[:,None]-q
    mis=(q[:,-1]-q[:,0])+20*(q[:,0]-y)*(y<q[:,0])+20*(y-q[:,-1])*(y>q[:,-1])
    actual=part.actual_freight_index.to_numpy();pred=part.p50_freight_index.to_numpy()
    return dict(expected_forecasts=len(part),missing_forecasts=int((~valid).sum()),coverage_90=float(np.mean(valid&(y>=q[:,0])&(y<=q[:,-1]))),
        mean_interval_score=float(mis.mean()),pinball_loss=float(np.maximum(QUANTILES*residual,(QUANTILES-1)*residual).mean()),
        mape_freight_index=float(np.mean(abs(actual-pred)/actual)*100),mae_freight_index=float(np.mean(abs(actual-pred))),
        rmse_freight_index=float(np.sqrt(np.mean((actual-pred)**2))),interval_width_log_change=float(np.mean(q[:,-1]-q[:,0])))


def select_validation(raw,data):
    comparisons=[];candidate_configs={};selections={};cache={}
    for h in [3,6]:
        for name in MODELS:
            for window in WINDOWS:
                for mode in SCALE_MODES:
                    records=replay(raw,data,name,h,window,mode,True)
                    cache[(name,h,window,mode)]=records
                    comparisons.append(dict(model=name,horizon_months=h,calibration_window=window,scale_mode=mode,**metrics_for(records)))
        rows=pd.DataFrame([r for r in comparisons if r['horizon_months']==h])
        baseline=rows[rows.model=='no_change']
        qualified=baseline[baseline.coverage_90>=.90]
        best_base=(qualified if len(qualified) else baseline).sort_values(['mean_interval_score','pinball_loss']).iloc[0]
        baseline_config=dict(window=int(best_base.calibration_window),mode=best_base.scale_mode)
        candidate_configs[('no_change',h)]=baseline_config
        best_models=[]
        for name in MODELS[:-1]:
            group=rows[rows.model==name].copy()
            paired=baseline[['calibration_window','scale_mode','mean_interval_score']].rename(columns={'mean_interval_score':'matched_baseline_mis'})
            group=group.merge(paired,on=['calibration_window','scale_mode'])
            group['best_baseline_mis']=float(best_base.mean_interval_score)
            group['validation_acceptance']=(group.coverage_90>=.90)&(group.mean_interval_score<group.matched_baseline_mis)&(group.mean_interval_score<group.best_baseline_mis)
            eligible=group[group.validation_acceptance]
            best=(eligible if len(eligible) else group).sort_values(['mean_interval_score','pinball_loss']).iloc[0]
            candidate_configs[(name,h)]=dict(window=int(best.calibration_window),mode=best.scale_mode)
            best_models.append(best.to_dict())
        candidates=pd.DataFrame(best_models)
        accepted=candidates[candidates.validation_acceptance]
        selected=accepted.sort_values('mean_interval_score').iloc[0].model if len(accepted) else 'no_change'
        best_ml=candidates.sort_values('mean_interval_score').iloc[0].model
        selections[str(h)]=dict(selected_model=selected,best_ml_candidate=best_ml,
            configuration=candidate_configs[(selected,h)],best_no_change_configuration=baseline_config,
            validation_acceptance=selected!='no_change',reference_baseline_mis=float(best_base.mean_interval_score))
        print(f'Frozen H{h} selection: {selections[str(h)]}',flush=True)
    # All H12 choices inherit H6 settings; H12 outcomes never enter selection.
    scores={name:np.mean([metrics_for(cache[(name,h,candidate_configs[(name,h)]['window'],candidate_configs[(name,h)]['mode'])])['mean_interval_score']/selections[str(h)]['reference_baseline_mis'] for h in [3,6]]) for name in MODELS[:-1]}
    global_best=min(scores,key=scores.get)
    selected_h12=global_best if any(selections[str(h)]['selected_model']!='no_change' for h in [3,6]) else 'no_change'
    for name in MODELS:candidate_configs[(name,12)]=candidate_configs[(name,6)].copy()
    selections['12']=dict(selected_model=selected_h12,best_ml_candidate=global_best,configuration=candidate_configs[(selected_h12,12)],
        best_no_change_configuration=candidate_configs[('no_change',12)],h12_reporting_only=True,
        selection_basis='Model family from H3/H6 validation average MIS ratios; calibration settings borrowed from H6; no H12 selection.')
    pd.DataFrame(comparisons).to_csv(RUN/'results/validation_configuration_comparison.csv',index=False)
    save_json(RUN/'results/selected_models.json',selections)
    save_json(RUN/'results/candidate_calibration_settings.json',{f'{name}_h{h}':conf for (name,h),conf in candidate_configs.items()})
    return candidate_configs,selections


def evaluate(raw,data,configs):
    forecasts=[];matched=[]
    for h in HORIZONS:
        for name in MODELS:
            conf=configs[(name,h)]
            part=replay(raw,data,name,h,conf['window'],conf['mode'])
            forecasts.append(part)
            paired=replay(raw,data,'no_change',h,conf['window'],conf['mode'])
            paired['compared_model']=name;matched.append(paired)
    records=pd.concat(forecasts,ignore_index=True)
    matched_records=pd.concat(matched,ignore_index=True)
    results=[]
    for (period,name,h),part in records.groupby(['period','model','horizon_months']):
        base=records[(records.period==period)&(records.model=='no_change')&(records.horizon_months==h)]
        paired=matched_records[(matched_records.period==period)&(matched_records.compared_model==name)&(matched_records.horizon_months==h)]
        score=metrics_for(part);bscore=metrics_for(base);pscore=metrics_for(paired)
        score.update(period=period,model=name,model_label=LABELS[name],horizon_months=int(h),
            baseline_mis=bscore['mean_interval_score'],matched_baseline_mis=pscore['mean_interval_score'],
            mis_ratio_vs_baseline=score['mean_interval_score']/bscore['mean_interval_score'],
            mis_ratio_vs_matched_baseline=score['mean_interval_score']/pscore['mean_interval_score'],
            coverage_pass=score['coverage_90']>=.90,
            probability_gates_pass=score['coverage_90']>=.90 and score['mean_interval_score']<bscore['mean_interval_score'] and score['mean_interval_score']<pscore['mean_interval_score'],
            h12_reporting_only=h==12)
        results.append(score)
    return records,matched_records,pd.DataFrame(results)


def save_models(data,raw,configs,selections):
    origin=data.iloc[-1];forecasts=[];influence=[]
    for h in HORIZONS:
        train=historical_training(data,origin,h)
        for name in MODELS:
            bundle=fit_point(name,train,h)
            bundle['fit_asof']=origin.forecast_date_assumed.strftime('%Y-%m-%d')
            conf=configs[(name,h)]
            history=raw[(raw.model==name)&(raw.horizon_months==h)].sort_values('origin_month')
            current=history[history.origin_month==origin.date.strftime('%Y-%m')].iloc[0].copy()
            current.raw_point_log_change=float(point_predict(bundle,origin[FEATURES].to_numpy(float)[None,:])[0])
            for key,value in scales(bundle,data.tail(1)).items():current[f'scale_{key}']=float(value[0])
            quantiles,pool=predict_from_history(current,history,conf['window'],conf['mode'])
            bundle.update(calibration_configuration=conf,calibration_history=pool.to_dict('records'),
                quantile_levels=QUANTILES.tolist(),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                calibration_method='Finite-order quantiles of completed out-of-sample residuals, optionally normalized by observed freight volatility')
            joblib.dump(bundle,RUN/f'models/{name}_h{h}.joblib',compress=3)
            row=dict(model=name,horizon_months=h,origin_month=origin.date.strftime('%Y-%m'),
                forecast_date_assumed=origin.forecast_date_assumed.strftime('%Y-%m-%d'),target_month=origin[f'target_month_h{h}'].strftime('%Y-%m'),
                origin_freight_index=float(origin.freight_index),selected_for_horizon=selections[str(h)]['selected_model']==name)
            for i,p in enumerate(QNAMES):
                row[f'{p}_log_change']=float(quantiles[i]);row[f'{p}_freight_index']=float(origin.freight_index*np.exp(quantiles[i]));row[f'{p}_pct_change']=float(np.expm1(quantiles[i])*100)
            forecasts.append(row)
            if name in ['recent_ridge','linear_median']:values=np.abs(bundle['estimator'].named_steps['model'].coef_);method='Absolute weighted-standardized coefficient'
            elif name!='no_change':values=bundle['estimator'].feature_importances_;method='Tree impurity importance'
            else:continue
            influence.extend(dict(model=name,horizon_months=h,feature=f,influence=float(v),method=method) for f,v in zip(FEATURES,values))
    return pd.DataFrame(forecasts),pd.DataFrame(influence)


def compare_original(metrics,selections):
    old=pd.read_csv(ROOT/'training_v1/results/evaluation_metrics.csv')
    old_selections=json.loads((ROOT/'training_v1/results/selected_models.json').read_text())
    rows=[]
    for h in HORIZONS:
        for period in ['validation','disruption_test','post_disruption']:
            before=old[(old.model==old_selections[str(h)]['selected_model'])&(old.horizon_months==h)&(old.period==period)].iloc[0]
            after=metrics[(metrics.model==selections[str(h)]['selected_model'])&(metrics.horizon_months==h)&(metrics.period==period)].iloc[0]
            rows.append(dict(period=period,horizon_months=h,previous_model=before.model,new_selected_model=after.model,
                previous_coverage=before.coverage_90,new_coverage=after.coverage_90,previous_mis=before.mean_interval_score,new_mis=after.mean_interval_score,
                previous_mape=before.mape_freight_index,new_mape=after.mape_freight_index,
                mis_change_percent=100*(after.mean_interval_score/before.mean_interval_score-1),h12_reporting_only=h==12))
    return pd.DataFrame(rows)


def render():
    records=pd.read_csv(RUN/'predictions/all_predictions.csv')
    metrics=pd.read_csv(RUN/'results/evaluation_metrics.csv')
    comparison=pd.read_csv(RUN/'results/previous_vs_improved_selected.csv')
    selections=json.loads((RUN/'results/selected_models.json').read_text())
    future=pd.read_csv(RUN/'predictions/latest_forecasts.csv')
    for h in [3,6,12]:
        fig,axes=plt.subplots(3,2,figsize=(14,12));axes=axes.ravel()
        for ax,name in zip(axes,MODELS):
            part=records[records.period.eq('disruption_test')&records.model.eq(name)&records.horizon_months.eq(h)].sort_values('target_month')
            x=pd.to_datetime(part.target_month)
            ax.fill_between(x,part.p05_freight_index,part.p95_freight_index,color='#347D78',alpha=.2,label='P05-P95')
            ax.plot(x,part.actual_freight_index,color='#243B48',label='Actual');ax.plot(x,part.p50_freight_index,color='#CC703A',label='Predicted median')
            ax.set_title(f'{LABELS[name]} H{h}',fontsize=10);ax.set_ylabel('Freight index points');ax.tick_params(axis='x',rotation=25,labelsize=8);ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[-1].axis('off');fig.suptitle('Reused 2020-2021 disruption test: target-month actual and prediction',fontsize=14)
        fig.tight_layout();fig.savefig(RUN/f'visualizations/actual_vs_predicted_test_h{h}.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,h in zip(axes,[3,6]):
        part=comparison[comparison.horizon_months.eq(h)]
        x=np.arange(len(part));ax.bar(x-.18,part.previous_mis,.36,label='Previous selected model',color='#8B9CA6');ax.bar(x+.18,part.new_mis,.36,label='New validation selection',color='#347D78')
        ax.set_xticks(x,part.period.str.replace('_',' '),rotation=20);ax.set_title(f'H{h} MIS (lower is better)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(RUN/'visualizations/previous_vs_new_mis.png',dpi=150);plt.close(fig)
    def img(name):return 'data:image/png;base64,'+base64.b64encode((RUN/'visualizations'/name).read_bytes()).decode()
    table=metrics[metrics.period.isin(['validation','disruption_test','post_disruption'])].copy()
    table['Coverage']=table.coverage_90.map(lambda v:f'{v:.1%}');table['Median error']=table.mape_freight_index.map(lambda v:f'{v:.2f}%')
    table['MIS / matched baseline']=table.mis_ratio_vs_matched_baseline.map(lambda v:f'{v:.3f}')
    table['MIS / best baseline']=table.mis_ratio_vs_baseline.map(lambda v:f'{v:.3f}')
    table['Gate']=np.where(table.model.eq('no_change'),'Baseline reference',np.where(table.h12_reporting_only,'Reporting only',np.where(table.probability_gates_pass,'Coverage + MIS pass','Coverage/MIS fail')))
    table_html=table[['period','model_label','horizon_months','Coverage','Median error','MIS / matched baseline','MIS / best baseline','Gate']].rename(columns={'period':'Evaluation','model_label':'Model','horizon_months':'Horizon'}).to_html(index=False,border=0)
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model C improvement study</title><style>*{box-sizing:border-box}body{margin:0;background:linear-gradient(125deg,#E9EFED,#F7F4EA);color:#233C48;font:16px Georgia,serif}main{max-width:1250px;margin:auto;padding:30px}h1{font-size:36px}h2{font-size:25px}section{padding:23px;background:#FFFCF5;border:1px solid #CCD6D1;margin:22px 0}p{line-height:1.6}.notice{padding:14px;background:#F4E7D6;border-left:4px solid #BA7238}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font:13px Verdana,sans-serif}th,td{padding:10px;border-bottom:1px solid #DCE4DF;text-align:left}th{background:#233C48;color:white;white-space:nowrap}img{width:100%;height:auto}select,button{font:14px Verdana,sans-serif;padding:9px;margin:5px;border:1px solid #9CAEA4;background:white}.cards{display:flex;flex-wrap:wrap;gap:12px}.cards div{flex:1;min-width:220px;padding:17px;border:1px solid #CCD6D1}.cards b{display:block;font-size:22px;margin-top:9px}svg{width:100%;min-width:600px}@media(max-width:650px){main{padding:12px}section{padding:14px}h1{font-size:28px}}</style></head><body><main><h1>Model C: improved freight forecasts</h1>'''
    page+='<section><h2>Frozen validation choices</h2><div class="cards">'+''.join(f'<div>H{h}<b>{LABELS[selections[str(h)]["selected_model"]]}</b>{selections[str(h)]["configuration"]["window"]} completed errors; {selections[str(h)]["configuration"]["mode"]}</div>' for h in HORIZONS)+'</div></section>'
    page+='<section><h2>What changed</h2><p>Estimators now fit all eligible recent history, rather than discarding the latest 24 labels for a calibration holdout. The last ten years are used with greater weight on recent observations. Intervals come from past out-of-sample forecast errors; pre-2020 validation chooses the error window and whether observed freight volatility scales those errors. The matched no-change baseline uses the identical calibration rule.</p><p class="notice">The 2020-2021 and later periods were already inspected during the previous study. These are reused historical tests, not a new untouched holdout. Actual release dates and historical revision vintages remain unverified. No new source observations or predictor columns were invented.</p></section>'
    h3=comparison[comparison.period.eq('disruption_test')&comparison.horizon_months.eq(3)].iloc[0]
    linear_test=metrics[metrics.period.eq('disruption_test')&metrics.model.eq('linear_median')&metrics.horizon_months.eq(6)].iloc[0]
    linear_post=metrics[metrics.period.eq('post_disruption')&metrics.model.eq('linear_median')&metrics.horizon_months.eq(6)].iloc[0]
    page+=f'<section><h2>Outcome: partial gains, no replacement promoted</h2><p>The new H3 selection improves validation MIS, but disruption coverage falls from {h3.previous_coverage:.1%} to {h3.new_coverage:.1%}. Its disruption MIS is worse. Retain the original study as the main reference.</p><p>The new linear H6 candidate has disruption median error {linear_test.mape_freight_index:.2f}% and coverage {linear_test.coverage_90:.1%}. That is a useful improvement in the central forecast, but coverage remains below 90% and it does not beat its matched baseline MIS. Later median error rises to {linear_post.mape_freight_index:.2f}%, so this improvement is not consistent across periods.</p><p>No new candidate passes all H3/H6 historical probability gates. The optional RMS calibration extension did not improve the frozen selections. These artifacts are research experiments, not recommended production replacements.</p></section>'
    page+='<section><h2>Evaluation</h2><p>Coverage must reach 90%; MIS should be below both the equally calibrated matched baseline and the strongest baseline selected using validation. A coverage + MIS pass does not include the separate recovery requirement. H12 is reporting-only.</p><div class="scroll">'+table_html+'</div></section>'
    page+='<section><h2>Previous versus new validation-selected models</h2><img alt="MIS comparison" src="'+img('previous_vs_new_mis.png')+'"><div class="scroll">'+comparison.to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')+'</div></section>'
    page+='<section><h2>Actual versus predicted records</h2><label>Model <select id="model">'+''.join(f'<option value="{n}">{LABELS[n]}</option>' for n in MODELS)+'</select></label><label>Horizon <select id="h"><option value="3">H3</option><option value="6">H6</option><option value="12">H12</option></select></label><label>Period <select id="period"><option value="disruption_test">2020-2021</option><option value="validation">Validation</option><option value="post_disruption">Later monitoring</option></select></label><button id="download">Download filtered CSV</button><p id="count"></p><div class="scroll"><svg id="chart" viewBox="0 0 1000 350"></svg></div><div class="scroll"><table><thead><tr><th>Origin</th><th>Target</th><th>Actual index</th><th>Predicted P50</th><th>P05</th><th>P95</th><th>Inside range</th></tr></thead><tbody id="rows"></tbody></table></div></section>'
    page+=''.join('<section><h2>H'+str(h)+' disruption comparisons</h2><img alt="Actual versus predicted H'+str(h)+'" src="'+img(f'actual_vs_predicted_test_h{h}.png')+'"></section>' for h in HORIZONS)
    page+='<section><h2>Latest selected forecasts</h2><p>Based on August 2026 observations. These are changes in a freight index, not route-specific shipping charges in GBP.</p><div class="scroll">'+future[future.selected_for_horizon].to_html(index=False,border=0,float_format=lambda v:f'{v:.4f}')+'</div></section>'
    fields=['model','horizon_months','period','origin_month','target_month','actual_freight_index','p50_freight_index','p05_freight_index','p95_freight_index','inside_90_interval']
    payload=json.dumps(clean(records[fields].to_dict('records')),allow_nan=False).replace('</','<\\/')
    page+='''<script>const data=PAYLOAD;const el=id=>document.getElementById(id);const fmt=v=>Number(v).toFixed(3);function filtered(){return data.filter(r=>r.model===el('model').value&&r.horizon_months===Number(el('h').value)&&r.period===el('period').value).sort((a,b)=>a.target_month.localeCompare(b.target_month))}function draw(){const rows=filtered();el('count').textContent=rows.length+' forecasts. Actual is dark blue, predicted median is orange; shading shows P05-P95.';el('rows').innerHTML=rows.map(r=>'<tr>'+[r.origin_month,r.target_month,fmt(r.actual_freight_index),fmt(r.p50_freight_index),fmt(r.p05_freight_index),fmt(r.p95_freight_index),r.inside_90_interval?'Yes':'No'].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');if(!rows.length)return;const all=rows.flatMap(r=>[r.actual_freight_index,r.p05_freight_index,r.p95_freight_index]);let low=Math.min(...all),high=Math.max(...all);const pad=Math.max((high-low)*.1,1);low-=pad;high+=pad;const x=i=>70+i*850/Math.max(rows.length-1,1),y=v=>285-(v-low)*240/(high-low);const line=k=>rows.map((r,i)=>(i?'L':'M')+x(i)+','+y(r[k])).join(' ');const band=rows.map((r,i)=>x(i)+','+y(r.p05_freight_index)).concat([...rows].reverse().map((r,j)=>x(rows.length-1-j)+','+y(r.p95_freight_index))).join(' ');let svg='<polygon points="'+band+'" fill="#BCD8D1"/>';for(let j=0;j<5;j++){const v=low+(high-low)*j/4;svg+='<line x1="70" x2="920" y1="'+y(v)+'" y2="'+y(v)+'" stroke="#CCD6D1"/><text x="60" y="'+(y(v)+4)+'" text-anchor="end" font-size="12">'+v.toFixed(0)+'</text>';}svg+='<path d="'+line('actual_freight_index')+'" fill="none" stroke="#233C48" stroke-width="2"/><path d="'+line('p50_freight_index')+'" fill="none" stroke="#CC703A" stroke-width="2"/>';rows.forEach((r,i)=>{if(i%Math.max(1,Math.ceil(rows.length/7))===0||i===rows.length-1)svg+='<text x="'+x(i)+'" y="310" text-anchor="middle" font-size="12">'+r.target_month+'</text>';});el('chart').innerHTML=svg;}['model','h','period'].forEach(id=>el(id).addEventListener('change',draw));draw();el('download').onclick=()=>{const rows=filtered();if(!rows.length)return;const keys=Object.keys(rows[0]),q=v=>'"'+String(v??'').replace(/"/g,'""')+'"';const csv=[keys.map(q).join(','),...rows.map(r=>keys.map(k=>q(r[k])).join(','))].join('\\r\\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='improved_model_c_predictions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script></main></body></html>'''.replace('PAYLOAD',payload)
    page=page.replace('<h1>Model C: improved freight forecasts</h1>','<h1>Model C: freight forecast improvement study</h1>')
    (RUN/'visualizations/improved_model_c_dashboard.html').write_text(page,encoding='utf-8')


def main():
    for folder in ['models','predictions','results','visualizations']:(RUN/folder).mkdir(parents=True,exist_ok=True)
    if (RUN/'results/run_manifest.json').exists():raise FileExistsError('Completed experiment exists; use --render-only')
    data=load_data();raw=build_raw_forecasts(data)
    configs,selections=select_validation(raw,data)
    records,matched,metrics=evaluate(raw,data,configs)
    records.to_csv(RUN/'predictions/all_predictions.csv',index=False);matched.to_csv(RUN/'predictions/matched_baseline_predictions.csv',index=False)
    for period,part in records.groupby('period'):part.to_csv(RUN/f'predictions/{period}_actual_vs_predicted.csv',index=False)
    old=original_module()
    recovery,recovery_summary=old.recovery_assessment(records)
    recovery.to_csv(RUN/'results/recovery_trajectory.csv',index=False);recovery_summary.to_csv(RUN/'results/recovery_summary.csv',index=False)
    metrics=metrics.merge(recovery_summary[['model','horizon_months','recovery_within_six_months']],on=['model','horizon_months'])
    metrics['full_disruption_gate_pass']=metrics.probability_gates_pass & metrics.recovery_within_six_months
    metrics.to_csv(RUN/'results/evaluation_metrics.csv',index=False)
    issue=records[pd.to_datetime(records.forecast_date_assumed).between('2020-01-01','2021-12-31')].copy()
    issue['period']='specification_issue_date_test';issue.to_csv(RUN/'predictions/specification_issue_date_test.csv',index=False)
    comparison=compare_original(metrics,selections);comparison.to_csv(RUN/'results/previous_vs_improved_selected.csv',index=False)
    forecasts,influence=save_models(data,raw,configs,selections)
    forecasts.to_csv(RUN/'predictions/latest_forecasts.csv',index=False);influence.to_csv(RUN/'results/feature_influence.csv',index=False)
    save_json(RUN/'results/model_c_response.json',dict(model='Freight-cost movement',version='recent_models_oos_volatility_calibration',status='Research experiment; reused historical tests and unverified publication/vintage dates',
        quantile_levels=QUANTILES.tolist(),h12_reporting_only=True,forecasts=forecasts[forecasts.selected_for_horizon].to_dict('records')))
    save_json(RUN/'results/run_manifest.json',dict(source_csv=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),features=FEATURES,
        models=LABELS,seed=SEED,training_window_months=TRAINING_WINDOW,recency_half_life_months=RECENCY_HALF_LIFE,
        calibration_windows=WINDOWS,calibration_scale_options=SCALE_MODES,selections=selections,
        calibration='Completed historical out-of-sample standardized residual order quantiles; median bias adjustment for ML; baseline median exactly zero',
        validation='2015-2019 eligible origins, all selection labels available before 2020; quarterly model refits and sequential calibration',
        test='Monthly refits 2020 onward; choices frozen from validation. Tests are reused from previous research.',
        h12_selection_excluded=True,limitations=['Availability dates assumed','Historical revision vintages not reconstructed','FBX comparison absent','Test periods previously examined','Time-series overlap prevents a guaranteed conformal coverage claim']))
    render()
    print(comparison.to_string(index=False),flush=True)
    print(metrics[(metrics.period=='disruption_test')&metrics.horizon_months.isin([3,6])][['model','horizon_months','coverage_90','mean_interval_score','mis_ratio_vs_matched_baseline','mape_freight_index','full_disruption_gate_pass']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    render() if args.render_only else main()
