"""Plot verified saved model influence without retraining or changing forecasts."""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
MODELS = ['ridge','random_forest','extra_trees','gradient_boosting']
LABELS = {'ridge':'Ridge + residual quantiles','random_forest':'Quantile Random Forest',
          'extra_trees':'Quantile Extra Trees','gradient_boosting':'Gradient Boosting quantiles'}
FRIENDLY = {
    **{f'freight_log_change_{lag}m':f'Freight change: {lag} month'+('s' if lag>1 else '') for lag in [1,2,3,6,12]},
    'freight_volatility_6m':'Freight volatility: 6 months',
    'freight_volatility_12m':'Freight volatility: 12 months',
    'brent_log_change_1m':'Brent change: 1 month',
}
COLORS = {'ridge':'#355C7D','random_forest':'#347D78','extra_trees':'#78843F','gradient_boosting':'#CC703A'}


def create_feature_plots():
    data = pd.read_csv(RUN/'results/feature_influence.csv')
    selections = json.loads((RUN/'results/selected_models.json').read_text())
    assert len(data)==96 and not data.duplicated(['model','horizon_months','feature']).any()
    assert set(data.feature)==set(FRIENDLY)
    normalized=[]
    for model in MODELS:
        fig,axes=plt.subplots(1,3,figsize=(18,6.2))
        for ax,h in zip(axes,[3,6,12]):
            part=data[(data.model==model)&(data.horizon_months==h)].copy()
            bundle=joblib.load(RUN/f'models/{model}/model_c_h{h}.joblib')
            if model=='ridge':
                actual=np.abs(bundle['estimator'].named_steps['ridge'].coef_)
                xlabel='Relative coefficient magnitude (%, unsigned)'
            elif model=='gradient_boosting':
                actual=bundle['estimators'][3].feature_importances_
                xlabel='P50 tree impurity importance (%)'
            else:
                actual=bundle['estimator'].feature_importances_
                xlabel='Tree impurity importance (%)'
            recorded=part.set_index('feature').loc[bundle['features'],'influence'].to_numpy()
            assert np.allclose(recorded,actual,rtol=1e-10,atol=1e-12)
            total=float(part.influence.sum())
            part['relative_influence_pct']=100*part.influence/total if total>0 else 0.0
            assert total==0 or abs(part.relative_influence_pct.sum()-100)<1e-9
            part['feature_description']=part.feature.map(FRIENDLY)
            normalized.extend(part.to_dict('records'))
            part=part.sort_values('relative_influence_pct')
            bars=ax.barh(part.feature_description,part.relative_influence_pct,color=COLORS[model],height=.65)
            largest=float(part.relative_influence_pct.max())
            ax.set_xlim(0,max(largest*1.25,5));ax.set_xlabel(xlabel,fontsize=9)
            selected=selections[str(h)]['selected_model']==model
            title=f'H{h}'+(' - reporting only' if h==12 else '')
            if selected:title+='\nSelected from H3/H6 validation'
            ax.set_title(title,fontsize=11);ax.tick_params(axis='y',labelsize=9)
            ax.set_axisbelow(True);ax.grid(axis='x',alpha=.20)
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
            for bar,value in zip(bars,part.relative_influence_pct):
                ax.text(value+max(largest*.02,.1),bar.get_y()+bar.get_height()/2,f'{value:.1f}%',va='center',fontsize=9)
        if model=='ridge':
            explanation='Absolute coefficients after training-fold standardization, normalized within each horizon. Sign is not shown.'
        elif model=='gradient_boosting':
            explanation='Impurity importance from the fitted P50 quantile estimator. This does not describe every quantile or the calibration adjustment.'
        else:
            explanation='Impurity importance from the fitted forest. Correlated inputs can share influence; scores do not measure causal effects.'
        fig.suptitle(f'Model C feature influence - {LABELS[model]}',fontsize=16,y=.98)
        fig.text(.5,.92,explanation,ha='center',fontsize=10)
        fig.text(.5,.02,'Final fitted models. Larger bars mean more reported influence within that model; the percentages are not probabilities of forecast correctness.',ha='center',fontsize=10)
        fig.tight_layout(rect=[0,.07,1,.87],w_pad=2.0)
        destination=RUN/f'visualizations/feature_importance_{model}.png'
        fig.savefig(destination,dpi=160,bbox_inches='tight');plt.close(fig)
        print(f'Saved {destination.name}')
    pd.DataFrame(normalized).to_csv(RUN/'results/feature_importance_normalized.csv',index=False)
    print('Verified all 96 saved scores against their fitted model bundles.')


def refresh_dashboard():
    from train_model_c import dashboard
    predictions=pd.read_csv(RUN/'predictions/all_out_of_sample_predictions.csv')
    metrics=pd.read_csv(RUN/'results/evaluation_metrics.csv')
    forecasts=pd.read_csv(RUN/'predictions/latest_forecasts_all_models.csv')
    selections=json.loads((RUN/'results/selected_models.json').read_text())
    dashboard(predictions,metrics,forecasts,selections)
    print('Dashboard updated; no retraining performed.')


if __name__=='__main__':
    create_feature_plots()
    refresh_dashboard()
