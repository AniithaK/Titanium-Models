"""Verify package-local Model C inputs and reproduce saved forecast values."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import train_model_c as model


def main():
    run = Path(__file__).resolve().parents[1]
    data = model.load_data()
    assert model.DATA_PATH.parent == run / 'data'
    origin = data.iloc[-1]
    expected = pd.read_csv(run / 'predictions/latest_forecasts_all_models.csv')
    checks = 0
    for h, name in ((3, 'gradient_boosting'), (6, 'no_change'), (12, 'gradient_boosting'), (3, 'random_forest')):
        bundle = joblib.load(run / f'models/{name}/model_c_h{h}.joblib')
        q, _ = model.calibrated_predict(bundle, origin[bundle['features']].to_numpy(float)[None, :])
        row = expected[(expected.model == name) & (expected.horizon_months == h)].iloc[0]
        for i, quantile in enumerate(model.QNAMES):
            np.testing.assert_allclose(float(origin.freight_index * np.exp(q[0, i])), row[f'{quantile}_freight_index'], rtol=1e-8)
            checks += 1
    print(f'Local training CSV: {model.DATA_PATH}')
    print(f'{len(data)} rows: {data.date.min():%Y-%m} through {data.date.max():%Y-%m}')
    print(f'{checks} Model C reference forecast values verified; no retraining.')


if __name__ == '__main__':
    main()
