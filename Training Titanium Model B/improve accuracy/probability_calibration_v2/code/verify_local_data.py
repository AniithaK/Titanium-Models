"""Verify local Model B inputs and reproduce saved supplier probabilities."""
from pathlib import Path
import hashlib

import joblib
import numpy as np
import pandas as pd
import train_probability_model_b as model


def main():
    run = Path(__file__).resolve().parents[1]
    data, thresholds = model.load_data()
    assert model.DATA_PATH.parent == run / 'data'
    expected = pd.read_csv(run / 'results/latest_forecasts.csv')
    latest = data[data.date == data.date.max()]
    checks = 0
    for h in (3, 6):
        bundle = joblib.load(run / f'models/model_b_h{h}_calibrated.joblib')
        assert hashlib.sha256(model.DATA_PATH.read_bytes()).hexdigest() == bundle['source_sha256']
        assert thresholds == bundle['thresholds']
        predicted = model.predict_bundle(bundle, latest)
        for _, row in predicted.iterrows():
            reference = expected[(expected.country_key == row.country_key) & (expected.horizon_months == h)].iloc[0]
            probabilities = [row[f'probability_{c}'] for c in model.CLASSES]
            np.testing.assert_allclose(sum(probabilities), 1, atol=1e-10)
            assert row.target_month == reference.target_month
            assert row.predicted_class == reference.predicted_class
            for c in model.CLASSES:
                np.testing.assert_allclose(row[f'probability_{c}'], reference[f'probability_{c}'], atol=1e-10)
                checks += 1
    print(f'Local training CSV: {model.DATA_PATH}')
    print(f'{len(data)} country-month rows; {data.country_key.nunique()} supplier groups; {data.date.min():%Y-%m} through {data.date.max():%Y-%m}')
    print(f'{checks} saved probability values reproduced; source hash and fixed thresholds verified; no retraining.')


if __name__ == '__main__':
    main()
