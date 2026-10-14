"""Package-local V6 inference; preserves the original fitted prediction rules."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
QUANTILES = (.05, .10, .25, .50, .75, .90, .95)
QNAMES = ('p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95')


def enforce_non_crossing(values):
    ordered = np.maximum.accumulate([values[q] for q in QUANTILES])
    return {q: float(v) for q, v in zip(QUANTILES, ordered)}


def scale_distribution(values, fitted, row):
    current = float(row['price_volatility_6m'].iloc[0])
    reference = float(fitted.get('volatility_reference', .05))
    scale = 1.0
    if np.isfinite(current) and current > 0 and reference > 0:
        scale = float(np.clip(np.sqrt(current / reference), .80, 1.80))
    median = float(values[.50])
    return enforce_non_crossing({q: median + (float(v) - median) * scale for q, v in values.items()})


def predict_fitted(fitted, row, features):
    family = fitted['family']
    if family in ('linear', 'gradient_boosting'):
        raw = {q: float(fitted['quantile_models'][q].predict(row[features])[0]) for q in QUANTILES}
    elif family == 'random_forest':
        x = fitted['imputer'].transform(row[features])
        predictions = np.asarray([tree.predict(x)[0] for tree in fitted['model'].estimators_], dtype=float)
        raw = {q: float(np.quantile(predictions, q, method='linear')) for q in QUANTILES}
    elif family in ('elastic_net', 'huber'):
        center = float(fitted['model'].predict(row[features])[0])
        raw = {q: center + float(fitted['residual_quantiles'][q]) for q in QUANTILES}
    elif family == 'adaptive_baseline':
        raw = dict(fitted['quantiles'])
    else:
        raise ValueError(f'Unsupported fitted family: {family}')
    return scale_distribution(enforce_non_crossing(raw), fitted, row)


def apply_calibration(values, offset):
    calibrated = dict(values)
    calibrated[.05] -= float(offset)
    calibrated[.95] += float(offset)
    return enforce_non_crossing(calibrated)


def verify():
    data = pd.read_csv(RUN / 'data/titanium_model_a_production_v6_titanium_core_corrected_training.csv')
    expected = pd.read_csv(RUN / 'results/latest/latest_reference_forecasts.csv')
    checks = 0
    for hs in (810820, 810890):
        bundle = joblib.load(RUN / f'models/titanium_model_a_corrected_core_hs{hs}_selected_bundle.joblib')
        row = data[data.hs_code == hs].sort_values('date').tail(1)
        for h in (3, 6, 12):
            selection = bundle['horizons'][h]
            q = apply_calibration(predict_fitted(selection['fitted'], row, bundle['features']), selection['conformal_p05_p95_expansion'])
            reference = expected[(expected.hs_code == hs) & (expected.horizon_months == h)].iloc[0]
            for name, tau in zip(QNAMES, QUANTILES):
                value = float(row.import_unit_value_gbp_per_kg.iloc[0] * np.exp(q[tau]))
                np.testing.assert_allclose(value, reference[f'{name}_price_gbp_per_kg'], rtol=1e-8)
                checks += 1
    print(f'Model A: {checks} reference values verified using package-local data and saved bundles; no retraining.')


if __name__ == '__main__':
    verify()
