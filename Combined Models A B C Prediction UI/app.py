"""Local-only inference UI for trusted, existing Titanium model artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A = ROOT / 'training titanium model A/production_v6_titanium_core_corrected'
B = ROOT / 'Training Titanium Model B/improve accuracy/probability_calibration_v2'
C = ROOT / 'Training Titanium Model C/training_v1'
QUANTILES = (.05, .10, .25, .50, .75, .90, .95)
QNAMES = ('p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class Inference:
    def __init__(self):
        # Imports execute definitions only; none of the training entry points run.
        self.a_code = module('combined_a_inference', A / 'code/saved_model_inference.py')
        self.b_code = module('combined_b_inference', B / 'code/train_probability_model_b.py')
        self.c_code = module('combined_c_inference', C / 'code/train_model_c.py')
        self.a_data = pd.read_csv(A / 'data/titanium_model_a_production_v6_titanium_core_corrected_training.csv')
        self.a_data['date'] = pd.to_datetime(self.a_data.date)
        self.b_data, _ = self.b_code.load_data()
        self.c_data = self.c_code.load_data()
        self.paths = {}
        self.a_models = {}
        self.b_models = {}
        self.c_models = {}
        for hs in (810820, 810890):
            path = A / f'models/titanium_model_a_corrected_core_hs{hs}_selected_bundle.joblib'
            self.a_models[hs] = joblib.load(path)
            self.paths[f'A {hs}'] = path
        for h in (3, 6):
            path = B / f'models/model_b_h{h}_calibrated.joblib'
            self.b_models[h] = joblib.load(path)
            self.paths[f'B H{h}'] = path
        for h, name in ((3, 'gradient_boosting'), (6, 'no_change'), (12, 'gradient_boosting')):
            path = C / f'models/{name}/model_c_h{h}.joblib'
            self.c_models[h] = joblib.load(path)
            self.paths[f'C H{h}'] = path
        self.c_comparator = joblib.load(C / 'models/random_forest/model_c_h3.joblib')
        self.paths['C H3 retrospective comparator'] = C / 'models/random_forest/model_c_h3.joblib'
        self.lock = threading.Lock()
        self.verify()

    def predict_a(self, hs, h):
        row = self.a_data[self.a_data.hs_code == hs].sort_values('date').tail(1)
        origin = row.iloc[0]
        bundle = self.a_models[hs]
        if str(origin.date.to_period('M')) != str(bundle['forecast_origin'])[:7]:
            raise ValueError('Model A data changed: refit/approve updated artifacts before inference.')
        selected = bundle['horizons'][h]
        raw = self.a_code.predict_fitted(selected['fitted'], row, bundle['features'])
        q = self.a_code.apply_calibration(raw, selected['conformal_p05_p95_expansion'])
        base = float(origin.import_unit_value_gbp_per_kg)
        result = dict(model='A', hs_code=hs, product=bundle['hs_product'], horizon_months=h,
                      forecast_origin=str(origin.date.to_period('M')),
                      target_month=str(origin.date.to_period('M') + h),
                      selected_model=selected['candidate']['name'], origin_value=base,
                      unit='GBP/kg', reporting_only=h == 12,
                      data_available_date=str(bundle['forecast_data_available_date']),
                      artifact=str(self.paths[f'A {hs}']), status='Research/demo; not production approved')
        result.update({f'{name}_value': float(base * np.exp(q[tau])) for name, tau in zip(QNAMES, QUANTILES)})
        return result

    def predict_b(self, country, h):
        if h == 12:
            return dict(model='B', horizon_months=h, supported=False,
                        reason='This calibrated Model B package supports H3 and H6 only.')
        row = self.b_data[(self.b_data.date == self.b_data.date.max()) & (self.b_data.country_key == country)]
        bundle = self.b_models[h]
        if str(row.date.iloc[0].date()) != bundle['source_cutoff']:
            raise ValueError('Model B source cutoff changed; update model artifacts first.')
        result = self.b_code.predict_bundle(bundle, row).iloc[0].to_dict()
        result.update(model='B', supported=True, artifact=str(self.paths[f'B H{h}']),
                      status='Research/demo; disruption/post-disruption calibration limitations remain',
                      unit='Probability of supplier-share movement')
        return result

    def predict_c(self, h, comparator=False):
        bundle = self.c_comparator if comparator and h == 3 else self.c_models[h]
        origin = self.c_data.iloc[-1]
        if str(origin.date.to_period('M')) != str(bundle['fit_origin_reference_month'])[:7]:
            raise ValueError('Model C source cutoff changed; update model artifacts first.')
        q, _ = self.c_code.calibrated_predict(bundle, origin[bundle['features']].to_numpy(float)[None, :])
        base = float(origin.freight_index)
        result = dict(model='C', horizon_months=h, forecast_origin=str(origin.date.to_period('M')),
                      target_month=str(origin.date.to_period('M') + h), selected_model=bundle['model'],
                      origin_value=base, unit='Freight producer-price index', reporting_only=h == 12,
                      data_available_date=str(origin.forecast_date_assumed.date()),
                      retrospective_comparator=bool(comparator and h == 3),
                      artifact=str(self.paths['C H3 retrospective comparator' if comparator and h == 3 else f'C H{h}']),
                      status='Research/demo; acceptance gaps and assumed publication dates remain')
        for i, name in enumerate(QNAMES):
            result[f'{name}_value'] = float(base * np.exp(q[0, i]))
            result[f'{name}_log_change'] = float(q[0, i])
            result[f'{name}_pct_change'] = float(np.expm1(q[0, i]) * 100)
        return result

    def verify(self):
        expected_a = pd.read_csv(A / 'results/latest/latest_reference_forecasts.csv')
        expected_b = pd.read_csv(B / 'results/latest_forecasts.csv')
        expected_c = pd.read_csv(C / 'predictions/latest_forecasts_all_models.csv')
        checks = 0
        for _, row in expected_a.iterrows():
            actual = self.predict_a(int(row.hs_code), int(row.horizon_months))
            for q in QNAMES:
                np.testing.assert_allclose(actual[f'{q}_value'], row[f'{q}_price_gbp_per_kg'], rtol=1e-8)
                checks += 1
        for _, row in expected_b.iterrows():
            actual = self.predict_b(row.country_key, int(row.horizon_months))
            probabilities = [actual[f'probability_{c}'] for c in self.b_code.CLASSES]
            np.testing.assert_allclose(sum(probabilities), 1, atol=1e-10)
            for c in self.b_code.CLASSES:
                np.testing.assert_allclose(actual[f'probability_{c}'], row[f'probability_{c}'], atol=1e-10)
                checks += 1
        for h in (3, 6, 12):
            for comparator in ([False, True] if h == 3 else [False]):
                actual = self.predict_c(h, comparator)
                row = expected_c[(expected_c.horizon_months == h) & (expected_c.model == actual['selected_model'])].iloc[0]
                for q in QNAMES:
                    np.testing.assert_allclose(actual[f'{q}_value'], row[f'{q}_freight_index'], rtol=1e-8)
                    checks += 1
        self.verification = dict(passed=True, numeric_checks=checks,
                                 method='Live artifact inference matches previously saved reference forecasts',
                                 note='Numerical reproduction does not establish future forecast accuracy.')

    def meta(self):
        return dict(verification=self.verification,
                    suppliers=self.b_data[['country_key', 'country']].drop_duplicates().to_dict('records'),
                    artifacts=[dict(name=name, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in self.paths.items()],
                    source_dates=dict(A=str(self.a_data.date.max().date()), B=str(self.b_data.date.max().date()), C=str(self.c_data.date.max().date())))

    def predict(self, request):
        hs = int(request.get('hs_code', 810820))
        country = request.get('country', 'russia')
        comparator = request.get('c_policy', 'selected') == 'comparator'
        if hs not in self.a_models or country not in set(self.b_data.country_key):
            raise ValueError('Select a supported product and supplier.')
        if request.get('c_policy', 'selected') not in ('selected', 'comparator'):
            raise ValueError('Invalid Model C policy.')
        with self.lock:
            rows = []
            for h in (3, 6, 12):
                rows.extend([self.predict_a(hs, h), self.predict_b(country, h), self.predict_c(h, comparator)])
            for row in rows:
                if row['model'] in ('A', 'C'):
                    values = [row[f'{q}_value'] for q in QNAMES]
                    if not np.isfinite(values).all() or np.any(np.diff(values) < -1e-9):
                        raise ValueError('Invalid or crossing forecast quantiles.')
        return dict(generated_at=datetime.now(timezone.utc).isoformat(), inference='Saved Python models executed for this request; no retraining',
                    selection=dict(hs_code=hs, country=country, c_policy='comparator' if comparator else 'selected'), forecasts=rows)


def make_handler(engine, port):
    class Handler(BaseHTTPRequestHandler):
        def allowed(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
                return False
            origin = self.headers.get('Origin')
            return origin is None or origin in (f'http://127.0.0.1:{port}', f'http://localhost:{port}')

        def send(self, status, body, content_type='application/json'):
            if content_type == 'application/json':
                body = json.dumps(body, allow_nan=False, default=str).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {'error': 'Loopback access only.'})
            if self.path == '/api/meta':
                return self.send(200, engine.meta())
            files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if self.path not in files:
                return self.send(404, {'error': 'Not found.'})
            file, mime = files[self.path]
            self.send(200, (HERE / 'ui' / file).read_bytes(), mime)

        def do_POST(self):
            if not self.allowed():
                return self.send(403, {'error': 'Loopback access only.'})
            if self.path != '/api/predict':
                return self.send(404, {'error': 'Not found.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Request must contain at most 4096 bytes.')
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('Request must be a JSON object.')
                self.send(200, engine.predict(request))
            except (ValueError, TypeError) as error:
                self.send(400, {'error': str(error)})
            except Exception:
                logging.exception('Inference failed')
                self.send(500, {'error': 'Inference failed. See the local terminal; no fallback forecast was substituted.'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    engine = Inference()
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results/verification.json').write_text(json.dumps(engine.meta(), indent=2), encoding='utf-8')
    if args.verify_only:
        print(json.dumps(engine.verification, indent=2))
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine, args.port))
    print(f'Titanium live inference UI: http://127.0.0.1:{args.port}', flush=True)
    print('Local use only. Ctrl+C stops the app. Models are not retrained.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
