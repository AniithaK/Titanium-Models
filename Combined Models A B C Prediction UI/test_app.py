"""Check numerical consistency, selection behaviour and local API safeguards."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import app


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = app.Inference()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), lambda *args: None)
        cls.port = cls.server.server_address[1]
        cls.server.RequestHandlerClass = app.make_handler(cls.engine, cls.port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_reference_values(self):
        self.assertTrue(self.engine.verification['passed'])
        self.assertEqual(self.engine.verification['numeric_checks'], 118)

    def test_products_suppliers_horizons(self):
        for hs in (810820, 810890):
            for country in self.engine.b_data.country_key.unique():
                rows = self.engine.predict({'hs_code': hs, 'country': country})['forecasts']
                self.assertEqual(len(rows), 9)
                for row in rows:
                    if row['model'] == 'B' and row['horizon_months'] == 12:
                        self.assertFalse(row['supported'])
                        continue
                    self.assertEqual(str(app.pd.Period(row['forecast_origin']) + row['horizon_months']), row['target_month'])
                    if row['model'] == 'B':
                        self.assertAlmostEqual(sum(row[f'probability_{c}'] for c in self.engine.b_code.CLASSES), 1)
                    else:
                        values = [row[f'{q}_value'] for q in app.QNAMES]
                        self.assertEqual(values, sorted(values))

    def test_comparator_does_not_change_other_horizons(self):
        selected = self.engine.predict({})['forecasts']
        comparator = self.engine.predict({'c_policy': 'comparator'})['forecasts']
        for original, changed in zip(selected, comparator):
            if original['model'] == 'C' and original['horizon_months'] == 3:
                self.assertEqual(changed['selected_model'], 'random_forest')
                self.assertTrue(changed['retrospective_comparator'])
            else:
                self.assertEqual(original.keys(), changed.keys())
                for key, value in original.items():
                    # Parallel tree aggregation can vary at floating-point precision.
                    if isinstance(value, float):
                        self.assertAlmostEqual(value, changed[key], places=12)
                    else:
                        self.assertEqual(value, changed[key])

    def test_api_and_static_page(self):
        with urlopen(self.url) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b'Prediction Workbench', response.read())
        request = Request(self.url + '/api/predict', data=json.dumps({'hs_code': 810890, 'country': 'china'}).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            result = json.load(response)
            self.assertEqual(result['selection']['country'], 'china')
            self.assertEqual(len(result['forecasts']), 9)

    def test_bad_input_and_cross_origin_rejected(self):
        for headers, data, code in [({}, b'{"hs_code":999}', 400), ({'Origin': 'https://example.com'}, b'{}', 403), ({}, b'[]', 400)]:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.url + '/api/predict', data=data, headers=headers))
            self.assertEqual(error.exception.code, code)


if __name__ == '__main__':
    unittest.main()
