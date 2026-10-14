# Combined Titanium A / B / C Live Prediction UI

Run `START_UI.cmd`, then open http://127.0.0.1:8765 in your browser. Stop with Ctrl+C in the terminal. The app binds to loopback only, does not publish a website and requires no external services. Python must use the existing model-training environment.

## What Runs

- A: selected V6 bundles for HS 810820 and 810890; import unit value in GBP/kg; direct H3/H6/H12 quantiles.
- B: probability_calibration_v2 selected H3/H6 classifiers; country supply-share collapse/fall/stable/rise probabilities. H12 is unsupported.
- C: training_v1 selected H3/H12 Gradient Boosting and H6 no-change baseline; freight log changes converted to index levels. Brent is an input, not a predicted output. H3 Random Forest is an optional retrospective comparator, not a promoted validation winner.

Every prediction request executes the saved models against the latest prepared rows. No models are trained or modified. Horizon selection changes the view of the outputs; product, supplier, policy changes and Run execute inference again. JSON/CSV downloads contain all horizons and artifact paths, not fabricated future actuals.

## Data And Limitations

A/B reference June 2026; C references August 2026. Horizons count from those observations, not the computer date. Publication dates remain assumed. Data/model cutoff mismatches are rejected rather than silently extrapolated. Updated data require a separately validated refresh/refit process. Latest inference is not an untouched historical backtest.

All packages remain research/demo. B calibration failures remain visible. C's H6 fallback and outstanding acceptance gaps remain visible. A unit values can reflect product mix rather than a benchmark market price. No landed-cost total is formed because the models use different units.

## Verification And Dependencies

`python app.py --verify-only` runs inference and verifies it against existing saved forecasts. Results and SHA-256 artifact fingerprints are written to `results/verification.json`. Reference reproduction confirms implementation consistency, not future accuracy. Do not replace trusted joblib files with untrusted downloads: pickle-based files execute code when loaded.

Install the pinned `requirements.txt` here, or the combined root requirements. No new web framework is required. The app imports package-local inference/feature code inside the three selected model folders; A no longer requires the `TRAIL CODES` adapter. Keep the model-folder hierarchy beneath the combined root. This UI folder alone is not a copy of all models.

Files: `app.py` inference/API server; `ui/` browser interface; `START_UI.cmd` launcher; `test_app.py` integration/numerical tests; `results/` verification and QA records.
