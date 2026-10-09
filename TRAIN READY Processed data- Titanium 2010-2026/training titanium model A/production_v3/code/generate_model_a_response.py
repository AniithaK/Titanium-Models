from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
RUN_DIR = HERE.parents[1]
DATA_PATH = RUN_DIR / "data" / "titanium_model_a_production_v3_training.csv"
MODELS_DIR = RUN_DIR / "models"
POST_METRICS_PATH = RUN_DIR / "results" / "post_2024_2025" / "evaluation_metrics.csv"
OUTPUT_JSON = RUN_DIR / "results" / "latest" / "model_a_production_response.json"
OUTPUT_CSV = RUN_DIR / "results" / "latest" / "model_a_latest_forecasts.csv"
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def enforce_non_crossing(values: dict[float, float]) -> dict[float, float]:
    ordered = np.maximum.accumulate([values[quantile] for quantile in QUANTILES])
    return {quantile: float(value) for quantile, value in zip(QUANTILES, ordered)}


def scale_distribution(
    values: dict[float, float], fitted: dict[str, object], row: pd.DataFrame
) -> dict[float, float]:
    current = float(row["price_volatility_6m"].iloc[0])
    reference = float(fitted.get("volatility_reference", 0.05))
    scale = 1.0
    if np.isfinite(current) and current > 0 and reference > 0:
        scale = float(np.clip(np.sqrt(current / reference), 0.80, 1.80))
    median = float(values[0.50])
    return enforce_non_crossing(
        {
            quantile: median + (float(value) - median) * scale
            for quantile, value in values.items()
        }
    )


def predict_fitted(
    fitted: dict[str, object], row: pd.DataFrame, features: list[str]
) -> dict[float, float]:
    family = str(fitted["family"])
    if family in {"linear", "gradient_boosting"}:
        raw = {
            quantile: float(fitted["quantile_models"][quantile].predict(row[features])[0])
            for quantile in QUANTILES
        }
    elif family == "random_forest":
        x = fitted["imputer"].transform(row[features])
        tree_predictions = np.asarray(
            [tree.predict(x)[0] for tree in fitted["model"].estimators_], dtype=float
        )
        raw = {
            quantile: float(np.quantile(tree_predictions, quantile, method="linear"))
            for quantile in QUANTILES
        }
    elif family in {"elastic_net", "huber"}:
        center = float(fitted["model"].predict(row[features])[0])
        raw = {
            quantile: center + float(fitted["residual_quantiles"][quantile])
            for quantile in QUANTILES
        }
    elif family == "adaptive_baseline":
        raw = dict(fitted["quantiles"])
    else:
        raise ValueError(f"Unsupported fitted family: {family}")
    return scale_distribution(enforce_non_crossing(raw), fitted, row)


def apply_calibration(values: dict[float, float], offset: float) -> dict[float, float]:
    calibrated = dict(values)
    calibrated[0.05] -= float(offset)
    calibrated[0.95] += float(offset)
    return enforce_non_crossing(calibrated)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["period_m"] = data["date"].dt.to_period("M")
    metrics = pd.read_csv(POST_METRICS_PATH)
    selected_metrics = metrics[metrics["model"] != "no_change_baseline"].set_index(
        ["hs_code", "horizon_months"]
    )

    response = {
        "model": "SCI Model A titanium import price",
        "model_version": "production_v3_research",
        "status": "not_production_approved",
        "currency": "GBP",
        "unit": "kg",
        "target": "UK monthly titanium import unit value",
        "generated_from_data_through": str(data["period_m"].max()),
        "forecasts_by_hs_code": [],
    }
    flat_records = []
    for hs_code in (810820, 810890):
        row = data[data["hs_code"] == hs_code].sort_values("date").tail(1)
        origin = row.iloc[0]
        model_files = list(MODELS_DIR.glob(f"*hs{hs_code}*selected_bundle.joblib"))
        if len(model_files) != 1:
            raise RuntimeError(f"Expected one saved bundle for HS {hs_code}, found {model_files}")
        bundle = joblib.load(model_files[0])
        features = list(bundle["features"])
        forecasts = []
        for horizon in (3, 6, 12):
            horizon_bundle = bundle["horizons"][horizon]
            fitted = horizon_bundle["fitted"]
            raw = predict_fitted(fitted, row, features)
            calibrated = apply_calibration(
                raw, float(horizon_bundle["conformal_p05_p95_expansion"])
            )
            prices = {
                quantile: float(origin["import_unit_value_gbp_per_kg"] * np.exp(value))
                for quantile, value in calibrated.items()
            }
            candidate = str(horizon_bundle["candidate"]["name"])
            post_pass = bool(selected_metrics.loc[(hs_code, horizon), "passes_acceptance"])
            forecast = {
                "horizon_months": horizon,
                "target_month": str(origin["period_m"] + horizon),
                "p05_gbp_per_kg": round(prices[0.05], 4),
                "p10_gbp_per_kg": round(prices[0.10], 4),
                "p25_gbp_per_kg": round(prices[0.25], 4),
                "p50_gbp_per_kg": round(prices[0.50], 4),
                "p75_gbp_per_kg": round(prices[0.75], 4),
                "p90_gbp_per_kg": round(prices[0.90], 4),
                "p95_gbp_per_kg": round(prices[0.95], 4),
                "selected_model": candidate,
                "fallback_used": candidate == "adaptive_no_change",
                "post_evaluation_pass": post_pass,
            }
            forecasts.append(forecast)
            flat_records.append(
                {
                    "hs_code": hs_code,
                    "forecast_origin": str(origin["period_m"]),
                    **forecast,
                }
            )
        response["forecasts_by_hs_code"].append(
            {
                "hs_code": str(hs_code),
                "product": origin["hs_product"],
                "forecast_origin": str(origin["period_m"]),
                "origin_price_gbp_per_kg": round(float(origin["import_unit_value_gbp_per_kg"]), 4),
                "data_quality": {
                    "target_quality_status": origin["target_quality_status"],
                    "world_bank_stale": bool(origin["world_bank_stale_flag"]),
                    "world_bank_source_age_months": int(origin["world_bank_source_age_months"]),
                    "usgs_stale": bool(origin["usgs_stale_flag"]),
                    "usgs_source_age_years": float(origin["usgs_source_age_years"]),
                },
                "forecasts": forecasts,
            }
        )

    OUTPUT_JSON.write_text(json.dumps(response, indent=2), encoding="utf-8")
    pd.DataFrame(flat_records).to_csv(OUTPUT_CSV, index=False)
    print(f"JSON response: {OUTPUT_JSON}")
    print(f"Forecast CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
