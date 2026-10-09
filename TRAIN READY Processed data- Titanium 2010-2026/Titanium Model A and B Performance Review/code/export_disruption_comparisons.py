from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS = Path(
    r"C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM"
    r"\TRAIN READY Processed data- Titanium 2010-2026"
    r"\Titanium Model A and B Performance Review\RESULT ANALYSIS"
)
OUTPUT = RESULTS / "ground_truth_comparisons"

MODEL_A_COLUMNS = [
    "split_label",
    "hs_code",
    "hs_product",
    "horizon_months",
    "forecast_origin",
    "target_month",
    "model",
    "origin_price_gbp_per_kg",
    "actual_price_gbp_per_kg",
    "p50_price_gbp_per_kg",
    "p05_price_gbp_per_kg",
    "p95_price_gbp_per_kg",
    "signed_error_gbp_per_kg",
    "absolute_error_gbp_per_kg",
    "signed_deviation_pct",
    "absolute_deviation_pct",
    "covered_p05_p95",
    "actual_direction",
    "predicted_direction",
    "direction_correct",
    "within_5_pct",
    "within_10_pct",
    "within_20_pct",
]

MODEL_B_COLUMNS = [
    "split_label",
    "country_key",
    "country",
    "horizon_months",
    "forecast_origin",
    "target_month",
    "actual_class",
    "predicted_class",
    "correct",
    "confidence",
    "actual_class_probability",
    "probability_collapse",
    "probability_fall",
    "probability_stable",
    "probability_rise",
    "second_highest_probability",
    "prediction_margin",
    "probability_sum",
]

EXPORTS = [
    (
        "model_a_disruption_ground_truth_vs_predicted.csv",
        "model_a",
        "disruption_test",
    ),
    (
        "model_a_post_disruption_ground_truth_vs_predicted.csv",
        "model_a",
        "post_disruption_test",
    ),
    (
        "model_b_disruption_ground_truth_vs_predicted.csv",
        "model_b",
        "disruption_test",
    ),
    (
        "model_b_post_disruption_ground_truth_vs_predicted.csv",
        "model_b",
        "post_disruption_test",
    ),
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_a = pd.read_csv(RESULTS / "model_a_ground_truth_vs_predictions_analysis.csv")
    model_b = pd.read_csv(RESULTS / "model_b_ground_truth_vs_predictions_analysis.csv")

    created: list[tuple[str, int]] = []
    for filename, model_name, split in EXPORTS:
        if model_name == "model_a":
            frame = model_a.loc[model_a["split"] == split, MODEL_A_COLUMNS].copy()
            frame = frame.sort_values(
                ["target_month", "hs_code", "horizon_months", "forecast_origin"]
            )
        else:
            frame = model_b.loc[model_b["split"] == split, MODEL_B_COLUMNS].copy()
            frame = frame.sort_values(
                ["target_month", "country", "horizon_months", "forecast_origin"]
            )

        if frame.empty:
            raise RuntimeError(f"No rows found for {model_name} split {split}")

        output_path = OUTPUT / filename
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        created.append((filename, len(frame)))

    print(f"Comparison files saved in: {OUTPUT}")
    for filename, rows in created:
        print(f"{filename}: {rows} rows")


if __name__ == "__main__":
    main()
