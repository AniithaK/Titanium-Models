from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
MODEL_A = BASE / "training titanium model A" / "production_v6_titanium_core_corrected"
MODEL_B = BASE / "Training Titanium Model B"
RESULTS = ROOT / "results"
VIS = ROOT / "visualizations"

CLASSES = ["collapse", "fall", "stable", "rise"]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pinball(y: np.ndarray, prediction: np.ndarray, quantile: float) -> np.ndarray:
    residual = y - prediction
    return np.maximum(quantile * residual, (quantile - 1.0) * residual)


def interval_score(y: np.ndarray, low: np.ndarray, high: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    score = high - low
    score += (2.0 / alpha) * np.maximum(low - y, 0)
    score += (2.0 / alpha) * np.maximum(y - high, 0)
    return score


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    valid = denominator > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(2 * np.abs(actual[valid] - predicted[valid]) / denominator[valid]))


def reliability(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    one_hot = np.eye(len(CLASSES))[y].ravel()
    predicted = probability.ravel()
    bins = np.minimum((predicted * 10).astype(int), 9)
    gaps, weights = [], []
    for index in range(10):
        mask = bins == index
        if not mask.any():
            continue
        gaps.append(abs(float(predicted[mask].mean()) - float(one_hot[mask].mean())))
        weights.append(int(mask.sum()))
    return float(np.average(gaps, weights=weights)), float(max(gaps))


def multiclass_brier(y: np.ndarray, probability: np.ndarray) -> float:
    one_hot = np.eye(len(CLASSES))[y]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def load_model_a() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training = pd.read_csv(
        MODEL_A / "data" / "titanium_model_a_production_v6_titanium_core_corrected_training.csv",
        low_memory=False,
    )
    training["date"] = pd.to_datetime(training["date"])
    price_lookup = training.set_index(["hs_code", "date"])["import_unit_value_gbp_per_kg"]

    comparison = pd.read_csv(
        MODEL_A / "results" / "validation" / "pretest_validation_model_comparison.csv"
    )
    selected = comparison.loc[as_bool(comparison["selected_by_pretest_validation"])].copy()
    selected_map = {
        (int(row.hs_code), int(row.horizon_months)): row.model
        for row in selected.itertuples()
    }

    validation = pd.read_csv(
        MODEL_A / "results" / "validation" / "pretest_validation_predictions.csv"
    )
    validation = validation.loc[
        validation.apply(
            lambda row: row["model"]
            == selected_map[(int(row["hs_code"]), int(row["horizon_months"]))],
            axis=1,
        )
    ].copy()
    validation["forecast_origin"] = pd.to_datetime(validation["forecast_origin"])
    validation["target_month"] = validation.apply(
        lambda row: row["forecast_origin"] + pd.DateOffset(months=int(row["horizon_months"])),
        axis=1,
    )
    validation["origin_price_gbp_per_kg"] = validation.apply(
        lambda row: price_lookup.loc[(int(row["hs_code"]), row["forecast_origin"])], axis=1
    )
    for label in ["actual", "p05", "p10", "p25", "p50", "p75", "p90", "p95"]:
        source = "actual_log_change" if label == "actual" else f"{label}_log_change"
        validation[f"{label}_log_change"] = validation[source]
        validation[f"{label}_price_gbp_per_kg"] = (
            validation["origin_price_gbp_per_kg"] * np.exp(validation[source])
        )
    validation["split"] = "validation"
    validation["model_output_available"] = True
    validation["hs_product"] = validation["hs_code"].map(
        {
            810820: "Titanium unwrought and powders",
            810890: "Titanium bars, sheet, forgings and other articles",
        }
    )

    frames = [validation]
    test_specs = [
        ("disruption_test", MODEL_A / "results" / "stress_2022_2023" / "selected_and_baseline_predictions.csv"),
        ("post_disruption_test", MODEL_A / "results" / "post_2024_2025" / "selected_and_baseline_predictions.csv"),
    ]
    for split, path in test_specs:
        frame = pd.read_csv(path, low_memory=False)
        frame = frame.loc[as_bool(frame["selected_model"])].copy()
        frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
        frame["target_month"] = pd.to_datetime(frame["target_month"])
        frame["origin_price_gbp_per_kg"] = frame["actual_origin_price_gbp_per_kg"]
        frame["actual_price_gbp_per_kg"] = frame["actual_target_price_gbp_per_kg"]
        frame["actual_log_change"] = frame["actual_log_change_gbp_per_kg"]
        frame["split"] = split
        frame["model_output_available"] = as_bool(frame["model_output_available"])
        frames.append(frame)

    keep = [
        "split", "hs_code", "hs_product", "horizon_months", "forecast_origin", "target_month",
        "model", "model_output_available", "origin_price_gbp_per_kg", "actual_price_gbp_per_kg",
        "p05_price_gbp_per_kg", "p50_price_gbp_per_kg", "p95_price_gbp_per_kg",
        "actual_log_change", "p05_log_change", "p10_log_change", "p25_log_change",
        "p50_log_change", "p75_log_change", "p90_log_change", "p95_log_change",
    ]
    normalized = pd.concat([frame.reindex(columns=keep) for frame in frames], ignore_index=True)
    normalized["covered_p05_p95"] = (
        normalized["model_output_available"]
        & (normalized["actual_log_change"] >= normalized["p05_log_change"])
        & (normalized["actual_log_change"] <= normalized["p95_log_change"])
    )
    normalized["absolute_error_gbp_per_kg"] = (
        normalized["actual_price_gbp_per_kg"] - normalized["p50_price_gbp_per_kg"]
    ).abs()
    normalized["signed_error_gbp_per_kg"] = (
        normalized["p50_price_gbp_per_kg"] - normalized["actual_price_gbp_per_kg"]
    )
    normalized["forecast_origin"] = normalized["forecast_origin"].dt.strftime("%Y-%m")
    normalized["target_month"] = normalized["target_month"].dt.strftime("%Y-%m")

    summary_rows = []
    for keys, group in normalized.groupby(["split", "hs_code", "horizon_months", "model"], dropna=False):
        split, hs_code, horizon, model = keys
        available = group.loc[group["model_output_available"]].dropna(
            subset=["actual_price_gbp_per_kg", "p50_price_gbp_per_kg"]
        )
        if available.empty:
            continue
        actual = available["actual_price_gbp_per_kg"].to_numpy(float)
        predicted = available["p50_price_gbp_per_kg"].to_numpy(float)
        actual_log = available["actual_log_change"].to_numpy(float)
        pinball_values = []
        for quantile in QUANTILES:
            column = f"p{int(quantile * 100):02d}_log_change"
            pinball_values.append(pinball(actual_log, available[column].to_numpy(float), quantile))
        expected_rows = len(group)
        available_rows = len(available)
        coverage = float(group["covered_p05_p95"].sum() / expected_rows)
        mis = float(np.mean(interval_score(
            actual_log,
            available["p05_log_change"].to_numpy(float),
            available["p95_log_change"].to_numpy(float),
        ))) if available_rows == expected_rows else float("inf")
        summary_rows.append({
            "split": split,
            "hs_code": int(hs_code),
            "horizon_months": int(horizon),
            "selected_model": model,
            "expected_rows": expected_rows,
            "available_rows": available_rows,
            "mae_gbp_per_kg": float(np.mean(np.abs(actual - predicted))),
            "rmse_gbp_per_kg": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "smape": smape(actual, predicted),
            "p05_p95_coverage": coverage,
            "mean_interval_score": mis,
            "mean_pinball_loss": float(np.mean(np.column_stack(pinball_values))),
        })
    summary = pd.DataFrame(summary_rows)

    canonical = []
    for row in selected.itertuples():
        canonical.append({
            "split": "validation", "hs_code": int(row.hs_code),
            "horizon_months": int(row.horizon_months), "selected_model": row.model,
            "canonical_coverage": float(row.calibrated_p05_p95_coverage),
            "canonical_mis": float(row.calibrated_mean_interval_score),
            "baseline_mis": float(row.calibrated_mean_interval_score / row.validation_mis_ratio_to_baseline),
            "mis_ratio_to_baseline": float(row.validation_mis_ratio_to_baseline),
            "passes_acceptance": bool(row.validation_gate_eligible),
        })
    for split, path in [
        ("disruption_test", MODEL_A / "results" / "stress_2022_2023" / "evaluation_metrics.csv"),
        ("post_disruption_test", MODEL_A / "results" / "post_2024_2025" / "evaluation_metrics.csv"),
    ]:
        metrics = pd.read_csv(path)
        metrics = metrics.loc[~metrics["model"].eq("no_change_baseline")]
        for row in metrics.itertuples():
            canonical.append({
                "split": split, "hs_code": int(row.hs_code),
                "horizon_months": int(row.horizon_months), "selected_model": row.model,
                "canonical_coverage": float(row.p05_p95_coverage),
                "canonical_mis": float(row.mean_interval_score_alpha_010),
                "baseline_mis": float(row.baseline_mean_interval_score_alpha_010),
                "mis_ratio_to_baseline": float(row.mis_ratio_to_baseline) if finite(row.mis_ratio_to_baseline) else np.nan,
                "passes_acceptance": bool(row.passes_acceptance),
            })
    canonical = pd.DataFrame(canonical)
    summary = summary.merge(
        canonical,
        on=["split", "hs_code", "horizon_months", "selected_model"],
        how="left",
    )
    return normalized, summary, selected


def normalize_model_b_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for horizon in sorted(frame["horizon_months"].dropna().astype(int).unique()):
        group = frame.loc[frame["horizon_months"].astype(int).eq(horizon)].copy()
        target = f"target_class_h{horizon}"
        probabilities = (
            [f"calibrated_p_{label}" for label in CLASSES]
            if split == "validation"
            else [f"p_{label}" for label in CLASSES]
        )
        for source in group.itertuples(index=False):
            values = {column: getattr(source, column) for column in probabilities}
            actual = getattr(source, target)
            if pd.isna(actual):
                continue
            probability = np.array([values[column] for column in probabilities], dtype=float)
            predicted = CLASSES[int(probability.argmax())]
            origin = pd.Timestamp(getattr(source, "date"))
            target_month_name = f"target_month_h{horizon}"
            target_month = (
                pd.Timestamp(getattr(source, target_month_name))
                if hasattr(source, target_month_name) and pd.notna(getattr(source, target_month_name))
                else origin + pd.DateOffset(months=horizon)
            )
            rows.append({
                "split": split,
                "horizon_months": horizon,
                "forecast_origin": origin.strftime("%Y-%m"),
                "target_month": target_month.strftime("%Y-%m"),
                "country_key": getattr(source, "country_key"),
                "country": getattr(source, "country"),
                "actual_class": actual,
                "predicted_class": predicted,
                "correct": actual == predicted,
                "confidence": float(probability.max()),
                **{f"probability_{label}": float(probability[index]) for index, label in enumerate(CLASSES)},
            })
    return pd.DataFrame(rows)


def score_model_b(frame: pd.DataFrame) -> dict:
    mapping = {label: index for index, label in enumerate(CLASSES)}
    y = frame["actual_class"].map(mapping).to_numpy(int)
    probability = frame[[f"probability_{label}" for label in CLASSES]].to_numpy(float)
    predicted = probability.argmax(axis=1)
    ece, max_gap = reliability(y, probability)
    return {
        "rows": len(frame),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "brier_score": multiclass_brier(y, probability),
        "log_loss": float(log_loss(y, probability, labels=list(range(len(CLASSES))))),
        "expected_calibration_error": ece,
        "max_calibration_gap": max_gap,
    }


def model_b_training_diagnostics() -> pd.DataFrame:
    bundle = joblib.load(MODEL_B / "models" / "titanium_model_b_country_supply_shift_selected.joblib")
    data = pd.read_csv(MODEL_B / "data" / "titanium_model_b_country_month_training_2010_2026.csv", low_memory=False)
    data["date"] = pd.to_datetime(data["date"])
    rows = []
    for horizon in (3, 6, 12):
        entry = bundle["horizons"][f"h{horizon}"]
        target = f"target_class_h{horizon}"
        eligible = data.loc[
            as_bool(data[f"model_b_training_eligible_h{horizon}"])
            & data[target].notna()
            & (data["date"] < pd.Timestamp("2026-06-01"))
        ].tail(int(entry["training_rows"])).copy()
        model = entry["estimator"]
        raw = model.predict_proba(eligible[bundle["features"]])
        aligned = np.zeros((len(eligible), len(CLASSES)))
        for source_index, label in enumerate(model.named_steps["model"].classes_):
            aligned[:, CLASSES.index(label)] = raw[:, source_index]
        clipped = np.clip(aligned, 1e-9, 1)
        logits = np.log(clipped) / float(entry["temperature"])
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        temp = pd.DataFrame({
            "actual_class": eligible[target].to_numpy(),
            **{f"probability_{label}": probability[:, index] for index, label in enumerate(CLASSES)},
        })
        rows.append({"model": "Model B", "horizon_months": horizon, "diagnostic_split": "apparent_training", **score_model_b(temp)})
    return pd.DataFrame(rows)


def load_model_b() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_raw = pd.read_csv(MODEL_B / "results" / "selected_validation_predictions.csv")
    disruption_raw = pd.read_csv(MODEL_B / "results" / "disruption_2021_2023_predictions.csv")
    post_raw = pd.read_csv(MODEL_B / "results" / "post_disruption_predictions.csv")
    normalized = pd.concat(
        [
            normalize_model_b_frame(validation_raw, "validation"),
            normalize_model_b_frame(disruption_raw, "disruption_test"),
            normalize_model_b_frame(post_raw, "post_disruption_test"),
        ],
        ignore_index=True,
    )
    summary_rows = []
    for (split, horizon), group in normalized.groupby(["split", "horizon_months"]):
        summary_rows.append({"split": split, "horizon_months": int(horizon), **score_model_b(group)})
    summary = pd.DataFrame(summary_rows)

    selected = json.loads((MODEL_B / "results" / "selected_models.json").read_text(encoding="utf-8"))
    validation_baseline = []
    for horizon in (3, 6, 12):
        metrics = selected[f"h{horizon}"]["validation_metrics"]
        validation_baseline.append({
            "split": "validation", "horizon_months": horizon,
            "selected_model": metrics["candidate"],
            "historical_frequency_brier": metrics["historical_frequency_brier"],
            "persistence_brier": metrics["persistence_brier"],
            "passes_acceptance": False,
        })
    evaluation = pd.read_csv(MODEL_B / "results" / "evaluation_metrics.csv")
    evaluation["split"] = evaluation["period"].map({
        "disruption_2021_2023": "disruption_test",
        "post_disruption": "post_disruption_test",
    })
    baseline = pd.concat([
        pd.DataFrame(validation_baseline),
        evaluation[[
            "split", "horizon_months", "selected_model", "historical_frequency_brier",
            "persistence_brier", "passes_acceptance",
        ]],
    ], ignore_index=True)
    summary = summary.merge(baseline, on=["split", "horizon_months"], how="left")
    return normalized, summary, model_b_training_diagnostics()


def rounded_records(frame: pd.DataFrame, digits: int = 6) -> list[dict]:
    output = frame.copy()
    for column in output.select_dtypes(include=["float"]).columns:
        output[column] = output[column].round(digits)
    output = output.replace({np.nan: None, np.inf: None, -np.inf: None})
    return output.to_dict(orient="records")


def build_html(a_data: pd.DataFrame, a_summary: pd.DataFrame, b_data: pd.DataFrame, b_summary: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    payload = json.dumps({
        "modelA": rounded_records(a_data),
        "modelASummary": rounded_records(a_summary),
        "modelB": rounded_records(b_data),
        "modelBSummary": rounded_records(b_summary),
        "trainingDiagnostics": rounded_records(diagnostics),
    }, separators=(",", ":"))
    template = (ROOT / "code" / "dashboard_template.html").read_text(encoding="utf-8")
    return template.replace("/*__DATA__*/", f"const DATA={payload};")


def verify(a_summary: pd.DataFrame, b_data: pd.DataFrame, b_summary: pd.DataFrame) -> None:
    finite_a = a_summary.loc[a_summary["canonical_mis"].map(finite)]
    assert np.allclose(finite_a["mean_interval_score"], finite_a["canonical_mis"], atol=1e-8)
    assert np.allclose(a_summary["p05_p95_coverage"], a_summary["canonical_coverage"], atol=1e-8)
    probability = b_data[[f"probability_{label}" for label in CLASSES]].to_numpy(float)
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-8)
    canonical = pd.read_csv(MODEL_B / "results" / "evaluation_metrics.csv")
    split_map = {"disruption_2021_2023": "disruption_test", "post_disruption": "post_disruption_test"}
    for row in canonical.itertuples():
        observed = b_summary.loc[
            b_summary["split"].eq(split_map[row.period])
            & b_summary["horizon_months"].eq(row.horizon_months)
        ].iloc[0]
        assert math.isclose(observed.brier_score, row.brier_score, abs_tol=1e-8)
        assert math.isclose(observed.accuracy, row.accuracy, abs_tol=1e-8)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)
    a_data, a_summary, a_selected = load_model_a()
    b_data, b_summary, b_training = load_model_b()

    a_data.to_csv(RESULTS / "model_a_ground_truth_vs_predictions.csv", index=False)
    a_summary.to_csv(RESULTS / "model_a_performance_summary.csv", index=False)
    b_data.to_csv(RESULTS / "model_b_ground_truth_vs_predictions.csv", index=False)
    b_summary.to_csv(RESULTS / "model_b_performance_summary.csv", index=False)
    b_training.to_csv(RESULTS / "training_diagnostics.csv", index=False)
    a_selected.to_csv(RESULTS / "model_a_selected_validation_models.csv", index=False)

    verify(a_summary, b_data, b_summary)
    html = build_html(a_data, a_summary, b_data, b_summary, b_training)
    dashboard = VIS / "titanium_model_a_b_performance_dashboard.html"
    dashboard.write_text(html, encoding="utf-8")
    print(f"Model A evidence rows: {len(a_data)}")
    print(f"Model B evidence rows: {len(b_data)}")
    print("Verification: PASS")
    print(f"Dashboard: {dashboard}")


if __name__ == "__main__":
    main()
