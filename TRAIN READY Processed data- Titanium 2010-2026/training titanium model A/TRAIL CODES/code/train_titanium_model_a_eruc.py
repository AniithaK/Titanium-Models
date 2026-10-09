from __future__ import annotations

import base64
import json
import math
import shutil
import warnings
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
HORIZONS = (3, 6, 12)
HS_CODES = (810820, 810890)
TEST_START = pd.Period("2022-02", freq="M")
TEST_END = pd.Period("2023-12", freq="M")
MIN_TRAINING_ROWS = 48
ALPHA = 0.10

HERE = Path(__file__).resolve()
RUN_DIR = HERE.parents[1]
SOURCE_DIR = RUN_DIR.parent
SOURCE_DATA_PATH = SOURCE_DIR / "titanium_model_a_training_with_fx_eruc_2010_2026.csv"

DATA_DIR = RUN_DIR / "data"
MODELS_DIR = RUN_DIR / "models"
RESULTS_DIR = RUN_DIR / "results"
VIS_DIR = RUN_DIR / "visualizations"
DOCS_DIR = RUN_DIR / "documentation"
DATA_COPY_PATH = DATA_DIR / SOURCE_DATA_PATH.name

FEATURES = [
    "price_log_change_gbp_1m",
    "price_log_change_gbp_2m",
    "price_log_change_gbp_3m",
    "price_log_change_gbp_6m",
    "price_log_change_gbp_12m",
    "quantity_log_change_1m",
    "quantity_log_change_3m",
    "quantity_log_change_12m",
    "supplier_share_usa",
    "supplier_share_japan",
    "supplier_share_france",
    "supplier_share_russia",
    "supplier_share_china",
    "supplier_share_other",
    "eruc_log_change_1m",
    "eruc_log_change_3m",
    "eruc_log_change_6m",
    "eruc_log_change_12m",
    "eu_collection_change_flag",
]

VALIDATION_FOLDS = (
    (pd.Period("2017-01", "M"), pd.Period("2017-12", "M")),
    (pd.Period("2018-01", "M"), pd.Period("2018-12", "M")),
    (pd.Period("2019-01", "M"), pd.Period("2019-12", "M")),
    (pd.Period("2020-02", "M"), pd.Period("2021-01", "M")),
    (pd.Period("2021-02", "M"), pd.Period("2022-01", "M")),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    description: str


CANDIDATES = (
    Candidate(
        name="linear_quantile",
        family="linear",
        description="Regularised linear quantile regression",
    ),
    Candidate(
        name="gradient_boosting_quantile",
        family="gradient_boosting",
        description="Non-linear gradient-boosted quantile regression",
    ),
    Candidate(
        name="random_forest_distribution",
        family="random_forest",
        description="Random-forest tree prediction distribution",
    ),
)
BASELINE_NAME = "no_change_baseline"


def ensure_directories() -> None:
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR, VIS_DIR, DOCS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def target_column(horizon: int) -> str:
    return f"target_log_change_gbp_per_kg_h{horizon}"


def target_price_column(horizon: int) -> str:
    return f"target_unit_value_gbp_per_kg_h{horizon}"


def target_available_column(horizon: int) -> str:
    return f"target_data_available_date_assumed_h{horizon}"


def eligibility_column(horizon: int) -> str:
    return f"model_a_training_eligible_h{horizon}"


def quantile_suffix(quantile: float) -> str:
    return f"p{int(quantile * 100):02d}"


def load_data() -> pd.DataFrame:
    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(SOURCE_DATA_PATH)
    data = pd.read_csv(SOURCE_DATA_PATH)
    required = [
        "date",
        "hs_code",
        "hs_product",
        "trade_data_available_date_assumed",
        "import_unit_value_gbp_per_kg",
        "model_a_selected_feature_complete",
        *FEATURES,
    ]
    for horizon in HORIZONS:
        required.extend(
            [
                target_column(horizon),
                target_price_column(horizon),
                target_available_column(horizon),
                eligibility_column(horizon),
            ]
        )
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")

    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["period_m"] = data["date"].dt.to_period("M")
    data["trade_data_available_date_assumed"] = pd.to_datetime(
        data["trade_data_available_date_assumed"], errors="raise"
    )
    for horizon in HORIZONS:
        data[target_available_column(horizon)] = pd.to_datetime(
            data[target_available_column(horizon)], errors="coerce"
        )
    if data.duplicated(["hs_code", "period_m"]).any():
        raise ValueError("Duplicate HS-code/month rows found in training data.")
    if set(data["hs_code"].astype(int)) != set(HS_CODES):
        raise ValueError(f"Expected HS codes {HS_CODES}.")
    if data[FEATURES].replace([np.inf, -np.inf], np.nan).isna().all(axis=1).any():
        raise ValueError("At least one row has no usable model features.")
    return data.sort_values(["hs_code", "date"]).reset_index(drop=True)


def eligible_training_rows(
    hs_data: pd.DataFrame, horizon: int, cutoff: pd.Timestamp
) -> pd.DataFrame:
    mask = (
        hs_data[eligibility_column(horizon)].astype(bool)
        & hs_data[FEATURES].notna().all(axis=1)
        & hs_data[target_column(horizon)].notna()
        & hs_data[target_available_column(horizon)].notna()
        & (hs_data[target_available_column(horizon)] <= cutoff)
    )
    return hs_data.loc[mask].copy()


def make_quantile_pipeline(candidate: Candidate, quantile: float) -> Pipeline:
    if candidate.family == "linear":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    QuantileRegressor(
                        quantile=quantile,
                        alpha=0.01,
                        solver="highs",
                    ),
                ),
            ]
        )
    if candidate.family == "gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=quantile,
                        random_state=SEED,
                        n_estimators=180,
                        learning_rate=0.035,
                        max_depth=2,
                        min_samples_leaf=10,
                        subsample=0.85,
                    ),
                ),
            ]
        )
    raise ValueError(f"Quantile pipeline is not defined for {candidate.family}.")


def fit_candidate(
    candidate: Candidate, train: pd.DataFrame, horizon: int
) -> dict[str, object]:
    x = train[FEATURES]
    y = train[target_column(horizon)].to_numpy(dtype=float)
    if candidate.family in {"linear", "gradient_boosting"}:
        models: dict[float, Pipeline] = {}
        for quantile in QUANTILES:
            model = make_quantile_pipeline(candidate, quantile)
            model.fit(x, y)
            models[quantile] = model
        return {"family": candidate.family, "quantile_models": models}

    if candidate.family == "random_forest":
        imputer = SimpleImputer(strategy="median")
        x_imputed = imputer.fit_transform(x)
        model = RandomForestRegressor(
            n_estimators=320,
            max_depth=7,
            min_samples_leaf=7,
            max_features=0.80,
            bootstrap=True,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(x_imputed, y)
        return {"family": candidate.family, "imputer": imputer, "model": model}
    raise ValueError(candidate.family)


def enforce_non_crossing(values: dict[float, float]) -> dict[float, float]:
    ordered = np.maximum.accumulate([values[q] for q in QUANTILES])
    return {q: float(value) for q, value in zip(QUANTILES, ordered)}


def predict_candidate(
    fitted: dict[str, object], row: pd.DataFrame
) -> dict[float, float]:
    family = str(fitted["family"])
    if family in {"linear", "gradient_boosting"}:
        models = fitted["quantile_models"]
        raw = {q: float(models[q].predict(row[FEATURES])[0]) for q in QUANTILES}
        return enforce_non_crossing(raw)

    if family == "random_forest":
        imputer = fitted["imputer"]
        model = fitted["model"]
        x_imputed = imputer.transform(row[FEATURES])
        tree_predictions = np.array(
            [tree.predict(x_imputed)[0] for tree in model.estimators_], dtype=float
        )
        raw = {
            q: float(np.quantile(tree_predictions, q, method="linear"))
            for q in QUANTILES
        }
        return enforce_non_crossing(raw)
    raise ValueError(family)


def baseline_quantiles(target: pd.Series) -> dict[float, float]:
    values = target.dropna().to_numpy(dtype=float)
    empirical = {q: float(np.quantile(values, q)) for q in QUANTILES}
    result = {
        0.05: min(empirical[0.05], 0.0),
        0.10: min(empirical[0.10], 0.0),
        0.25: min(empirical[0.25], 0.0),
        0.50: 0.0,
        0.75: max(empirical[0.75], 0.0),
        0.90: max(empirical[0.90], 0.0),
        0.95: max(empirical[0.95], 0.0),
    }
    return enforce_non_crossing(result)


def interval_score(actual: float, lower: float, upper: float) -> float:
    score = upper - lower
    if actual < lower:
        score += (2.0 / ALPHA) * (lower - actual)
    elif actual > upper:
        score += (2.0 / ALPHA) * (actual - upper)
    return float(score)


def conformal_quantile(scores: np.ndarray) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("No validation scores are available for conformal calibration.")
    probability = min(1.0, math.ceil((clean.size + 1) * (1.0 - ALPHA)) / clean.size)
    return float(np.quantile(clean, probability, method="higher"))


def apply_calibration(
    prediction: dict[float, float], offset: float
) -> dict[float, float]:
    calibrated = dict(prediction)
    calibrated[0.05] -= offset
    calibrated[0.95] += offset
    return enforce_non_crossing(calibrated)


def prediction_record_values(
    prediction: dict[float, float], prefix: str = ""
) -> dict[str, float]:
    return {
        f"{prefix}{quantile_suffix(q)}_log_change": float(prediction[q])
        for q in QUANTILES
    }


def run_pretest_validation(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[int, int], Candidate]]:
    records: list[dict[str, object]] = []
    candidate_by_name = {candidate.name: candidate for candidate in CANDIDATES}

    for hs_code in HS_CODES:
        hs_data = data[data["hs_code"] == hs_code].sort_values("date")
        for horizon in HORIZONS:
            for fold_number, (fold_start, fold_end) in enumerate(VALIDATION_FOLDS, 1):
                validation = hs_data[
                    (hs_data["period_m"] >= fold_start)
                    & (hs_data["period_m"] <= fold_end)
                    & hs_data[FEATURES].notna().all(axis=1)
                    & hs_data[target_column(horizon)].notna()
                ]
                if validation.empty:
                    continue
                cutoff = validation["trade_data_available_date_assumed"].min()
                train = eligible_training_rows(hs_data, horizon, cutoff)
                if len(train) < MIN_TRAINING_ROWS:
                    continue

                fitted_candidates = {
                    candidate.name: fit_candidate(candidate, train, horizon)
                    for candidate in CANDIDATES
                }
                for _, row in validation.iterrows():
                    actual = float(row[target_column(horizon)])
                    common = {
                        "hs_code": hs_code,
                        "horizon_months": horizon,
                        "fold": fold_number,
                        "forecast_origin": str(row["period_m"]),
                        "actual_log_change": actual,
                        "training_rows": len(train),
                    }
                    for candidate in CANDIDATES:
                        prediction = predict_candidate(
                            fitted_candidates[candidate.name], row.to_frame().T
                        )
                        records.append(
                            {
                                **common,
                                "model": candidate.name,
                                **prediction_record_values(prediction, "raw_"),
                            }
                        )
                    baseline = baseline_quantiles(train[target_column(horizon)])
                    records.append(
                        {
                            **common,
                            "model": BASELINE_NAME,
                            **prediction_record_values(baseline, "raw_"),
                        }
                    )

    predictions = pd.DataFrame(records)
    if predictions.empty:
        raise RuntimeError("Pre-test validation produced no predictions.")

    calibration_records: list[dict[str, object]] = []
    for keys, group in predictions.groupby(["hs_code", "horizon_months", "model"]):
        lower = group["raw_p05_log_change"].to_numpy(dtype=float)
        upper = group["raw_p95_log_change"].to_numpy(dtype=float)
        actual = group["actual_log_change"].to_numpy(dtype=float)
        scores = np.maximum.reduce([lower - actual, actual - upper, np.zeros_like(actual)])
        calibration_records.append(
            {
                "hs_code": keys[0],
                "horizon_months": keys[1],
                "model": keys[2],
                "calibration_rows": len(group),
                "conformal_p05_p95_expansion": conformal_quantile(scores),
            }
        )
    calibration = pd.DataFrame(calibration_records)
    predictions = predictions.merge(
        calibration,
        on=["hs_code", "horizon_months", "model"],
        how="left",
        validate="many_to_one",
    )

    calibrated_rows: list[dict[str, float]] = []
    for _, row in predictions.iterrows():
        raw = {q: float(row[f"raw_{quantile_suffix(q)}_log_change"]) for q in QUANTILES}
        calibrated_rows.append(
            prediction_record_values(
                apply_calibration(raw, float(row["conformal_p05_p95_expansion"]))
            )
        )
    predictions = pd.concat([predictions.reset_index(drop=True), pd.DataFrame(calibrated_rows)], axis=1)

    comparison_records: list[dict[str, object]] = []
    for keys, group in predictions.groupby(["hs_code", "horizon_months", "model"]):
        pinball = np.mean(
            [
                mean_pinball_loss(
                    group["actual_log_change"],
                    group[f"raw_{quantile_suffix(q)}_log_change"],
                    alpha=q,
                )
                for q in QUANTILES
            ]
        )
        covered = (
            (group["actual_log_change"] >= group["p05_log_change"])
            & (group["actual_log_change"] <= group["p95_log_change"])
        )
        mis = [
            interval_score(actual, lower, upper)
            for actual, lower, upper in zip(
                group["actual_log_change"],
                group["p05_log_change"],
                group["p95_log_change"],
            )
        ]
        comparison_records.append(
            {
                "hs_code": keys[0],
                "horizon_months": keys[1],
                "model": keys[2],
                "validation_forecasts": len(group),
                "raw_mean_pinball_loss": float(pinball),
                "calibrated_p05_p95_coverage": float(covered.mean()),
                "calibrated_mean_interval_score": float(np.mean(mis)),
                "conformal_p05_p95_expansion": float(
                    group["conformal_p05_p95_expansion"].iloc[0]
                ),
            }
        )
    comparison = pd.DataFrame(comparison_records)

    selected: dict[tuple[int, int], Candidate] = {}
    for hs_code in HS_CODES:
        for horizon in HORIZONS:
            choices = comparison[
                (comparison["hs_code"] == hs_code)
                & (comparison["horizon_months"] == horizon)
                & (comparison["model"] != BASELINE_NAME)
            ]
            winner_name = str(
                choices.sort_values("raw_mean_pinball_loss").iloc[0]["model"]
            )
            selected[(hs_code, horizon)] = candidate_by_name[winner_name]
    comparison["selected_by_pretest_validation"] = comparison.apply(
        lambda row: row["model"]
        == selected.get((int(row["hs_code"]), int(row["horizon_months"])), Candidate("", "", "")).name,
        axis=1,
    )

    predictions.to_csv(RESULTS_DIR / "pretest_validation_predictions.csv", index=False)
    comparison.sort_values(
        ["hs_code", "horizon_months", "raw_mean_pinball_loss"]
    ).to_csv(RESULTS_DIR / "pretest_validation_model_comparison.csv", index=False)
    calibration.to_csv(RESULTS_DIR / "conformal_calibration_offsets.csv", index=False)
    return predictions, comparison, calibration, selected


def calibration_lookup(calibration: pd.DataFrame) -> dict[tuple[int, int, str], float]:
    return {
        (int(row.hs_code), int(row.horizon_months), str(row.model)): float(
            row.conformal_p05_p95_expansion
        )
        for row in calibration.itertuples(index=False)
    }


def run_disruption_test(
    data: pd.DataFrame,
    calibration: pd.DataFrame,
    selected: dict[tuple[int, int], Candidate],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offsets = calibration_lookup(calibration)
    records: list[dict[str, object]] = []

    for hs_code in HS_CODES:
        hs_data = data[data["hs_code"] == hs_code].sort_values("date")
        test = hs_data[
            (hs_data["period_m"] >= TEST_START) & (hs_data["period_m"] <= TEST_END)
        ]
        for horizon in HORIZONS:
            for _, row in test.iterrows():
                cutoff = row["trade_data_available_date_assumed"]
                train = eligible_training_rows(hs_data, horizon, cutoff)
                actual = row[target_column(horizon)]
                output_available = bool(
                    pd.notna(actual)
                    and row[FEATURES].notna().all()
                    and len(train) >= MIN_TRAINING_ROWS
                )
                common: dict[str, object] = {
                    "hs_code": hs_code,
                    "hs_product": row["hs_product"],
                    "horizon_months": horizon,
                    "forecast_origin": str(row["period_m"]),
                    "target_month": str(row["period_m"] + horizon),
                    "forecast_data_available_date": cutoff.date().isoformat(),
                    "actual_log_change_gbp_per_kg": actual,
                    "actual_origin_price_gbp_per_kg": row["import_unit_value_gbp_per_kg"],
                    "actual_target_price_gbp_per_kg": row[target_price_column(horizon)],
                    "training_rows": len(train),
                    "model_output_available": output_available,
                }
                if not output_available:
                    for model_name in [candidate.name for candidate in CANDIDATES] + [BASELINE_NAME]:
                        records.append(
                            {
                                **common,
                                "model": model_name,
                                "selected_by_pretest_validation": model_name
                                == selected[(hs_code, horizon)].name,
                            }
                        )
                    continue

                for candidate in CANDIDATES:
                    fitted = fit_candidate(candidate, train, horizon)
                    raw = predict_candidate(fitted, row.to_frame().T)
                    calibrated = apply_calibration(
                        raw, offsets[(hs_code, horizon, candidate.name)]
                    )
                    record = {
                        **common,
                        "model": candidate.name,
                        "selected_by_pretest_validation": candidate.name
                        == selected[(hs_code, horizon)].name,
                    }
                    record.update(prediction_record_values(calibrated))
                    for quantile in QUANTILES:
                        suffix = quantile_suffix(quantile)
                        record[f"{suffix}_price_gbp_per_kg"] = float(
                            row["import_unit_value_gbp_per_kg"]
                            * np.exp(calibrated[quantile])
                        )
                    records.append(record)

                baseline_raw = baseline_quantiles(train[target_column(horizon)])
                baseline = apply_calibration(
                    baseline_raw, offsets[(hs_code, horizon, BASELINE_NAME)]
                )
                baseline_record = {
                    **common,
                    "model": BASELINE_NAME,
                    "selected_by_pretest_validation": False,
                }
                baseline_record.update(prediction_record_values(baseline))
                for quantile in QUANTILES:
                    suffix = quantile_suffix(quantile)
                    baseline_record[f"{suffix}_price_gbp_per_kg"] = float(
                        row["import_unit_value_gbp_per_kg"]
                        * np.exp(baseline[quantile])
                    )
                records.append(baseline_record)

    predictions = pd.DataFrame(records)
    predictions["covered_p05_p95"] = (
        predictions["p05_log_change"].notna()
        & (predictions["actual_log_change_gbp_per_kg"] >= predictions["p05_log_change"])
        & (predictions["actual_log_change_gbp_per_kg"] <= predictions["p95_log_change"])
    )
    predictions["interval_score_alpha_010"] = predictions.apply(
        lambda row: interval_score(
            row["actual_log_change_gbp_per_kg"],
            row["p05_log_change"],
            row["p95_log_change"],
        )
        if pd.notna(row.get("p05_log_change"))
        else np.nan,
        axis=1,
    )
    predictions = predictions.sort_values(
        ["hs_code", "horizon_months", "model", "forecast_origin"]
    ).reset_index(drop=True)

    metric_records: list[dict[str, object]] = []
    for keys, group in predictions.groupby(["hs_code", "horizon_months", "model"]):
        group = group.sort_values("forecast_origin").copy()
        expected = len(group)
        outputs = int(group["model_output_available"].sum())
        valid = group[group["model_output_available"]].copy()
        coverage = float(group["covered_p05_p95"].sum() / expected)
        mean_is = float(group["interval_score_alpha_010"].fillna(np.inf).mean())
        pinball = float(
            np.mean(
                [
                    mean_pinball_loss(
                        valid["actual_log_change_gbp_per_kg"],
                        valid[f"{quantile_suffix(q)}_log_change"],
                        alpha=q,
                    )
                    for q in QUANTILES
                ]
            )
        )
        median_mae = float(
            mean_absolute_error(
                valid["actual_log_change_gbp_per_kg"], valid["p50_log_change"]
            )
        )
        group["rolling_6m_coverage"] = (
            group["covered_p05_p95"].astype(float).rolling(6, min_periods=6).mean()
        )
        recovered = group[
            (group["forecast_origin"] <= "2022-08")
            & (group["rolling_6m_coverage"] >= 0.90)
        ]
        recovery_month = None if recovered.empty else str(recovered.iloc[0]["forecast_origin"])
        metric_records.append(
            {
                "hs_code": keys[0],
                "horizon_months": keys[1],
                "model": keys[2],
                "selected_by_pretest_validation": bool(
                    group["selected_by_pretest_validation"].iloc[0]
                ),
                "expected_test_origins": expected,
                "model_outputs": outputs,
                "no_output_misses": expected - outputs,
                "p05_p95_coverage": coverage,
                "mean_interval_score_alpha_010": mean_is,
                "median_mae_log_change": median_mae,
                "mean_pinball_loss": pinball,
                "coverage_requirement_met": coverage >= 0.90,
                "first_six_month_coverage_recovery": recovery_month,
                "coverage_recovery_by_2022_08": recovery_month is not None,
            }
        )

    metrics = pd.DataFrame(metric_records)
    baseline_mis = metrics[metrics["model"] == BASELINE_NAME][
        ["hs_code", "horizon_months", "mean_interval_score_alpha_010"]
    ].rename(columns={"mean_interval_score_alpha_010": "baseline_mean_interval_score_alpha_010"})
    metrics = metrics.merge(
        baseline_mis, on=["hs_code", "horizon_months"], how="left", validate="many_to_one"
    )
    metrics["mis_ratio_to_baseline"] = (
        metrics["mean_interval_score_alpha_010"]
        / metrics["baseline_mean_interval_score_alpha_010"]
    )
    metrics["mis_beats_baseline"] = metrics["mis_ratio_to_baseline"] < 1.0
    metrics.loc[metrics["model"] == BASELINE_NAME, "mis_beats_baseline"] = False
    metrics["passes_spec_h3_h6"] = (
        metrics["horizon_months"].isin([3, 6])
        & metrics["coverage_requirement_met"]
        & metrics["mis_beats_baseline"]
        & (metrics["no_output_misses"] == 0)
        & metrics["coverage_recovery_by_2022_08"]
    )
    metrics["test_mis_rank_among_ml"] = np.nan
    for (hs_code, horizon), indices in metrics[metrics["model"] != BASELINE_NAME].groupby(
        ["hs_code", "horizon_months"]
    ).groups.items():
        ranked = metrics.loc[list(indices), "mean_interval_score_alpha_010"].rank(
            method="min"
        )
        metrics.loc[list(indices), "test_mis_rank_among_ml"] = ranked

    predictions.to_csv(RESULTS_DIR / "disruption_test_predictions_all_models.csv", index=False)
    metrics.sort_values(["hs_code", "horizon_months", "model"]).to_csv(
        RESULTS_DIR / "disruption_test_metrics_all_models.csv", index=False
    )
    return predictions, metrics


def fit_reference_models(
    data: pd.DataFrame,
    calibration: pd.DataFrame,
    selected: dict[tuple[int, int], Candidate],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    offsets = calibration_lookup(calibration)
    forecast_records: list[dict[str, object]] = []
    importance_records: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "model_name": "Titanium Model A GBP import-price change with ERUC",
        "status": "research model; deployment requires acceptance review",
        "random_seed": SEED,
        "quantiles": list(QUANTILES),
        "horizons_months": list(HORIZONS),
        "features": FEATURES,
        "selection_period": "blocked validation ending 2022-01",
        "frozen_test_period": "2022-02 through 2023-12",
        "calibration": "split conformal expansion of P05-P95 using pre-test validation residuals",
        "specification_deviation": (
            "ERUC replaces the unavailable/incomplete titanium-specific external proxy; "
            "World Bank aluminium/nickel and USGS sponge features are not in this feature set."
        ),
        "hs_models": {},
    }

    for hs_code in HS_CODES:
        hs_data = data[data["hs_code"] == hs_code].sort_values("date")
        origin = hs_data.iloc[-1]
        bundle: dict[str, object] = {
            "hs_code": hs_code,
            "hs_product": origin["hs_product"],
            "features": FEATURES,
            "quantiles": QUANTILES,
            "forecast_origin": str(origin["period_m"]),
            "forecast_data_available_date": origin[
                "trade_data_available_date_assumed"
            ].date().isoformat(),
            "horizons": {},
        }
        hs_manifest: dict[str, object] = {}
        for horizon in HORIZONS:
            candidate = selected[(hs_code, horizon)]
            train = eligible_training_rows(
                hs_data, horizon, origin["trade_data_available_date_assumed"]
            )
            fitted = fit_candidate(candidate, train, horizon)
            offset = offsets[(hs_code, horizon, candidate.name)]
            raw = predict_candidate(fitted, origin.to_frame().T)
            prediction = apply_calibration(raw, offset)
            bundle["horizons"][horizon] = {
                "candidate": asdict(candidate),
                "training_rows": len(train),
                "conformal_p05_p95_expansion": offset,
                "fitted": fitted,
            }
            hs_manifest[f"h{horizon}"] = {
                "candidate": candidate.name,
                "training_rows": len(train),
                "conformal_p05_p95_expansion": offset,
            }
            record: dict[str, object] = {
                "hs_code": hs_code,
                "hs_product": origin["hs_product"],
                "forecast_origin": str(origin["period_m"]),
                "target_month": str(origin["period_m"] + horizon),
                "horizon_months": horizon,
                "selected_model": candidate.name,
                "origin_price_gbp_per_kg": float(origin["import_unit_value_gbp_per_kg"]),
                "training_rows": len(train),
            }
            record.update(prediction_record_values(prediction))
            for quantile in QUANTILES:
                suffix = quantile_suffix(quantile)
                record[f"{suffix}_price_gbp_per_kg"] = float(
                    origin["import_unit_value_gbp_per_kg"]
                    * np.exp(prediction[quantile])
                )
            forecast_records.append(record)

            importance = feature_importance(candidate, fitted)
            for feature, value in importance.items():
                importance_records.append(
                    {
                        "hs_code": hs_code,
                        "horizon_months": horizon,
                        "selected_model": candidate.name,
                        "feature": feature,
                        "relative_importance_pct": value,
                    }
                )

        model_path = MODELS_DIR / f"titanium_model_a_eruc_hs{hs_code}_selected_bundle.joblib"
        joblib.dump(bundle, model_path)
        manifest["hs_models"][str(hs_code)] = {
            "file": model_path.name,
            "product": origin["hs_product"],
            "reference_origin": str(origin["period_m"]),
            "horizons": hs_manifest,
        }

    forecasts = pd.DataFrame(forecast_records)
    importance = pd.DataFrame(importance_records)
    forecasts.to_csv(RESULTS_DIR / "latest_reference_forecasts.csv", index=False)
    (RESULTS_DIR / "latest_reference_forecasts.json").write_text(
        json.dumps(forecasts.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    importance.to_csv(RESULTS_DIR / "selected_model_feature_importance.csv", index=False)
    (RESULTS_DIR / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return forecasts, importance, manifest


def feature_importance(
    candidate: Candidate, fitted: dict[str, object]
) -> dict[str, float]:
    if candidate.family == "linear":
        model = fitted["quantile_models"][0.50].named_steps["model"]
        raw = np.abs(np.asarray(model.coef_, dtype=float))
    elif candidate.family == "gradient_boosting":
        model = fitted["quantile_models"][0.50].named_steps["model"]
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif candidate.family == "random_forest":
        raw = np.asarray(fitted["model"].feature_importances_, dtype=float)
    else:
        raise ValueError(candidate.family)
    total = float(raw.sum())
    if total <= 0:
        values = np.zeros_like(raw)
    else:
        values = raw / total * 100.0
    return {feature: float(value) for feature, value in zip(FEATURES, values)}


def save_selected_summary(
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    selected: dict[tuple[int, int], Candidate],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for hs_code in HS_CODES:
        for horizon in HORIZONS:
            candidate = selected[(hs_code, horizon)]
            validation = comparison[
                (comparison["hs_code"] == hs_code)
                & (comparison["horizon_months"] == horizon)
                & (comparison["model"] == candidate.name)
            ].iloc[0]
            test = metrics[
                (metrics["hs_code"] == hs_code)
                & (metrics["horizon_months"] == horizon)
                & (metrics["model"] == candidate.name)
            ].iloc[0]
            records.append(
                {
                    "hs_code": hs_code,
                    "horizon_months": horizon,
                    "validation_selected_model": candidate.name,
                    "validation_mean_pinball_loss": validation["raw_mean_pinball_loss"],
                    "test_p05_p95_coverage": test["p05_p95_coverage"],
                    "test_mean_interval_score": test["mean_interval_score_alpha_010"],
                    "baseline_mean_interval_score": test[
                        "baseline_mean_interval_score_alpha_010"
                    ],
                    "test_mis_ratio_to_baseline": test["mis_ratio_to_baseline"],
                    "test_median_mae_log_change": test["median_mae_log_change"],
                    "coverage_recovery_by_2022_08": test[
                        "coverage_recovery_by_2022_08"
                    ],
                    "passes_spec_h3_h6": test["passes_spec_h3_h6"],
                    "recommendation": (
                        "eligible for further review"
                        if bool(test["passes_spec_h3_h6"])
                        else "research only - acceptance criteria not met"
                    ),
                }
            )
    summary = pd.DataFrame(records)
    summary.to_csv(RESULTS_DIR / "selected_model_summary.csv", index=False)
    return summary


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "#f4f0e8",
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#b9ad99",
            "grid.color": "#ded5c6",
            "grid.alpha": 0.75,
        }
    )


MODEL_COLORS = {
    "linear_quantile": "#0d7c80",
    "gradient_boosting_quantile": "#d05c2b",
    "random_forest_distribution": "#d39b28",
    "no_change_baseline": "#66757d",
}


def plot_validation_comparison(comparison: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    for row_index, hs_code in enumerate(HS_CODES):
        for column_index, horizon in enumerate(HORIZONS):
            ax = axes[row_index, column_index]
            group = comparison[
                (comparison["hs_code"] == hs_code)
                & (comparison["horizon_months"] == horizon)
                & (comparison["model"] != BASELINE_NAME)
            ].sort_values("raw_mean_pinball_loss")
            bars = ax.bar(
                group["model"],
                group["raw_mean_pinball_loss"],
                color=[MODEL_COLORS[name] for name in group["model"]],
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
            ax.set_title(f"HS {hs_code} | H{horizon}")
            ax.set_ylabel("Mean pinball loss")
            ax.tick_params(axis="x", rotation=20, labelsize=8)
            ax.grid(axis="y")
    fig.suptitle("Pre-test validation: lower pinball loss is better", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = VIS_DIR / "pretest_validation_model_comparison.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_test_coverage(metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    order = [candidate.name for candidate in CANDIDATES] + [BASELINE_NAME]
    for row_index, hs_code in enumerate(HS_CODES):
        for column_index, horizon in enumerate(HORIZONS):
            ax = axes[row_index, column_index]
            group = metrics[
                (metrics["hs_code"] == hs_code)
                & (metrics["horizon_months"] == horizon)
            ].set_index("model").reindex(order).reset_index()
            bars = ax.bar(
                group["model"],
                group["p05_p95_coverage"] * 100,
                color=[MODEL_COLORS[name] for name in group["model"]],
            )
            ax.axhline(90, color="#9b2f2f", linestyle="--", linewidth=1.5)
            ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
            ax.set_ylim(0, 110)
            ax.set_title(f"HS {hs_code} | H{horizon}")
            ax.set_ylabel("P05-P95 coverage")
            ax.tick_params(axis="x", rotation=22, labelsize=8)
            ax.grid(axis="y")
    fig.suptitle("Frozen disruption test coverage (target: at least 90%)", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = VIS_DIR / "disruption_test_coverage_comparison.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_test_mis(metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    order = [candidate.name for candidate in CANDIDATES]
    for row_index, hs_code in enumerate(HS_CODES):
        for column_index, horizon in enumerate(HORIZONS):
            ax = axes[row_index, column_index]
            group = metrics[
                (metrics["hs_code"] == hs_code)
                & (metrics["horizon_months"] == horizon)
                & (metrics["model"].isin(order))
            ].set_index("model").reindex(order).reset_index()
            bars = ax.bar(
                group["model"],
                group["mis_ratio_to_baseline"],
                color=[MODEL_COLORS[name] for name in group["model"]],
            )
            ax.axhline(1.0, color="#9b2f2f", linestyle="--", linewidth=1.5)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
            ax.set_title(f"HS {hs_code} | H{horizon}")
            ax.set_ylabel("MIS / baseline MIS")
            ax.tick_params(axis="x", rotation=22, labelsize=8)
            ax.grid(axis="y")
    fig.suptitle("Frozen disruption test sharpness: below 1.0 beats baseline", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = VIS_DIR / "disruption_test_mis_ratio.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selected_actual_vs_forecast(predictions: pd.DataFrame) -> Path:
    selected = predictions[predictions["selected_by_pretest_validation"]].copy()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=False)
    for row_index, hs_code in enumerate(HS_CODES):
        for column_index, horizon in enumerate(HORIZONS):
            ax = axes[row_index, column_index]
            group = selected[
                (selected["hs_code"] == hs_code)
                & (selected["horizon_months"] == horizon)
            ].sort_values("target_month")
            x = pd.to_datetime(group["target_month"], format="%Y-%m")
            ax.fill_between(
                x,
                group["p05_price_gbp_per_kg"].astype(float),
                group["p95_price_gbp_per_kg"].astype(float),
                color="#9fc9c5",
                alpha=0.45,
                label="90% range",
            )
            ax.plot(
                x,
                group["actual_target_price_gbp_per_kg"],
                color="#173a42",
                linewidth=2,
                label="Actual",
            )
            ax.plot(
                x,
                group["p50_price_gbp_per_kg"],
                color="#d05c2b",
                linewidth=1.8,
                label="Median forecast",
            )
            model_name = group["model"].iloc[0] if not group.empty else "N/A"
            ax.set_title(f"HS {hs_code} | H{horizon} | {model_name}")
            ax.set_ylabel("GBP/kg")
            ax.grid(axis="both")
            ax.tick_params(axis="x", rotation=25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Validation-selected models: actual price versus probabilistic forecast", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = VIS_DIR / "selected_models_actual_vs_forecast.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rolling_coverage(predictions: pd.DataFrame) -> Path:
    selected = predictions[
        predictions["selected_by_pretest_validation"]
        & predictions["horizon_months"].isin([3, 6])
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for row_index, hs_code in enumerate(HS_CODES):
        for column_index, horizon in enumerate((3, 6)):
            ax = axes[row_index, column_index]
            group = selected[
                (selected["hs_code"] == hs_code)
                & (selected["horizon_months"] == horizon)
            ].sort_values("forecast_origin")
            rolling = group["covered_p05_p95"].astype(float).rolling(6, min_periods=6).mean()
            x = pd.to_datetime(group["forecast_origin"], format="%Y-%m")
            ax.plot(x, rolling * 100, marker="o", color="#0d7c80", linewidth=2)
            ax.axhline(90, color="#9b2f2f", linestyle="--")
            ax.axvline(pd.Timestamp("2022-08-01"), color="#d39b28", linestyle=":")
            ax.set_ylim(0, 105)
            ax.set_title(f"HS {hs_code} | H{horizon}")
            ax.set_ylabel("Rolling six-month coverage")
            ax.grid(axis="both")
    fig.suptitle("Coverage recovery during the 2022-2023 disruption test", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = VIS_DIR / "selected_models_rolling_six_month_coverage.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(importance: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for row_index, hs_code in enumerate(HS_CODES):
        for column_index, horizon in enumerate(HORIZONS):
            ax = axes[row_index, column_index]
            group = importance[
                (importance["hs_code"] == hs_code)
                & (importance["horizon_months"] == horizon)
            ].nlargest(10, "relative_importance_pct").sort_values("relative_importance_pct")
            ax.barh(group["feature"], group["relative_importance_pct"], color="#0d7c80")
            model_name = group["selected_model"].iloc[0] if not group.empty else "N/A"
            ax.set_title(f"HS {hs_code} | H{horizon} | {model_name}")
            ax.set_xlabel("Relative importance (%)")
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x")
    fig.suptitle("Reference-fit feature influence for validation-selected models", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = VIS_DIR / "selected_model_feature_importance.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latest_forecasts(forecasts: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, hs_code in zip(axes, HS_CODES):
        group = forecasts[forecasts["hs_code"] == hs_code].sort_values("horizon_months")
        x = np.arange(len(group))
        med = group["p50_price_gbp_per_kg"].to_numpy(float)
        lower = med - group["p05_price_gbp_per_kg"].to_numpy(float)
        upper = group["p95_price_gbp_per_kg"].to_numpy(float) - med
        ax.errorbar(
            x,
            med,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            color="#d05c2b",
            ecolor="#0d7c80",
            elinewidth=5,
            capsize=8,
        )
        ax.set_xticks(x, [f"H{h}\n{month}" for h, month in zip(group["horizon_months"], group["target_month"])])
        ax.set_title(f"HS {hs_code}: {group['hs_product'].iloc[0]}")
        ax.set_ylabel("Forecast GBP/kg")
        ax.grid(axis="y")
    fig.suptitle("Latest direct H3, H6 and H12 reference forecasts", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = VIS_DIR / "latest_reference_forecasts.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def format_summary_table(summary: pd.DataFrame) -> str:
    display = summary.copy()
    display["validation_mean_pinball_loss"] = display["validation_mean_pinball_loss"].map("{:.4f}".format)
    display["test_p05_p95_coverage"] = display["test_p05_p95_coverage"].map(lambda x: f"{x:.1%}")
    display["test_mean_interval_score"] = display["test_mean_interval_score"].map("{:.4f}".format)
    display["baseline_mean_interval_score"] = display["baseline_mean_interval_score"].map("{:.4f}".format)
    display["test_mis_ratio_to_baseline"] = display["test_mis_ratio_to_baseline"].map("{:.2f}".format)
    display["passes_spec_h3_h6"] = display["passes_spec_h3_h6"].map(lambda x: "PASS" if x else "FAIL / H12 report-only")
    display = display.rename(
        columns={
            "hs_code": "HS code",
            "horizon_months": "Horizon",
            "validation_selected_model": "Pre-test selected model",
            "validation_mean_pinball_loss": "Validation pinball",
            "test_p05_p95_coverage": "Test coverage",
            "test_mean_interval_score": "Model MIS",
            "baseline_mean_interval_score": "Baseline MIS",
            "test_mis_ratio_to_baseline": "MIS ratio",
            "passes_spec_h3_h6": "Acceptance",
            "recommendation": "Recommendation",
        }
    )
    keep = [
        "HS code",
        "Horizon",
        "Pre-test selected model",
        "Validation pinball",
        "Test coverage",
        "Model MIS",
        "Baseline MIS",
        "MIS ratio",
        "Acceptance",
        "Recommendation",
    ]
    return display[keep].to_html(index=False, border=0, classes="data-table")


def format_forecast_table(forecasts: pd.DataFrame) -> str:
    display = forecasts[
        [
            "hs_code",
            "horizon_months",
            "forecast_origin",
            "target_month",
            "selected_model",
            "origin_price_gbp_per_kg",
            "p05_price_gbp_per_kg",
            "p50_price_gbp_per_kg",
            "p95_price_gbp_per_kg",
        ]
    ].copy()
    for column in [
        "origin_price_gbp_per_kg",
        "p05_price_gbp_per_kg",
        "p50_price_gbp_per_kg",
        "p95_price_gbp_per_kg",
    ]:
        display[column] = display[column].map(lambda x: f"GBP {x:,.2f}/kg")
    display = display.rename(
        columns={
            "hs_code": "HS code",
            "horizon_months": "Horizon",
            "forecast_origin": "Origin",
            "target_month": "Target month",
            "selected_model": "Model",
            "origin_price_gbp_per_kg": "Origin price",
            "p05_price_gbp_per_kg": "P05",
            "p50_price_gbp_per_kg": "P50",
            "p95_price_gbp_per_kg": "P95",
        }
    )
    return display.to_html(index=False, border=0, classes="data-table")


def build_dashboard(
    summary: pd.DataFrame,
    forecasts: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    selected_h36 = summary[summary["horizon_months"].isin([3, 6])]
    pass_count = int(selected_h36["passes_spec_h3_h6"].sum())
    cards = [
        ("Test combinations passing", f"{pass_count} / 4"),
        ("Forecast origins per test", "23"),
        ("Quantiles per forecast", "7"),
        ("Model families compared", "3 + baseline"),
    ]
    card_html = "".join(
        f'<div class="card"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in cards
    )
    figures = "".join(
        f'<figure><img src="{image_data_uri(path)}" alt="{path.stem}"><figcaption>{path.stem.replace("_", " ")}</figcaption></figure>'
        for path in plot_paths
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Titanium Model A - ERUC experiment</title>
<style>
:root {{ --ink:#173a42; --teal:#0d7c80; --orange:#d05c2b; --gold:#d39b28; --paper:#fffdf8; --sand:#f4f0e8; --line:#d8cdbd; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:linear-gradient(135deg,#f4f0e8 0%,#e6efec 100%); font-family:"Trebuchet MS",sans-serif; }}
header {{ padding:54px 6vw 42px; background:var(--ink); color:#fffdf8; position:relative; overflow:hidden; }}
header:after {{ content:""; position:absolute; width:330px; height:330px; border:55px solid var(--teal); border-radius:50%; right:-90px; top:-170px; opacity:.55; }}
.eyebrow {{ color:#efaa70; text-transform:uppercase; letter-spacing:.18em; font-weight:700; font-size:.78rem; }}
h1,h2 {{ font-family:Georgia,serif; font-weight:500; margin:.25em 0; }}
h1 {{ font-size:clamp(2.3rem,5vw,4.9rem); max-width:1000px; }}
header p {{ max-width:900px; color:#c9dad8; font-size:1.08rem; line-height:1.65; }}
main {{ max-width:1500px; margin:auto; padding:34px 3vw 70px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:-58px; position:relative; z-index:2; }}
.card {{ background:var(--paper); border:1px solid var(--line); padding:22px; box-shadow:0 14px 36px rgba(23,58,66,.10); }}
.card span {{ display:block; font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; color:#66757d; }}
.card strong {{ display:block; margin-top:8px; font-family:Georgia,serif; font-size:2rem; color:var(--teal); }}
section {{ background:var(--paper); border:1px solid var(--line); padding:30px; margin-top:25px; box-shadow:0 10px 28px rgba(23,58,66,.07); }}
h2 {{ font-size:2rem; }}
.note {{ border-left:5px solid var(--orange); padding:15px 18px; background:#fff4eb; line-height:1.55; }}
.data-table {{ width:100%; border-collapse:collapse; font-size:.88rem; margin-top:18px; }}
.data-table th {{ background:var(--ink); color:white; padding:10px; text-align:left; }}
.data-table td {{ border-bottom:1px solid var(--line); padding:9px 10px; vertical-align:top; }}
.data-table tr:nth-child(even) td {{ background:#f7f3ec; }}
figure {{ margin:24px 0 0; }}
figure img {{ width:100%; height:auto; border:1px solid var(--line); background:white; }}
figcaption {{ text-transform:capitalize; color:#66757d; font-size:.82rem; margin-top:6px; }}
.method {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
.method div {{ border-top:4px solid var(--teal); padding:14px; background:#f2f7f5; line-height:1.5; }}
code {{ background:#e9eee9; padding:2px 5px; }}
@media (max-width:900px) {{ .cards,.method {{ grid-template-columns:1fr; }} .cards {{ margin-top:18px; }} section {{ padding:18px; overflow-x:auto; }} }}
</style>
</head>
<body>
<header>
<div class="eyebrow">SCI Model A / research disruption test</div>
<h1>Titanium import-price movement, with uncertainty</h1>
<p>Separate direct H3, H6 and H12 forecasts for HS 810820 and HS 810890. Models predict GBP/kg log change and return seven non-crossing quantiles. Selection uses only pre-February-2022 validation; February 2022 to December 2023 remains the frozen disruption test.</p>
</header>
<main>
<div class="cards">{card_html}</div>
<section>
<div class="eyebrow">Decision summary</div><h2>Validation-selected models and frozen-test results</h2>
<p class="note"><strong>Selection boundary:</strong> the recommended family for each HS/horizon is selected only from blocked pre-test validation. Test-period rankings are shown for diagnosis and are not used to switch the selected model. H12 is report-only under the specification.</p>
{format_summary_table(summary)}
</section>
<section>
<div class="eyebrow">Latest reference output</div><h2>Direct probabilistic forecasts from the latest complete feature month</h2>
{format_forecast_table(forecasts)}
<p>The P05-P95 columns are the calibrated 90% prediction range. P50 is the median forecast. These are research outputs, not approved production forecasts.</p>
</section>
<section>
<div class="eyebrow">Method</div><h2>How the experiment works</h2>
<div class="method">
<div><strong>Three model families</strong><br>Linear quantile regression, gradient-boosted quantile regression and a random-forest tree distribution are compared against a no-change baseline.</div>
<div><strong>Honest timing</strong><br>Each rolling forecast trains only on targets whose assumed publication date is no later than the forecast date. HS codes are never pooled.</div>
<div><strong>Calibrated uncertainty</strong><br>Pre-test residuals provide a split-conformal expansion for the P05-P95 range. Quantiles are monotonically ordered after prediction.</div>
</div>
<p class="note"><strong>Specification deviation:</strong> this experiment uses ONS ERUC as the added external metal-inflation feature. It does not include the specification's World Bank aluminium/nickel or USGS titanium-sponge inputs. Results therefore assess the approved ERUC feature substitution, not literal full-spec compliance.</p>
</section>
<section><div class="eyebrow">Evidence</div><h2>Validation, disruption test and interpretation</h2>{figures}</section>
</main>
</body></html>"""
    path = VIS_DIR / "titanium_model_a_eruc_dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_readme(
    summary: pd.DataFrame,
    forecasts: pd.DataFrame,
    dashboard_path: Path,
) -> None:
    pass_count = int(
        summary[
            summary["horizon_months"].isin([3, 6])
            & summary["passes_spec_h3_h6"]
        ].shape[0]
    )
    text = f"""# Titanium Model A - FX and ERUC Experiment

## Purpose

Predict the distribution of the direct GBP/kg titanium import-price change at H3, H6 and H12. HS 810820 and HS 810890 are modelled separately. H3 and H6 determine acceptance; H12 is report-only.

## Data

- Source training CSV: `{SOURCE_DATA_PATH}`
- Coverage: January 2010 to June 2026
- Bank of England XUMAUSS converts USD values to GBP.
- ONS ERUC supplies broader imported non-ferrous-metal inflation changes.
- This experiment intentionally does not use World Bank aluminium/nickel or USGS titanium sponge, so it is not literal full-spec compliance.

## Models

1. Regularised linear quantile regression.
2. Gradient-boosted quantile regression.
3. Random-forest predictive distribution using quantiles across trees.
4. Calibrated no-change baseline for comparison.

Every forecast returns P05, P10, P25, P50, P75, P90 and P95. A split-conformal adjustment based only on blocked pre-test residuals calibrates the P05-P95 range.

## Evaluation

- Model selection: five blocked validation windows ending January 2022.
- Frozen disruption test: monthly origins from February 2022 to December 2023.
- Rolling retraining uses only targets available at each forecast date.
- Acceptance at H3/H6 requires at least 90% P05-P95 coverage, MIS below the baseline, no missing output and six-month coverage recovery by August 2022.
- Validation-selected H3/H6 combinations passing all rules: **{pass_count}/4**.

## Outputs

- Dashboard: `{dashboard_path}`
- Validation and test metrics: `results/`
- Validation-selected model bundles: `models/`
- Charts: `visualizations/`
- Latest direct reference forecasts: `results/latest_reference_forecasts.csv` and `.json`

## Run

```powershell
python "{HERE}"
```

Random seed: {SEED}.
"""
    (RUN_DIR / "README.md").write_text(text, encoding="utf-8")


def write_methodology() -> None:
    text = """# Methodology and Interpretation

## Target

For horizon h, the target is `ln(GBP_price[t+h]) - ln(GBP_price[t])`. H3, H6 and H12 are fitted directly; they are not recursive H1 forecasts.

## Feature set

The predictors are GBP price changes over 1, 2, 3, 6 and 12 months; quantity changes over 1, 3 and 12 months; fixed supplier-country shares; ERUC changes over 1, 3, 6 and 12 months; and the January 2021 EU collection-change flag.

## Selection and test boundary

Five blocked validation windows ending January 2022 select the model family by mean seven-quantile pinball loss. The selected family is frozen before the February 2022-December 2023 disruption test. Test rankings are diagnostic and do not change the selected family.

## Calibration

For each HS code, horizon and model family, out-of-sample pre-test validation residuals calculate a finite-sample split-conformal expansion. This expansion is subtracted from P05 and added to P95. It is fixed before the disruption test. Interior quantiles come directly from the fitted model. All quantiles are made non-crossing.

## Baseline

The baseline central forecast is no change. Its surrounding quantiles come from historical target changes available at the forecast date and receive the same pre-test conformal interval expansion.

## Interpretation boundary

Passing the numerical test rules supports further review; it does not establish causal validity or production readiness. ERUC is a broad non-ferrous import-price index, not a titanium benchmark. PPI revisions and assumed publication dates remain limitations.
"""
    (DOCS_DIR / "METHODOLOGY.md").write_text(text, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_directories()
    shutil.copy2(SOURCE_DATA_PATH, DATA_COPY_PATH)
    setup_plot_style()

    data = load_data()
    print(f"Loaded {len(data)} rows from {SOURCE_DATA_PATH.name}")
    print("Running blocked pre-test validation and conformal calibration...")
    _, comparison, calibration, selected = run_pretest_validation(data)
    for key, candidate in sorted(selected.items()):
        print(f"Selected HS {key[0]} H{key[1]}: {candidate.name}")

    print("Running frozen 2022-02 to 2023-12 rolling disruption test...")
    test_predictions, test_metrics = run_disruption_test(data, calibration, selected)
    summary = save_selected_summary(comparison, test_metrics, selected)

    print("Fitting validation-selected reference models at the latest feature origin...")
    forecasts, importance, manifest = fit_reference_models(
        data, calibration, selected
    )

    plot_paths = [
        plot_validation_comparison(comparison),
        plot_test_coverage(test_metrics),
        plot_test_mis(test_metrics),
        plot_selected_actual_vs_forecast(test_predictions),
        plot_rolling_coverage(test_predictions),
        plot_feature_importance(importance),
        plot_latest_forecasts(forecasts),
    ]
    dashboard = build_dashboard(summary, forecasts, plot_paths)
    write_readme(summary, forecasts, dashboard)
    write_methodology()

    pass_count = int(
        summary[
            summary["horizon_months"].isin([3, 6])
            & summary["passes_spec_h3_h6"]
        ].shape[0]
    )
    print("=" * 78)
    print("Training pipeline complete")
    print(f"H3/H6 validation-selected combinations passing all rules: {pass_count}/4")
    print(f"Dashboard: {dashboard}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Models: {MODELS_DIR}")


if __name__ == "__main__":
    main()

