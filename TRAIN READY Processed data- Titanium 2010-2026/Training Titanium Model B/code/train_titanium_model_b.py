from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanium_model_b_country_month_training_2010_2026.csv"
THRESHOLD_PATH = ROOT / "data" / "model_b_fixed_class_thresholds.json"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
VIS_DIR = ROOT / "visualizations"
DOCS_DIR = ROOT / "documentation"

RANDOM_SEED = 42
CLASSES = ["collapse", "fall", "stable", "rise"]
HORIZONS = (3, 6, 12)
PRIMARY_HORIZONS = (3, 6)
NUMERIC_FEATURES = [
    "country_share",
    "log1p_country_quantity_kg",
    "country_share_change_1m",
    "country_share_change_3m",
    "country_share_change_6m",
    "country_share_change_12m",
    "country_quantity_log1p_change_1m",
    "country_quantity_log1p_change_3m",
    "country_quantity_log1p_change_6m",
    "country_quantity_log1p_change_12m",
    "total_import_log_change_1m",
    "total_import_log_change_3m",
    "total_import_log_change_6m",
    "total_import_log_change_12m",
    "titanium_price_log_change_1m",
    "titanium_price_log_change_3m",
    "titanium_price_log_change_6m",
    "titanium_price_log_change_12m",
    "fx_log_change_1m",
    "fx_log_change_3m",
    "fx_log_change_6m",
    "fx_log_change_12m",
    "eruc_log_change_1m",
    "eruc_log_change_3m",
    "eruc_log_change_6m",
    "eruc_log_change_12m",
    "supplier_hhi_weight",
    "top_supplier_share_weight",
    "partner_count_total",
    "month_sin",
    "month_cos",
    "eu_collection_change_flag",
    "weight_reconstructed_any_flag",
    "world_weight_estimated_any_flag",
]
CATEGORICAL_FEATURES = ["country_key"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
VALIDATION_BLOCKS = [
    ("2018-01-01", "2018-12-01"),
    ("2019-01-01", "2019-12-01"),
    ("2020-01-01", "2021-06-01"),
]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params: dict


CANDIDATES = [
    Candidate("multinomial_logistic_c0_1", "logistic", {"C": 0.1, "class_weight": None}),
    Candidate("multinomial_logistic_c1", "logistic", {"C": 1.0, "class_weight": None}),
    Candidate("multinomial_logistic_balanced", "logistic", {"C": 0.3, "class_weight": "balanced"}),
    Candidate(
        "random_forest",
        "random_forest",
        {"n_estimators": 140, "max_depth": 8, "min_samples_leaf": 4},
    ),
    Candidate(
        "extra_trees",
        "extra_trees",
        {"n_estimators": 160, "max_depth": 10, "min_samples_leaf": 3},
    ),
]


def make_pipeline(candidate: Candidate) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if candidate.family == "logistic":
        numeric_steps.append(("scale", StandardScaler()))
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "country",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    if candidate.family == "logistic":
        estimator = LogisticRegression(
            C=candidate.params["C"],
            class_weight=candidate.params["class_weight"],
            max_iter=4000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
    elif candidate.family == "random_forest":
        estimator = RandomForestClassifier(
            **candidate.params,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    elif candidate.family == "extra_trees":
        estimator = ExtraTreesClassifier(
            **candidate.params,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(candidate.family)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def class_indices(values: pd.Series) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(CLASSES)}
    return values.map(mapping).to_numpy(dtype=int)


def aligned_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    aligned = np.zeros((len(features), len(CLASSES)), dtype=float)
    fitted_classes = model.named_steps["model"].classes_
    for source_index, label in enumerate(fitted_classes):
        aligned[:, CLASSES.index(label)] = raw[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sums, out=np.full_like(aligned, 1 / len(CLASSES)), where=row_sums > 0)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    logits = np.log(clipped) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(len(CLASSES))[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def reliability_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    one_hot = np.eye(len(CLASSES))[y_true].ravel()
    predicted = probabilities.ravel()
    bins = np.minimum((predicted * 10).astype(int), 9)
    rows = []
    for bin_index in range(10):
        mask = bins == bin_index
        if not mask.any():
            rows.append(
                {
                    "bin": bin_index + 1,
                    "lower": bin_index / 10,
                    "upper": (bin_index + 1) / 10,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_frequency": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue
        mean_predicted = float(predicted[mask].mean())
        observed = float(one_hot[mask].mean())
        rows.append(
            {
                "bin": bin_index + 1,
                "lower": bin_index / 10,
                "upper": (bin_index + 1) / 10,
                "count": int(mask.sum()),
                "mean_predicted_probability": mean_predicted,
                "observed_frequency": observed,
                "absolute_gap": abs(mean_predicted - observed),
            }
        )
    return pd.DataFrame(rows)


def score_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    reliability = reliability_table(y_true, probabilities)
    populated = reliability[reliability["count"] > 0]
    weights = populated["count"] / populated["count"].sum()
    return {
        "brier_score": multiclass_brier(y_true, probabilities),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(range(len(CLASSES))))),
        "accuracy": float(accuracy_score(y_true, probabilities.argmax(axis=1))),
        "max_calibration_gap": float(populated["absolute_gap"].max()),
        "expected_calibration_error": float((weights * populated["absolute_gap"]).sum()),
        "reliability_bins_populated": int(len(populated)),
    }


def historical_frequency_probabilities(train: pd.DataFrame, prediction: pd.DataFrame, target: str) -> np.ndarray:
    global_counts = train[target].value_counts().reindex(CLASSES, fill_value=0).astype(float)
    if global_counts.sum() == 0:
        global_frequency = np.full(len(CLASSES), 1 / len(CLASSES))
    else:
        global_frequency = (global_counts / global_counts.sum()).to_numpy()
    country_frequencies = {}
    for country, group in train.groupby("country_key"):
        counts = group[target].value_counts().reindex(CLASSES, fill_value=0).astype(float)
        country_frequencies[country] = (
            (counts / counts.sum()).to_numpy() if counts.sum() else global_frequency
        )
    return np.vstack(
        [country_frequencies.get(country, global_frequency) for country in prediction["country_key"]]
    )


def persistence_probabilities(prediction: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(fallback)
    for row_index, label in enumerate(prediction["last_observed_3m_class"]):
        if pd.isna(label) or label not in CLASSES:
            probabilities[row_index] = fallback[row_index]
        else:
            probabilities[row_index, CLASSES.index(label)] = 1.0
    return probabilities


def load_data() -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    for column in [
        "date",
        "origin_data_available_date_assumed",
        *[f"target_month_h{h}" for h in HORIZONS],
        *[f"target_data_available_date_assumed_h{h}" for h in HORIZONS],
    ]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    missing = sorted(set(FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Prepared data is missing model features: {missing}")
    if data.duplicated(["date", "country_key"]).any():
        raise ValueError("Prepared data has duplicate country/month rows.")
    forbidden = [
        column
        for column in data.columns
        if any(token in column.lower() for token in ("usgs", "world_bank", "aluminium", "nickel"))
    ]
    if forbidden:
        raise ValueError(f"Forbidden external-source columns remain: {forbidden}")
    thresholds = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    return data.sort_values(["date", "country_key"]).reset_index(drop=True), thresholds


def fit_for_origin(
    data: pd.DataFrame,
    horizon: int,
    candidate: Candidate,
    origin_rows: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame]:
    target = f"target_class_h{horizon}"
    target_available = f"target_data_available_date_assumed_h{horizon}"
    as_of = origin_rows["origin_data_available_date_assumed"].max()
    train = data.loc[
        data[f"model_b_training_eligible_h{horizon}"].astype(bool)
        & (data[target_available] <= as_of)
        & (data["date"] < origin_rows["date"].min())
    ].copy()
    if len(train) < 150 or train[target].nunique() < 4:
        raise ValueError(
            f"Insufficient rolling training data for H{horizon} at {origin_rows['date'].min():%Y-%m}."
        )
    model = make_pipeline(candidate)
    model.fit(train[FEATURES], train[target])
    return model, train


def validation_predictions(data: pd.DataFrame, horizon: int, candidate: Candidate) -> pd.DataFrame:
    target = f"target_class_h{horizon}"
    target_available = f"target_data_available_date_assumed_h{horizon}"
    pieces = []
    for fold, (start_raw, end_raw) in enumerate(VALIDATION_BLOCKS, start=1):
        start, end = pd.Timestamp(start_raw), pd.Timestamp(end_raw)
        validation = data.loc[
            data["date"].between(start, end)
            & data[f"model_b_training_eligible_h{horizon}"].astype(bool)
            & (data[f"target_month_h{horizon}"] <= pd.Timestamp("2021-06-01"))
        ].copy()
        if validation.empty:
            continue
        first_as_of = validation["origin_data_available_date_assumed"].min()
        train = data.loc[
            data[f"model_b_training_eligible_h{horizon}"].astype(bool)
            & (data[target_available] <= first_as_of)
            & (data["date"] < start)
        ].copy()
        if len(train) < 150 or train[target].nunique() < 4:
            continue
        model = make_pipeline(candidate)
        model.fit(train[FEATURES], train[target])
        probabilities = aligned_probabilities(model, validation[FEATURES])
        historical = historical_frequency_probabilities(train, validation, target)
        persistence = persistence_probabilities(validation, historical)
        result = validation[
            ["date", "country_key", "country", target, "last_observed_3m_class"]
        ].copy()
        result["fold"] = fold
        for index, label in enumerate(CLASSES):
            result[f"p_{label}"] = probabilities[:, index]
            result[f"historical_p_{label}"] = historical[:, index]
            result[f"persistence_p_{label}"] = persistence[:, index]
        pieces.append(result)
    if not pieces:
        raise ValueError(f"No validation predictions for H{horizon}, {candidate.name}.")
    return pd.concat(pieces, ignore_index=True)


def select_candidates(data: pd.DataFrame) -> tuple[dict[int, dict], pd.DataFrame, pd.DataFrame]:
    selection = {}
    comparison_rows = []
    selected_validation = []
    temperature_grid = [0.60, 0.75, 0.90, 1.00, 1.15, 1.35, 1.60, 2.00, 2.50]
    for horizon in HORIZONS:
        target = f"target_class_h{horizon}"
        horizon_results = []
        for candidate in CANDIDATES:
            predictions = validation_predictions(data, horizon, candidate)
            y_true = class_indices(predictions[target])
            raw = predictions[[f"p_{label}" for label in CLASSES]].to_numpy()
            temperature_scores = []
            for temperature in temperature_grid:
                calibrated = apply_temperature(raw, temperature)
                metrics = score_probabilities(y_true, calibrated)
                temperature_scores.append((metrics["brier_score"], temperature, metrics, calibrated))
            _, temperature, metrics, calibrated = min(temperature_scores, key=lambda item: item[0])
            historical = predictions[[f"historical_p_{label}" for label in CLASSES]].to_numpy()
            persistence = predictions[[f"persistence_p_{label}" for label in CLASSES]].to_numpy()
            hist_score = multiclass_brier(y_true, historical)
            persistence_score = multiclass_brier(y_true, persistence)
            row = {
                "horizon_months": horizon,
                "candidate": candidate.name,
                "family": candidate.family,
                "temperature": temperature,
                **metrics,
                "historical_frequency_brier": hist_score,
                "persistence_brier": persistence_score,
                "beats_historical_frequency": metrics["brier_score"] < hist_score,
                "beats_persistence": metrics["brier_score"] < persistence_score,
                "calibration_gap_within_0_10": metrics["max_calibration_gap"] <= 0.10,
            }
            comparison_rows.append(row)
            horizon_results.append((row, candidate, predictions, calibrated))

        eligible = [
            item
            for item in horizon_results
            if item[0]["beats_historical_frequency"]
            and item[0]["beats_persistence"]
            and item[0]["calibration_gap_within_0_10"]
        ]
        pool = eligible if eligible else horizon_results
        best = min(pool, key=lambda item: item[0]["brier_score"])
        row, candidate, predictions, calibrated = best
        selection[horizon] = {
            "candidate": candidate,
            "temperature": float(row["temperature"]),
            "validation_metrics": row,
            "met_all_validation_gates": bool(eligible),
        }
        predictions = predictions.copy()
        predictions["selected_candidate"] = candidate.name
        for index, label in enumerate(CLASSES):
            predictions[f"calibrated_p_{label}"] = calibrated[:, index]
        selected_validation.append(predictions.assign(horizon_months=horizon))
    return selection, pd.DataFrame(comparison_rows), pd.concat(selected_validation, ignore_index=True)


def rolling_predictions(
    data: pd.DataFrame,
    horizon: int,
    selected: dict,
    start: str,
    end: str,
) -> pd.DataFrame:
    target = f"target_class_h{horizon}"
    candidate = selected["candidate"]
    temperature = selected["temperature"]
    output = []
    origin_dates = sorted(
        data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end)), "date"].unique()
    )
    for origin_date in origin_dates:
        origin = data.loc[data["date"] == origin_date].copy()
        completed = origin[target].notna()
        if not completed.any():
            continue
        origin = origin.loc[completed].copy()
        model, train = fit_for_origin(data, horizon, candidate, origin)
        model_probability = apply_temperature(
            aligned_probabilities(model, origin[FEATURES]), temperature
        )
        historical = historical_frequency_probabilities(train, origin, target)
        persistence = persistence_probabilities(origin, historical)
        result = origin[
            [
                "date",
                "country_key",
                "country",
                "country_share",
                f"target_month_h{horizon}",
                f"target_country_share_h{horizon}",
                f"target_share_change_h{horizon}",
                target,
                "last_observed_3m_class",
            ]
        ].copy()
        result["training_rows"] = len(train)
        result["selected_model"] = candidate.name
        result["temperature"] = temperature
        for index, label in enumerate(CLASSES):
            result[f"p_{label}"] = model_probability[:, index]
            result[f"historical_p_{label}"] = historical[:, index]
            result[f"persistence_p_{label}"] = persistence[:, index]
        output.append(result)
    if not output:
        return pd.DataFrame()
    return pd.concat(output, ignore_index=True).assign(horizon_months=horizon)


def evaluation_metrics(predictions: pd.DataFrame, period: str) -> tuple[dict, pd.DataFrame]:
    horizon = int(predictions["horizon_months"].iloc[0])
    target = f"target_class_h{horizon}"
    y_true = class_indices(predictions[target])
    model_p = predictions[[f"p_{label}" for label in CLASSES]].to_numpy()
    historical_p = predictions[[f"historical_p_{label}" for label in CLASSES]].to_numpy()
    persistence_p = predictions[[f"persistence_p_{label}" for label in CLASSES]].to_numpy()
    metrics = score_probabilities(y_true, model_p)
    historical_score = multiclass_brier(y_true, historical_p)
    persistence_score = multiclass_brier(y_true, persistence_p)
    metrics.update(
        {
            "period": period,
            "horizon_months": horizon,
            "forecast_rows": len(predictions),
            "origin_months": predictions["date"].nunique(),
            "selected_model": predictions["selected_model"].iloc[0],
            "historical_frequency_brier": historical_score,
            "persistence_brier": persistence_score,
            "beats_historical_frequency": metrics["brier_score"] < historical_score,
            "beats_persistence": metrics["brier_score"] < persistence_score,
            "calibration_gap_within_0_10": metrics["max_calibration_gap"] <= 0.10,
        }
    )
    metrics["passes_acceptance"] = bool(
        metrics["beats_historical_frequency"]
        and metrics["beats_persistence"]
        and metrics["calibration_gap_within_0_10"]
    )
    reliability = reliability_table(y_true, model_p)
    reliability["period"] = period
    reliability["horizon_months"] = horizon
    return metrics, reliability


def fit_final_models(data: pd.DataFrame, selection: dict[int, dict]) -> tuple[dict, pd.DataFrame]:
    latest_date = data["date"].max()
    latest_rows = data.loc[data["date"] == latest_date].copy()
    bundle = {
        "model_name": "Titanium Model B country supply shift",
        "model_version": "model_b_v1_permitted_sources_2010_2026",
        "status": "research prototype",
        "random_seed": RANDOM_SEED,
        "classes": CLASSES,
        "features": FEATURES,
        "source_policy": ["UN Comtrade", "Bank of England FX", "ONS ERUC/PPI"],
        "excluded_sources": ["USGS", "World Bank cross-metal prices"],
        "latest_origin": latest_date.strftime("%Y-%m"),
        "horizons": {},
    }
    forecast_rows = []
    for horizon in HORIZONS:
        target = f"target_class_h{horizon}"
        target_available = f"target_data_available_date_assumed_h{horizon}"
        selected = selection[horizon]
        candidate = selected["candidate"]
        as_of = latest_rows["origin_data_available_date_assumed"].max()
        train = data.loc[
            data[f"model_b_training_eligible_h{horizon}"].astype(bool)
            & (data[target_available] <= as_of)
            & (data["date"] < latest_date)
        ].copy()
        model = make_pipeline(candidate)
        model.fit(train[FEATURES], train[target])
        model_probability = apply_temperature(
            aligned_probabilities(model, latest_rows[FEATURES]), selected["temperature"]
        )
        target_month = latest_date + pd.DateOffset(months=horizon)
        for row_index, (_, source_row) in enumerate(latest_rows.iterrows()):
            probabilities = model_probability[row_index]
            forecast_rows.append(
                {
                    "forecast_origin": latest_date.strftime("%Y-%m"),
                    "target_month": target_month.strftime("%Y-%m"),
                    "horizon_months": horizon,
                    "country_key": source_row["country_key"],
                    "country": source_row["country"],
                    "current_supply_share": source_row["country_share"],
                    "selected_model": candidate.name,
                    "predicted_class": CLASSES[int(probabilities.argmax())],
                    **{f"probability_{label}": probabilities[index] for index, label in enumerate(CLASSES)},
                }
            )
        bundle["horizons"][f"h{horizon}"] = {
            "candidate": candidate.name,
            "family": candidate.family,
            "params": candidate.params,
            "temperature": selected["temperature"],
            "training_rows": len(train),
            "reporting_only": horizon == 12,
            "estimator": model,
        }
    return bundle, pd.DataFrame(forecast_rows)


def feature_importance_rows(bundle: dict) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        entry = bundle["horizons"][f"h{horizon}"]
        model = entry["estimator"]
        names = model.named_steps["preprocessor"].get_feature_names_out()
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importance = np.mean(np.abs(estimator.coef_), axis=0)
        else:
            continue
        total = importance.sum()
        if total > 0:
            importance = importance / total
        for name, value in zip(names, importance):
            clean_name = name.split("__", 1)[-1]
            rows.append(
                {
                    "horizon_months": horizon,
                    "selected_model": entry["candidate"],
                    "feature": clean_name,
                    "relative_importance": float(value),
                }
            )
    return pd.DataFrame(rows)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def make_visualizations(
    data: pd.DataFrame,
    thresholds: dict,
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    stress_predictions: pd.DataFrame,
    forecasts: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    plt.figure(figsize=(10, 4.4))
    sample = data.loc[
        data["target_month_h3"].notna()
        & (data["target_month_h3"] <= pd.Timestamp("2021-06-01")),
        "target_share_change_h3",
    ].dropna()
    plt.hist(sample * 100, bins=35, color="#0B7D80", alpha=0.78)
    for value, label, colour in [
        (thresholds["collapse_upper"], "5th: collapse", "#A83F36"),
        (thresholds["fall_upper"], "33rd: fall", "#D25D2B"),
        (thresholds["stable_upper"], "67th: rise", "#2E805C"),
    ]:
        plt.axvline(value * 100, color=colour, linewidth=2, label=f"{label} ({value*100:.2f} pp)")
    plt.title("Fixed Model B class boundaries from pre-test H3 supply-share changes")
    plt.xlabel("Three-month country supply-share change (percentage points)")
    plt.ylabel("Country-month observations")
    plt.legend(frameon=False, ncol=3, fontsize=8)
    save_plot(VIS_DIR / "class_threshold_distribution.png")

    stress = metrics[metrics["period"] == "disruption_2021_2023"].copy()
    x = np.arange(len(stress))
    width = 0.24
    plt.figure(figsize=(10, 4.6))
    plt.bar(x - width, stress["brier_score"], width, label="Selected model", color="#0B7D80")
    plt.bar(x, stress["persistence_brier"], width, label="Last-change baseline", color="#D25D2B")
    plt.bar(x + width, stress["historical_frequency_brier"], width, label="Historical-frequency baseline", color="#8B9994")
    plt.xticks(x, [f"H{h}" for h in stress["horizon_months"]])
    plt.ylabel("Multiclass Brier score (lower is better)")
    plt.title("Disruption-test probability accuracy")
    plt.legend(frameon=False)
    save_plot(VIS_DIR / "disruption_brier_comparison.png")

    plt.figure(figsize=(9.5, 5.2))
    for horizon, group in reliability[
        (reliability["period"] == "disruption_2021_2023")
        & (reliability["count"] > 0)
    ].groupby("horizon_months"):
        plt.plot(
            group["mean_predicted_probability"],
            group["observed_frequency"],
            marker="o",
            label=f"H{horizon}",
        )
    plt.plot([0, 1], [0, 1], "--", color="#60777B", label="Perfect reliability")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Disruption-test probability reliability")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    save_plot(VIS_DIR / "disruption_probability_reliability.png")

    russia = stress_predictions[
        (stress_predictions["country_key"] == "russia")
        & (stress_predictions["horizon_months"].isin(PRIMARY_HORIZONS))
    ].copy()
    plt.figure(figsize=(11, 5))
    for horizon, group in russia.groupby("horizon_months"):
        plt.plot(group["date"], group["p_collapse"], marker="o", label=f"H{horizon} collapse probability")
        actual = group[f"target_class_h{horizon}"] == "collapse"
        plt.scatter(group.loc[actual, "date"], np.ones(actual.sum()), marker="x", s=65, label=f"H{horizon} actual collapse")
    plt.ylim(-0.03, 1.06)
    plt.ylabel("Probability")
    plt.xlabel("Forecast origin")
    plt.title("Russian supply-collapse signal during the disruption test")
    plt.legend(frameon=False, ncol=2)
    plt.grid(alpha=0.22)
    save_plot(VIS_DIR / "russia_collapse_probability.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        frame = forecasts[forecasts["horizon_months"] == horizon].set_index("country")
        matrix = frame[[f"probability_{label}" for label in CLASSES]].to_numpy()
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        axis.set_title(f"H{horizon} | {frame['target_month'].iloc[0]}")
        axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
        axis.set_yticks(range(len(frame)), frame.index if axis is axes[0] else [])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row,column]:.0%}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=axes, shrink=.75, label="Probability")
    fig.suptitle("Latest country supply-shift probabilities", fontsize=14)
    plt.tight_layout(rect=(0, 0, .95, .94))
    plt.savefig(VIS_DIR / "latest_probability_heatmap.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for axis, horizon in zip(axes, HORIZONS):
        top = (
            feature_importance[feature_importance["horizon_months"] == horizon]
            .nlargest(10, "relative_importance")
            .sort_values("relative_importance")
        )
        axis.barh(top["feature"], top["relative_importance"] * 100, color="#0B7D80")
        axis.set_title(f"H{horizon}")
        axis.set_xlabel("Relative influence (%)")
        axis.tick_params(axis="y", labelsize=7)
    fig.suptitle("Selected-model feature influence", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, .94))
    plt.savefig(VIS_DIR / "selected_model_feature_importance.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def html_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, callable] | None = None) -> str:
    formats = formats or {}
    header = "".join(f"<th>{column.replace('_', ' ').title()}</th>" for column in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column in formats:
                value = formats[column](value)
            cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_dashboard(metrics: pd.DataFrame, forecasts: pd.DataFrame, thresholds: dict) -> None:
    primary_stress = metrics[
        (metrics["period"] == "disruption_2021_2023")
        & metrics["horizon_months"].isin(PRIMARY_HORIZONS)
    ]
    pass_count = int(primary_stress["passes_acceptance"].sum())
    forecast_display = forecasts.copy()
    for label in CLASSES:
        forecast_display[f"probability_{label}"] = forecast_display[f"probability_{label}"].map(lambda value: f"{value:.1%}")
    metrics_display = metrics.copy()
    metrics_display["status"] = np.where(metrics_display["passes_acceptance"], "PASS", "REVIEW")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Titanium Model B - Country Supply Shift</title>
<style>
:root{{--ink:#14343b;--teal:#087f83;--orange:#d25d2b;--paper:#fbfaf4;--line:#d8d4c8;--muted:#60777b}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 90% 0,#dcefed,transparent 35rem),#edf0eb;color:var(--ink);font-family:Aptos,Segoe UI,sans-serif}}
.shell{{max-width:1420px;margin:auto;padding:28px}} .hero{{padding:44px;background:linear-gradient(120deg,#12333b,#087f83);color:white;box-shadow:0 18px 45px #14343b22}}
.eyebrow{{color:#f2a17d;font-size:12px;letter-spacing:.18em;font-weight:800}} h1,h2{{font-family:Georgia,serif;font-weight:500}} h1{{font-size:54px;line-height:1;margin:12px 0}} h2{{font-size:32px;margin:4px 0 18px}}
.hero p{{max-width:880px;color:#d6e7e5;line-height:1.65}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}
.card,.panel{{background:var(--paper);border:1px solid var(--line);box-shadow:0 12px 30px #14343b12}} .card{{padding:20px}} .card span{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}} .card strong{{display:block;font:500 30px Georgia;margin-top:8px}}
.panel{{padding:28px;margin:22px 0}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}} img{{width:100%;height:auto;border:1px solid var(--line)}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th{{background:var(--ink);color:white;text-align:left;padding:10px}} td{{padding:9px;border-bottom:1px solid var(--line)}} tr:nth-child(even){{background:#f4f1e8}}
.note{{padding:15px;border-left:4px solid var(--orange);background:#f7e5da;line-height:1.55}} footer{{text-align:center;color:var(--muted);font-size:12px;padding:20px}}
@media(max-width:900px){{.cards,.grid{{grid-template-columns:1fr 1fr}}}} @media(max-width:620px){{.shell{{padding:12px}}.hero{{padding:28px 20px}}h1{{font-size:38px}}.cards,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="shell">
<section class="hero"><div class="eyebrow">MODEL B | RESEARCH PROTOTYPE</div><h1>Titanium supply shift by country</h1><p>Probability that each main source country's UK titanium supply share will collapse, fall, remain stable or rise at 3, 6 and 12 months. H3 and H6 are acceptance horizons; H12 is reporting-only.</p></section>
<section class="cards"><div class="card"><span>Training coverage</span><strong>2010-2026</strong></div><div class="card"><span>Supplier groups</span><strong>6</strong></div><div class="card"><span>Primary stress passes</span><strong>{pass_count}/2</strong></div><div class="card"><span>Forecast origin</span><strong>{forecasts['forecast_origin'].iloc[0]}</strong></div></section>
<section class="panel"><h2>Latest supply-shift probabilities</h2>{html_table(forecast_display,['country','horizon_months','target_month','predicted_class','probability_collapse','probability_fall','probability_stable','probability_rise'])}</section>
<section class="grid"><section class="panel"><h2>Disruption-test accuracy</h2><img src="{image_data_uri(VIS_DIR/'disruption_brier_comparison.png')}" alt="Brier score comparison"></section><section class="panel"><h2>Probability reliability</h2><img src="{image_data_uri(VIS_DIR/'disruption_probability_reliability.png')}" alt="Reliability plot"></section></section>
<section class="panel"><h2>Russian supply-collapse signal</h2><img src="{image_data_uri(VIS_DIR/'russia_collapse_probability.png')}" alt="Russia collapse probability"></section>
<section class="panel"><h2>Evaluation summary</h2>{html_table(metrics_display,['period','horizon_months','selected_model','brier_score','persistence_brier','historical_frequency_brier','max_calibration_gap','status'],{'brier_score':lambda x:f'{x:.4f}','persistence_brier':lambda x:f'{x:.4f}','historical_frequency_brier':lambda x:f'{x:.4f}','max_calibration_gap':lambda x:f'{x:.3f}'})}</section>
<section class="grid"><section class="panel"><h2>Latest probability map</h2><img src="{image_data_uri(VIS_DIR/'latest_probability_heatmap.png')}" alt="Latest probabilities"></section><section class="panel"><h2>Feature influence</h2><img src="{image_data_uri(VIS_DIR/'selected_model_feature_importance.png')}" alt="Feature influence"></section></section>
<section class="panel"><h2>Class definition</h2><p>Collapse: change below {thresholds['collapse_upper']*100:.2f} percentage points. Fall: up to {thresholds['fall_upper']*100:.2f}. Stable: up to {thresholds['stable_upper']*100:.2f}. Above that is rise.</p><div class="note">Only UN Comtrade, Bank of England FX and ONS ERUC/PPI data are used. The requested 2005 start was unavailable, so fixed class limits use 2010 through June 2021. USGS and World Bank cross-metal inputs are excluded.</div></section>
<footer>Fixed random seed {RANDOM_SEED} | Model B research package</footer></main></body></html>"""
    (VIS_DIR / "titanium_model_b_dashboard.html").write_text(html, encoding="utf-8")


def serializable_bundle(bundle: dict) -> dict:
    output = {key: value for key, value in bundle.items() if key != "horizons"}
    output["horizons"] = {}
    for key, entry in bundle["horizons"].items():
        output["horizons"][key] = {k: v for k, v in entry.items() if k != "estimator"}
    return output


def main() -> None:
    for directory in (MODELS_DIR, RESULTS_DIR, VIS_DIR, DOCS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    data, thresholds = load_data()

    selection, validation_comparison, validation_predictions_frame = select_candidates(data)
    validation_comparison.to_csv(RESULTS_DIR / "validation_model_comparison.csv", index=False)
    validation_predictions_frame.to_csv(RESULTS_DIR / "selected_validation_predictions.csv", index=False)

    stress_frames = []
    post_frames = []
    metric_rows = []
    reliability_frames = []
    for horizon in HORIZONS:
        stress = rolling_predictions(
            data, horizon, selection[horizon], "2021-07-01", "2023-06-01"
        )
        post = rolling_predictions(
            data, horizon, selection[horizon], "2023-07-01", "2026-06-01"
        )
        stress_frames.append(stress)
        post_frames.append(post)
        if not stress.empty:
            metrics, reliability = evaluation_metrics(stress, "disruption_2021_2023")
            metric_rows.append(metrics)
            reliability_frames.append(reliability)
        if not post.empty:
            metrics, reliability = evaluation_metrics(post, "post_disruption")
            metric_rows.append(metrics)
            reliability_frames.append(reliability)

    stress_predictions = pd.concat(stress_frames, ignore_index=True)
    post_predictions = pd.concat(post_frames, ignore_index=True)
    evaluation = pd.DataFrame(metric_rows)
    reliability = pd.concat(reliability_frames, ignore_index=True)
    stress_predictions.to_csv(RESULTS_DIR / "disruption_2021_2023_predictions.csv", index=False)
    post_predictions.to_csv(RESULTS_DIR / "post_disruption_predictions.csv", index=False)
    evaluation.to_csv(RESULTS_DIR / "evaluation_metrics.csv", index=False)
    reliability.to_csv(RESULTS_DIR / "probability_reliability_bins.csv", index=False)

    bundle, forecasts = fit_final_models(data, selection)
    model_path = MODELS_DIR / "titanium_model_b_country_supply_shift_selected.joblib"
    joblib.dump(bundle, model_path)
    forecasts.to_csv(RESULTS_DIR / "latest_country_supply_shift_forecasts.csv", index=False)
    response = {
        "model": serializable_bundle(bundle),
        "forecasts": forecasts.to_dict(orient="records"),
    }
    (RESULTS_DIR / "model_b_response.json").write_text(
        json.dumps(response, indent=2, default=str), encoding="utf-8"
    )
    (RESULTS_DIR / "model_manifest.json").write_text(
        json.dumps(serializable_bundle(bundle), indent=2, default=str), encoding="utf-8"
    )

    importance = feature_importance_rows(bundle)
    importance.to_csv(RESULTS_DIR / "selected_model_feature_importance.csv", index=False)
    make_visualizations(
        data,
        thresholds,
        evaluation,
        reliability,
        stress_predictions,
        forecasts,
        importance,
    )
    build_dashboard(evaluation, forecasts, thresholds)

    selected_summary = {
        f"h{horizon}": {
            "candidate": selection[horizon]["candidate"].name,
            "temperature": selection[horizon]["temperature"],
            "validation_metrics": selection[horizon]["validation_metrics"],
            "met_all_validation_gates": selection[horizon]["met_all_validation_gates"],
        }
        for horizon in HORIZONS
    }
    (RESULTS_DIR / "selected_models.json").write_text(
        json.dumps(selected_summary, indent=2, default=str), encoding="utf-8"
    )

    print("Titanium Model B training complete")
    for horizon in HORIZONS:
        print(
            f"H{horizon}: {selection[horizon]['candidate'].name}, "
            f"temperature={selection[horizon]['temperature']:.2f}"
        )
    print("Evaluation:")
    print(
        evaluation[
            [
                "period",
                "horizon_months",
                "brier_score",
                "persistence_brier",
                "historical_frequency_brier",
                "max_calibration_gap",
                "passes_acceptance",
            ]
        ].to_string(index=False)
    )
    print(f"Model: {model_path}")
    print(f"Dashboard: {VIS_DIR / 'titanium_model_b_dashboard.html'}")


if __name__ == "__main__":
    main()
