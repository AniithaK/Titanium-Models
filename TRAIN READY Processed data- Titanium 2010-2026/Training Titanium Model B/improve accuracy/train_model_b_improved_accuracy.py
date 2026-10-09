from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve()
RUN_DIR = HERE.parent
MODEL_B_ROOT = RUN_DIR.parent
DATA_PATH = MODEL_B_ROOT / "data" / "titanium_model_b_country_month_training_2010_2026.csv"
THRESHOLD_PATH = MODEL_B_ROOT / "data" / "model_b_fixed_class_thresholds.json"
BASELINE_METRICS_PATH = MODEL_B_ROOT / "results" / "evaluation_metrics.csv"

RESULTS_DIR = RUN_DIR / "results"
MODELS_DIR = RUN_DIR / "models"
VIS_DIR = RUN_DIR / "visualizations"
DOCS_DIR = RUN_DIR / "documentation"

RANDOM_SEED = 42
CLASSES = ["collapse", "fall", "stable", "rise"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
HORIZONS = (3, 6, 12)
VALIDATION_BLOCKS = [
    ("2018-01-01", "2018-12-01"),
    ("2019-01-01", "2019-12-01"),
    ("2020-01-01", "2021-06-01"),
]

BASE_NUMERIC_FEATURES = [
    "country_share",
    "log1p_country_quantity_kg",
    "country_share_lag_1m",
    "country_share_lag_3m",
    "country_share_lag_6m",
    "country_share_lag_12m",
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

ENGINEERED_NUMERIC_FEATURES = [
    "year_trend",
    "country_share_rank",
    "country_share_relative_to_top",
    "country_share_gap_to_top",
    "is_top_supplier",
    "share_velocity_1m",
    "share_velocity_3m",
    "share_velocity_6m",
    "share_acceleration_1m_vs_3m",
    "share_acceleration_3m_vs_6m",
    "share_acceleration_6m_vs_12m",
    "quantity_velocity_1m",
    "quantity_velocity_3m",
    "quantity_acceleration_1m_vs_3m",
]
for window in (3, 6, 12):
    ENGINEERED_NUMERIC_FEATURES.extend(
        [
            f"country_share_rolling_mean_{window}m",
            f"country_share_rolling_std_{window}m",
            f"country_share_vs_rolling_mean_{window}m",
            f"country_share_rolling_range_{window}m",
        ]
    )

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = ["country_key", "last_observed_3m_class"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    task: str
    params: dict


CANDIDATES = [
    Candidate("logistic_accuracy", "logistic", "classification", {"C": 0.3}),
    Candidate(
        "random_forest_accuracy",
        "random_forest_classifier",
        "classification",
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2, "max_features": 0.75},
    ),
    Candidate(
        "extra_trees_accuracy",
        "extra_trees_classifier",
        "classification",
        {"n_estimators": 450, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.85},
    ),
    Candidate(
        "hist_gradient_boosting_classifier",
        "hist_gradient_boosting_classifier",
        "classification",
        {"max_iter": 250, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 12, "l2_regularization": 0.5},
    ),
    Candidate(
        "gradient_boosting_classifier",
        "gradient_boosting_classifier",
        "classification",
        {"n_estimators": 250, "learning_rate": 0.035, "max_depth": 2, "min_samples_leaf": 7, "subsample": 0.85},
    ),
    Candidate(
        "extra_trees_share_change",
        "extra_trees_regressor",
        "regression",
        {"n_estimators": 450, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.85},
    ),
    Candidate(
        "random_forest_share_change",
        "random_forest_regressor",
        "regression",
        {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 2, "max_features": 0.75},
    ),
    Candidate(
        "hist_gradient_boosting_share_change",
        "hist_gradient_boosting_regressor",
        "regression",
        {"max_iter": 280, "learning_rate": 0.045, "max_leaf_nodes": 15, "min_samples_leaf": 12, "l2_regularization": 0.5},
    ),
    Candidate(
        "gradient_boosting_share_change",
        "gradient_boosting_regressor",
        "regression",
        {"n_estimators": 250, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 7, "subsample": 0.85, "loss": "huber"},
    ),
]


def ensure_dirs() -> None:
    for directory in (RESULTS_DIR, MODELS_DIR, VIS_DIR, DOCS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().sort_values(["country_key", "date"]).reset_index(drop=True)
    frame["year_trend"] = (frame["date"].dt.year - 2010) * 12 + frame["date"].dt.month - 1
    frame["country_share_rank"] = frame.groupby("date")["country_share"].rank(method="average", ascending=False)
    frame["country_share_relative_to_top"] = np.divide(
        frame["country_share"],
        frame["top_supplier_share_weight"],
        out=np.zeros(len(frame), dtype=float),
        where=frame["top_supplier_share_weight"].to_numpy() > 0,
    )
    frame["country_share_gap_to_top"] = frame["top_supplier_share_weight"] - frame["country_share"]
    frame["is_top_supplier"] = (frame["country_share_rank"] == 1).astype(int)
    frame["share_velocity_1m"] = frame["country_share_change_1m"]
    frame["share_velocity_3m"] = frame["country_share_change_3m"] / 3.0
    frame["share_velocity_6m"] = frame["country_share_change_6m"] / 6.0
    frame["share_acceleration_1m_vs_3m"] = frame["share_velocity_1m"] - frame["share_velocity_3m"]
    frame["share_acceleration_3m_vs_6m"] = frame["share_velocity_3m"] - frame["share_velocity_6m"]
    frame["share_acceleration_6m_vs_12m"] = frame["share_velocity_6m"] - frame["country_share_change_12m"] / 12.0
    frame["quantity_velocity_1m"] = frame["country_quantity_log1p_change_1m"]
    frame["quantity_velocity_3m"] = frame["country_quantity_log1p_change_3m"] / 3.0
    frame["quantity_acceleration_1m_vs_3m"] = frame["quantity_velocity_1m"] - frame["quantity_velocity_3m"]

    pieces = []
    for _, group in frame.groupby("country_key", sort=False):
        group = group.sort_values("date").copy()
        for window in (3, 6, 12):
            rolling = group["country_share"].rolling(window, min_periods=2)
            group[f"country_share_rolling_mean_{window}m"] = rolling.mean()
            group[f"country_share_rolling_std_{window}m"] = rolling.std(ddof=0)
            group[f"country_share_vs_rolling_mean_{window}m"] = (
                group["country_share"] - group[f"country_share_rolling_mean_{window}m"]
            )
            group[f"country_share_rolling_range_{window}m"] = rolling.max() - rolling.min()
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "country_key"]).reset_index(drop=True)


def load_data() -> tuple[pd.DataFrame, dict[str, float]]:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    date_columns = [
        "date",
        "origin_data_available_date_assumed",
        *[f"target_month_h{h}" for h in HORIZONS],
        *[f"target_data_available_date_assumed_h{h}" for h in HORIZONS],
    ]
    for column in date_columns:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data = add_engineered_features(data)
    missing = sorted(set(FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing improved features: {missing}")
    if data.duplicated(["date", "country_key"]).any():
        raise ValueError("Duplicate country-month rows found.")
    thresholds = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    return data, thresholds


def class_from_change(values: np.ndarray | pd.Series, thresholds: dict[str, float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, "stable", dtype=object)
    result[array < thresholds["collapse_upper"]] = "collapse"
    result[(array >= thresholds["collapse_upper"]) & (array < thresholds["fall_upper"])] = "fall"
    result[(array >= thresholds["fall_upper"]) & (array <= thresholds["stable_upper"])] = "stable"
    result[array > thresholds["stable_upper"]] = "rise"
    return result.astype(str)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_pipeline(candidate: Candidate) -> Pipeline:
    if candidate.family == "logistic":
        estimator = LogisticRegression(
            C=candidate.params["C"], max_iter=5000, solver="lbfgs", random_state=RANDOM_SEED
        )
        scale = True
    elif candidate.family == "random_forest_classifier":
        estimator = RandomForestClassifier(
            **candidate.params, class_weight=None, n_jobs=-1, random_state=RANDOM_SEED
        )
        scale = False
    elif candidate.family == "extra_trees_classifier":
        estimator = ExtraTreesClassifier(
            **candidate.params, class_weight=None, n_jobs=-1, random_state=RANDOM_SEED
        )
        scale = False
    elif candidate.family == "hist_gradient_boosting_classifier":
        estimator = HistGradientBoostingClassifier(**candidate.params, random_state=RANDOM_SEED)
        scale = False
    elif candidate.family == "gradient_boosting_classifier":
        estimator = GradientBoostingClassifier(**candidate.params, random_state=RANDOM_SEED)
        scale = False
    elif candidate.family == "extra_trees_regressor":
        estimator = ExtraTreesRegressor(**candidate.params, n_jobs=-1, random_state=RANDOM_SEED)
        scale = False
    elif candidate.family == "random_forest_regressor":
        estimator = RandomForestRegressor(**candidate.params, n_jobs=-1, random_state=RANDOM_SEED)
        scale = False
    elif candidate.family == "hist_gradient_boosting_regressor":
        estimator = HistGradientBoostingRegressor(**candidate.params, random_state=RANDOM_SEED)
        scale = False
    elif candidate.family == "gradient_boosting_regressor":
        estimator = GradientBoostingRegressor(**candidate.params, random_state=RANDOM_SEED)
        scale = False
    else:
        raise ValueError(candidate.family)
    return Pipeline([("preprocessor", make_preprocessor(scale)), ("model", estimator)])


def aligned_classifier_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    aligned = np.zeros((len(features), len(CLASSES)), dtype=float)
    for source_index, label in enumerate(model.named_steps["model"].classes_):
        aligned[:, CLASS_TO_INDEX[str(label)]] = raw[:, source_index]
    sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, sums, out=np.full_like(aligned, 0.25), where=sums > 0)


def empirical_regression_probabilities(
    point_predictions: np.ndarray,
    residuals: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) > 400:
        positions = np.linspace(0, len(residuals) - 1, 400).astype(int)
        residuals = np.sort(residuals)[positions]
    simulated = point_predictions[:, None] + residuals[None, :]
    probabilities = np.ones((len(point_predictions), len(CLASSES)), dtype=float)
    simulated_classes = class_from_change(simulated, thresholds)
    for index, label in enumerate(CLASSES):
        probabilities[:, index] += (simulated_classes == label).sum(axis=1)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    logits = np.log(clipped) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def model_outputs(
    model: Pipeline,
    candidate: Candidate,
    train: pd.DataFrame,
    prediction: pd.DataFrame,
    horizon: int,
    thresholds: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if candidate.task == "classification":
        probabilities = aligned_classifier_probabilities(model, prediction[FEATURES])
        labels = np.array(CLASSES, dtype=object)[probabilities.argmax(axis=1)].astype(str)
        return labels, probabilities, None

    target_change = f"target_share_change_h{horizon}"
    point_predictions = model.predict(prediction[FEATURES])
    training_predictions = model.predict(train[FEATURES])
    residuals = train[target_change].to_numpy(dtype=float) - training_predictions
    probabilities = empirical_regression_probabilities(point_predictions, residuals, thresholds)
    labels = class_from_change(point_predictions, thresholds)
    return labels, probabilities, point_predictions


def fit_candidate(
    candidate: Candidate,
    train: pd.DataFrame,
    horizon: int,
) -> Pipeline:
    target = f"target_class_h{horizon}" if candidate.task == "classification" else f"target_share_change_h{horizon}"
    model = make_pipeline(candidate)
    model.fit(train[FEATURES], train[target])
    return model


def training_rows(data: pd.DataFrame, horizon: int, start: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame:
    eligible = bool_series(data[f"model_b_training_eligible_h{horizon}"])
    target_available = f"target_data_available_date_assumed_h{horizon}"
    target_change = f"target_share_change_h{horizon}"
    return data.loc[
        eligible
        & data[target_change].notna()
        & (data[target_available] <= as_of)
        & (data["date"] < start)
    ].copy()


def validation_predictions(
    data: pd.DataFrame,
    candidate: Candidate,
    horizon: int,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    target_class = f"target_class_h{horizon}"
    rows = []
    for fold, (start_raw, end_raw) in enumerate(VALIDATION_BLOCKS, start=1):
        start, end = pd.Timestamp(start_raw), pd.Timestamp(end_raw)
        validation = data.loc[
            data["date"].between(start, end)
            & bool_series(data[f"model_b_training_eligible_h{horizon}"])
            & (data[f"target_month_h{horizon}"] <= pd.Timestamp("2021-06-01"))
        ].copy()
        if validation.empty:
            continue
        train = training_rows(
            data,
            horizon,
            start,
            validation["origin_data_available_date_assumed"].min(),
        )
        if len(train) < 150 or train[target_class].nunique() < 4:
            continue
        model = fit_candidate(candidate, train, horizon)
        labels, probabilities, point = model_outputs(
            model, candidate, train, validation, horizon, thresholds
        )
        result = validation[
            [
                "date",
                "country_key",
                "country",
                f"target_month_h{horizon}",
                f"target_share_change_h{horizon}",
                target_class,
            ]
        ].copy()
        result["fold"] = fold
        result["candidate"] = candidate.name
        result["predicted_class"] = labels
        result["predicted_share_change"] = point if point is not None else np.nan
        for index, label in enumerate(CLASSES):
            result[f"probability_{label}"] = probabilities[:, index]
        rows.append(result)
    if not rows:
        raise RuntimeError(f"No validation rows for H{horizon} and {candidate.name}")
    return pd.concat(rows, ignore_index=True)


def multiclass_brier(actual: np.ndarray, probabilities: np.ndarray) -> float:
    indices = np.array([CLASS_TO_INDEX[label] for label in actual])
    one_hot = np.eye(len(CLASSES))[indices]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def metrics_from_predictions(frame: pd.DataFrame, horizon: int) -> dict[str, float]:
    actual = frame[f"target_class_h{horizon}"].astype(str).to_numpy()
    predicted = frame["predicted_class"].astype(str).to_numpy()
    probabilities = frame[[f"probability_{label}" for label in CLASSES]].to_numpy(dtype=float)
    actual_indices = np.array([CLASS_TO_INDEX[label] for label in actual])
    predicted_indices = np.array([CLASS_TO_INDEX[label] for label in predicted])
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=CLASSES, average="weighted", zero_division=0)),
        "adjacent_class_accuracy": float(np.mean(np.abs(actual_indices - predicted_indices) <= 1)),
        "brier_score": multiclass_brier(actual, probabilities),
        "log_loss": float(log_loss(actual, probabilities, labels=CLASSES)),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def tune_temperature(frame: pd.DataFrame, horizon: int) -> tuple[float, pd.DataFrame, dict[str, float]]:
    actual = frame[f"target_class_h{horizon}"].astype(str).to_numpy()
    raw = frame[[f"probability_{label}" for label in CLASSES]].to_numpy(dtype=float)
    choices = []
    for temperature in (0.7, 0.85, 1.0, 1.15, 1.35, 1.6, 2.0):
        probabilities = apply_temperature(raw, temperature)
        candidate_frame = frame.copy()
        for index, label in enumerate(CLASSES):
            candidate_frame[f"probability_{label}"] = probabilities[:, index]
        choices.append((multiclass_brier(actual, probabilities), temperature, candidate_frame))
    _, temperature, calibrated = min(choices, key=lambda item: item[0])
    return float(temperature), calibrated, metrics_from_predictions(calibrated, horizon)


def select_models(
    data: pd.DataFrame,
    thresholds: dict[str, float],
) -> tuple[dict[int, dict], pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    all_validation = []
    selections: dict[int, dict] = {}
    for horizon in HORIZONS:
        horizon_frames: dict[str, pd.DataFrame] = {}
        candidate_lookup = {candidate.name: candidate for candidate in CANDIDATES}
        for candidate in CANDIDATES:
            predictions = validation_predictions(data, candidate, horizon, thresholds)
            temperature, calibrated, metrics = tune_temperature(predictions, horizon)
            horizon_frames[candidate.name] = calibrated
            comparison_rows.append(
                {
                    "horizon_months": horizon,
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "task": candidate.task,
                    "temperature": temperature,
                    **metrics,
                }
            )

        horizon_comparison = pd.DataFrame(
            [row for row in comparison_rows if row["horizon_months"] == horizon]
        ).sort_values(["accuracy", "balanced_accuracy", "macro_f1"], ascending=False)
        top_names = horizon_comparison.head(3)["candidate"].tolist()
        ensemble = horizon_frames[top_names[0]].copy()
        averaged = np.mean(
            [horizon_frames[name][[f"probability_{label}" for label in CLASSES]].to_numpy() for name in top_names],
            axis=0,
        )
        ensemble["candidate"] = "soft_vote_top3"
        ensemble["predicted_class"] = np.array(CLASSES, dtype=object)[averaged.argmax(axis=1)]
        for index, label in enumerate(CLASSES):
            ensemble[f"probability_{label}"] = averaged[:, index]
        temperature, ensemble, ensemble_metrics = tune_temperature(ensemble, horizon)
        horizon_frames["soft_vote_top3"] = ensemble
        comparison_rows.append(
            {
                "horizon_months": horizon,
                "candidate": "soft_vote_top3",
                "family": "ensemble",
                "task": "ensemble",
                "temperature": temperature,
                "members": ",".join(top_names),
                **ensemble_metrics,
            }
        )

        updated = pd.DataFrame([row for row in comparison_rows if row["horizon_months"] == horizon])
        best_row = updated.sort_values(
            ["accuracy", "balanced_accuracy", "macro_f1", "brier_score"],
            ascending=[False, False, False, True],
        ).iloc[0]
        name = str(best_row["candidate"])
        if name == "soft_vote_top3":
            members = [candidate_lookup[item] for item in str(best_row["members"]).split(",")]
            selections[horizon] = {
                "kind": "ensemble",
                "name": name,
                "members": members,
                "temperature": float(best_row["temperature"]),
                "validation_metrics": best_row.to_dict(),
            }
        else:
            selections[horizon] = {
                "kind": "single",
                "name": name,
                "candidate": candidate_lookup[name],
                "temperature": float(best_row["temperature"]),
                "validation_metrics": best_row.to_dict(),
            }
        selected_frame = horizon_frames[name].copy()
        selected_frame["horizon_months"] = horizon
        all_validation.append(selected_frame)
        print(
            f"Selected H{horizon}: {name} | validation accuracy={best_row['accuracy']:.3f}, "
            f"balanced={best_row['balanced_accuracy']:.3f}"
        )
    return selections, pd.DataFrame(comparison_rows), pd.concat(all_validation, ignore_index=True)


def fit_selection(
    selection: dict,
    train: pd.DataFrame,
    horizon: int,
) -> list[tuple[Candidate, Pipeline]]:
    candidates = selection["members"] if selection["kind"] == "ensemble" else [selection["candidate"]]
    return [(candidate, fit_candidate(candidate, train, horizon)) for candidate in candidates]


def selection_outputs(
    fitted: list[tuple[Candidate, Pipeline]],
    train: pd.DataFrame,
    prediction: pd.DataFrame,
    horizon: int,
    thresholds: dict[str, float],
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    probabilities = []
    points = []
    for candidate, model in fitted:
        _, candidate_probabilities, point = model_outputs(
            model, candidate, train, prediction, horizon, thresholds
        )
        probabilities.append(candidate_probabilities)
        if point is not None:
            points.append(point)
    averaged = apply_temperature(np.mean(probabilities, axis=0), temperature)
    labels = np.array(CLASSES, dtype=object)[averaged.argmax(axis=1)].astype(str)
    point = np.mean(points, axis=0) if points else None
    return labels, averaged, point


def rolling_evaluation(
    data: pd.DataFrame,
    thresholds: dict[str, float],
    selection: dict,
    horizon: int,
    start: str,
    end: str,
    period: str,
) -> pd.DataFrame:
    target_class = f"target_class_h{horizon}"
    output = []
    dates = sorted(data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end)), "date"].unique())
    for origin_date in dates:
        origin = data.loc[(data["date"] == origin_date) & data[target_class].notna()].copy()
        if origin.empty:
            continue
        train = training_rows(
            data,
            horizon,
            pd.Timestamp(origin_date),
            origin["origin_data_available_date_assumed"].max(),
        )
        if len(train) < 150 or train[target_class].nunique() < 4:
            continue
        fitted = fit_selection(selection, train, horizon)
        labels, probabilities, point = selection_outputs(
            fitted,
            train,
            origin,
            horizon,
            thresholds,
            selection["temperature"],
        )
        result = origin[
            [
                "date",
                "country_key",
                "country",
                "country_share",
                f"target_month_h{horizon}",
                f"target_country_share_h{horizon}",
                f"target_share_change_h{horizon}",
                target_class,
            ]
        ].copy()
        result["period"] = period
        result["horizon_months"] = horizon
        result["selected_model"] = selection["name"]
        result["training_rows"] = len(train)
        result["predicted_class"] = labels
        result["correct"] = result[target_class].astype(str).to_numpy() == labels
        result["predicted_share_change"] = point if point is not None else np.nan
        for index, label in enumerate(CLASSES):
            result[f"probability_{label}"] = probabilities[:, index]
        result["confidence"] = probabilities.max(axis=1)
        output.append(result)
    if not output:
        return pd.DataFrame()
    return pd.concat(output, ignore_index=True)


def fit_final_models(
    data: pd.DataFrame,
    thresholds: dict[str, float],
    selections: dict[int, dict],
) -> tuple[dict, pd.DataFrame]:
    latest_date = data["date"].max()
    latest = data.loc[data["date"] == latest_date].copy()
    as_of = latest["origin_data_available_date_assumed"].max()
    bundle = {
        "model_name": "Titanium Model B improved accuracy",
        "source_cutoff": latest_date.strftime("%Y-%m"),
        "classes": CLASSES,
        "features": FEATURES,
        "thresholds": thresholds,
        "horizons": {},
    }
    forecast_rows = []
    for horizon in HORIZONS:
        train = training_rows(data, horizon, latest_date, as_of)
        fitted = fit_selection(selections[horizon], train, horizon)
        labels, probabilities, point = selection_outputs(
            fitted,
            train,
            latest,
            horizon,
            thresholds,
            selections[horizon]["temperature"],
        )
        bundle["horizons"][f"h{horizon}"] = {
            "selection": selections[horizon],
            "fitted_models": fitted,
            "training_rows": len(train),
        }
        for row_index, (_, source_row) in enumerate(latest.iterrows()):
            record = {
                "forecast_origin": latest_date.strftime("%Y-%m"),
                "target_month": (latest_date + pd.DateOffset(months=horizon)).strftime("%Y-%m"),
                "horizon_months": horizon,
                "country_key": source_row["country_key"],
                "country": source_row["country"],
                "current_supply_share": source_row["country_share"],
                "selected_model": selections[horizon]["name"],
                "predicted_class": labels[row_index],
                "confidence": float(probabilities[row_index].max()),
            }
            if point is not None:
                record["predicted_share_change"] = float(point[row_index])
            for index, label in enumerate(CLASSES):
                record[f"probability_{label}"] = float(probabilities[row_index, index])
            forecast_rows.append(record)
    return bundle, pd.DataFrame(forecast_rows)


def final_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, horizon), group in predictions.groupby(["period", "horizon_months"]):
        metrics = metrics_from_predictions(group, int(horizon))
        rows.append(
            {
                "period": period,
                "horizon_months": int(horizon),
                "selected_model": group["selected_model"].iloc[0],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def make_feature_importance(bundle: dict, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    recent = data.loc[data["date"] >= pd.Timestamp("2024-01-01")].copy()
    for horizon in HORIZONS:
        fitted = bundle["horizons"][f"h{horizon}"]["fitted_models"]
        candidate, model = fitted[0]
        sample = recent.loc[recent[f"target_class_h{horizon}"].notna()].copy()
        if sample.empty:
            continue
        sample = sample.tail(min(len(sample), 180))
        if candidate.task == "classification":
            y = sample[f"target_class_h{horizon}"]
            scoring = "accuracy"
        else:
            y = sample[f"target_share_change_h{horizon}"]

            def threshold_accuracy(estimator, features, actual_changes):
                actual = class_from_change(np.asarray(actual_changes), bundle["thresholds"])
                predicted = class_from_change(estimator.predict(features), bundle["thresholds"])
                return accuracy_score(actual, predicted)

            scoring = threshold_accuracy
        importance = permutation_importance(
            model,
            sample[FEATURES],
            y,
            scoring=scoring,
            n_repeats=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        for feature, value, std in zip(FEATURES, importance.importances_mean, importance.importances_std):
            rows.append(
                {
                    "horizon_months": horizon,
                    "feature": feature,
                    "permutation_importance": float(value),
                    "importance_std": float(std),
                    "model": candidate.name,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon_months", "permutation_importance"], ascending=[True, False])


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()


def make_visualizations(
    validation_comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    baseline: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = {"Original Model B": "#A97018", "Improved experiment": "#087D80"}

    comparison = metrics.merge(
        baseline[["period", "horizon_months", "accuracy"]].rename(columns={"accuracy": "original_accuracy"}),
        on=["period", "horizon_months"],
        how="left",
    )
    chart = comparison.melt(
        id_vars=["period", "horizon_months"],
        value_vars=["original_accuracy", "accuracy"],
        var_name="version",
        value_name="accuracy_score",
    )
    chart["version"] = chart["version"].map(
        {"original_accuracy": "Original Model B", "accuracy": "Improved experiment"}
    )
    chart["label"] = chart["period"].str.replace("_", " ").str.title() + " H" + chart["horizon_months"].astype(str)
    plt.figure(figsize=(13, 5.8))
    sns.barplot(data=chart, x="label", y="accuracy_score", hue="version", palette=palette)
    plt.axhline(0.80, color="#A43B32", linestyle="--", label="80% target")
    plt.ylim(0, 1)
    plt.ylabel("Exact four-class accuracy")
    plt.xlabel("")
    plt.xticks(rotation=28, ha="right")
    plt.title("Original versus improved Model B accuracy")
    plt.legend(frameon=False, ncol=3)
    save_plot(VIS_DIR / "original_vs_improved_accuracy.png")

    validation_plot = validation_comparison.copy()
    validation_plot["label"] = "H" + validation_plot["horizon_months"].astype(str) + " " + validation_plot["candidate"]
    top = validation_plot.sort_values(["horizon_months", "accuracy"], ascending=[True, False]).groupby("horizon_months").head(6)
    plt.figure(figsize=(12.5, 7))
    sns.barplot(data=top, y="label", x="accuracy", hue="horizon_months", palette="crest", dodge=False)
    plt.axvline(0.80, color="#A43B32", linestyle="--")
    plt.xlim(0, 1)
    plt.xlabel("Validation exact accuracy")
    plt.ylabel("")
    plt.title("Best validation candidates by horizon")
    plt.legend(title="Horizon", frameon=False)
    save_plot(VIS_DIR / "validation_candidate_accuracy.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for row_index, period in enumerate(["disruption_2021_2023", "post_disruption"]):
        for col_index, horizon in enumerate(HORIZONS):
            ax = axes[row_index, col_index]
            group = predictions[(predictions["period"] == period) & (predictions["horizon_months"] == horizon)]
            actual = group[f"target_class_h{horizon}"].astype(str)
            cm = confusion_matrix(actual, group["predicted_class"], labels=CLASSES)
            sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", cbar=False, xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
            ax.set_title(f"{period.replace('_', ' ').title()} - H{horizon}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.tick_params(axis="x", rotation=35)
            ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Improved Model B confusion matrices", fontsize=17, y=1.01)
    save_plot(VIS_DIR / "improved_confusion_matrices.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, horizon in zip(axes, HORIZONS):
        group = importance[importance["horizon_months"] == horizon].head(12).sort_values("permutation_importance")
        ax.barh(group["feature"], group["permutation_importance"], color="#087D80")
        ax.set_title(f"H{horizon}")
        ax.set_xlabel("Accuracy decrease after permutation")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Improved Model B feature importance", fontsize=17, y=1.02)
    save_plot(VIS_DIR / "improved_feature_importance.png")


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_dashboard(metrics: pd.DataFrame, validation: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    best_post = metrics[metrics["period"] == "post_disruption"].sort_values("accuracy", ascending=False).iloc[0]
    target_met = bool((metrics["accuracy"] >= 0.80).all())
    rows = "".join(
        "<tr>"
        f"<td>{row.period.replace('_', ' ').title()}</td><td>H{int(row.horizon_months)}</td>"
        f"<td>{row.selected_model}</td><td>{row.accuracy:.1%}</td><td>{row.balanced_accuracy:.1%}</td>"
        f"<td>{row.macro_f1:.1%}</td><td>{row.adjacent_class_accuracy:.1%}</td><td>{row.brier_score:.3f}</td>"
        "</tr>"
        for row in metrics.itertuples(index=False)
    )
    forecast_rows = "".join(
        "<tr>"
        f"<td>{row.country}</td><td>H{int(row.horizon_months)}</td><td>{row.target_month}</td>"
        f"<td>{row.predicted_class}</td><td>{row.confidence:.1%}</td>"
        "</tr>"
        for row in forecasts.itertuples(index=False)
    )
    status = "80% target met" if target_met else "80% target not met without changing the task"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Improved Titanium Model B</title>
<style>
:root{{--ink:#12353b;--teal:#087d80;--orange:#cc5d2b;--paper:#fbf8f1;--line:#d8d0c2;--pale:#e6f1ef}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf2ef;color:var(--ink);font-family:Georgia,serif}}
header{{padding:48px 6vw;background:linear-gradient(125deg,#12353b,#087d80);color:white}}header h1{{font-size:42px;margin:8px 0}}
main{{max-width:1320px;margin:auto;padding:28px}}section{{background:var(--paper);padding:28px;margin-bottom:22px;border:1px solid var(--line)}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{padding:20px;background:white;border-top:4px solid var(--orange)}}
.card strong{{display:block;font-size:31px;color:var(--teal);margin-top:8px}}table{{border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px}}
th{{background:var(--ink);color:white;padding:10px;text-align:left}}td{{padding:9px;border-bottom:1px solid var(--line)}}tr:nth-child(even){{background:#f3ede2}}
img{{width:100%;height:auto}}.warning{{padding:16px;background:#f8e5d8;border-left:5px solid var(--orange);font-family:Arial,sans-serif}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr}}header h1{{font-size:30px}}main{{padding:12px}}section{{padding:16px;overflow:auto}}}}
</style></head><body><header><div>MODEL B ACCURACY IMPROVEMENT EXPERIMENT</div><h1>Titanium supplier-share movement</h1><p>Leakage-safe four-class evaluation using the original collapse, fall, stable and rise definitions.</p></header><main>
<section><div class="cards"><div class="card">Best post-disruption exact accuracy<strong>{best_post.accuracy:.1%}</strong></div><div class="card">Best horizon<strong>H{int(best_post.horizon_months)}</strong></div><div class="card">Outcome<strong>{status}</strong></div></div></section>
<section><h2>What changed</h2><p>Added supplier-share lag levels, rolling means and volatility, cross-supplier rank, momentum acceleration and current market-level features. Tested tuned classifiers, continuous share-change regressors and a validation-selected soft-voting ensemble.</p><div class="warning"><strong>Integrity rule:</strong> selection uses only the 2018-June 2021 validation period. Disruption and post-disruption observations are not used to choose the model. Accuracy is not inflated by changing class thresholds or using future features.</div></section>
<section><h2>Evaluation metrics</h2><table><thead><tr><th>Period</th><th>Horizon</th><th>Selected model</th><th>Exact accuracy</th><th>Balanced accuracy</th><th>Macro F1</th><th>Adjacent-class accuracy</th><th>Brier</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Original versus improved</h2><img src="{image_uri(VIS_DIR/'original_vs_improved_accuracy.png')}"></section>
<section><h2>Confusion matrices</h2><img src="{image_uri(VIS_DIR/'improved_confusion_matrices.png')}"></section>
<section><h2>Feature importance</h2><img src="{image_uri(VIS_DIR/'improved_feature_importance.png')}"></section>
<section><h2>Latest forecasts</h2><table><thead><tr><th>Supplier</th><th>Horizon</th><th>Target</th><th>Predicted class</th><th>Confidence</th></tr></thead><tbody>{forecast_rows}</tbody></table></section>
</main></body></html>"""
    (VIS_DIR / "titanium_model_b_improved_accuracy_dashboard.html").write_text(html, encoding="utf-8")


def serializable_selection(selection: dict) -> dict:
    result = {
        "kind": selection["kind"],
        "name": selection["name"],
        "temperature": selection["temperature"],
        "validation_metrics": selection["validation_metrics"],
    }
    if selection["kind"] == "ensemble":
        result["members"] = [asdict(item) for item in selection["members"]]
    else:
        result["candidate"] = asdict(selection["candidate"])
    return result


def portable_model_bundle(bundle: dict) -> dict:
    """Remove script-local dataclass objects so joblib loads in any process."""
    portable = {key: value for key, value in bundle.items() if key != "horizons"}
    portable["horizons"] = {}
    for horizon, payload in bundle["horizons"].items():
        portable["horizons"][horizon] = {
            "selection": serializable_selection(payload["selection"]),
            "fitted_models": [
                {"candidate": asdict(candidate), "pipeline": pipeline}
                for candidate, pipeline in payload["fitted_models"]
            ],
            "training_rows": payload["training_rows"],
        }
    return portable


def main() -> None:
    ensure_dirs()
    data, thresholds = load_data()
    selections, validation_comparison, selected_validation = select_models(data, thresholds)

    evaluation_frames = []
    for horizon in HORIZONS:
        evaluation_frames.append(
            rolling_evaluation(
                data, thresholds, selections[horizon], horizon,
                "2021-07-01", "2023-06-01", "disruption_2021_2023",
            )
        )
        evaluation_frames.append(
            rolling_evaluation(
                data, thresholds, selections[horizon], horizon,
                "2023-07-01", "2026-06-01", "post_disruption",
            )
        )
    predictions = pd.concat(evaluation_frames, ignore_index=True)
    metrics = final_metrics(predictions)
    baseline = pd.read_csv(BASELINE_METRICS_PATH)
    comparison = metrics.merge(
        baseline[["period", "horizon_months", "accuracy", "brier_score"]].rename(
            columns={"accuracy": "original_accuracy", "brier_score": "original_brier_score"}
        ),
        on=["period", "horizon_months"],
        how="left",
    )
    comparison["accuracy_change_percentage_points"] = (
        comparison["accuracy"] - comparison["original_accuracy"]
    ) * 100
    comparison["meets_80_pct_accuracy"] = comparison["accuracy"] >= 0.80

    bundle, forecasts = fit_final_models(data, thresholds, selections)
    importance = make_feature_importance(bundle, data)

    validation_comparison.to_csv(RESULTS_DIR / "validation_candidate_comparison.csv", index=False)
    selected_validation.to_csv(RESULTS_DIR / "selected_validation_predictions.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "disruption_and_post_disruption_predictions.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "improved_evaluation_metrics.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "original_vs_improved_metrics.csv", index=False)
    forecasts.to_csv(RESULTS_DIR / "latest_improved_forecasts.csv", index=False)
    importance.to_csv(RESULTS_DIR / "improved_feature_importance.csv", index=False)
    joblib.dump(
        portable_model_bundle(bundle),
        MODELS_DIR / "titanium_model_b_improved_accuracy.joblib",
    )

    manifest = {
        "experiment": "Model B four-class accuracy improvement",
        "source_data": str(DATA_PATH),
        "source_cutoff": data["date"].max().strftime("%Y-%m"),
        "target_thresholds_unchanged": thresholds,
        "selection_period": "2018-01 through 2021-06",
        "test_periods": {
            "disruption": "2021-07 through 2023-06 origins",
            "post_disruption": "2023-07 through latest completed target",
        },
        "features": FEATURES,
        "selections": {f"h{horizon}": serializable_selection(selections[horizon]) for horizon in HORIZONS},
        "accuracy_target": 0.80,
        "accuracy_target_met_all_tests": bool((metrics["accuracy"] >= 0.80).all()),
    }
    (RESULTS_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    make_visualizations(validation_comparison, metrics, baseline, predictions, importance)
    build_dashboard(metrics, validation_comparison, forecasts)

    readme = f"""# Titanium Model B Accuracy Improvement

This experiment preserves the original four-class target and fixed class thresholds. It does not use disruption or post-disruption rows for model selection.

## Changes tested

- Added supplier-share lag levels omitted from the original feature list.
- Added rolling share mean, volatility, range and mean-reversion features.
- Added supplier rank, distance from the leading supplier and momentum acceleration.
- Compared tuned classifiers with regressors that predict continuous future share change before applying the original class thresholds.
- Tested a validation-selected soft-voting ensemble.

## Evaluation

Validation: January 2018 through June 2021. Disruption test: July 2021 through June 2023 origins. Post-disruption test: July 2023 through the latest completed target.

The 80% target is reported honestly. If it is not reached, the experiment does not change the class definitions or leak future information to manufacture a higher score.
"""
    (RUN_DIR / "README.md").write_text(readme, encoding="utf-8")

    print("Improved Model B experiment complete")
    print(comparison[[
        "period", "horizon_months", "selected_model", "original_accuracy", "accuracy",
        "accuracy_change_percentage_points", "balanced_accuracy", "macro_f1",
        "adjacent_class_accuracy", "meets_80_pct_accuracy",
    ]].to_string(index=False))
    print(f"Model: {MODELS_DIR / 'titanium_model_b_improved_accuracy.joblib'}")
    print(f"Dashboard: {VIS_DIR / 'titanium_model_b_improved_accuracy_dashboard.html'}")


def render_saved_results() -> None:
    """Rebuild charts and the dashboard without repeating model training."""
    ensure_dirs()
    validation_comparison = pd.read_csv(RESULTS_DIR / "validation_candidate_comparison.csv")
    metrics = pd.read_csv(RESULTS_DIR / "improved_evaluation_metrics.csv")
    baseline = pd.read_csv(BASELINE_METRICS_PATH)
    predictions = pd.read_csv(RESULTS_DIR / "disruption_and_post_disruption_predictions.csv")
    importance = pd.read_csv(RESULTS_DIR / "improved_feature_importance.csv")
    forecasts = pd.read_csv(RESULTS_DIR / "latest_improved_forecasts.csv")
    make_visualizations(validation_comparison, metrics, baseline, predictions, importance)
    build_dashboard(metrics, validation_comparison, forecasts)
    print(f"Dashboard: {VIS_DIR / 'titanium_model_b_improved_accuracy_dashboard.html'}")


def repair_saved_model_bundle() -> None:
    """Convert the first-run bundle to the portable dictionary-only format."""
    model_path = MODELS_DIR / "titanium_model_b_improved_accuracy.joblib"
    bundle = joblib.load(model_path)
    joblib.dump(portable_model_bundle(bundle), model_path)
    print(f"Portable model bundle: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild visualizations and dashboard from saved result CSV files.",
    )
    parser.add_argument(
        "--repair-model-bundle",
        action="store_true",
        help="Convert an earlier script-local joblib bundle to a portable format.",
    )
    args = parser.parse_args()
    if args.repair_model_bundle:
        repair_saved_model_bundle()
    elif args.render_only:
        render_saved_results()
    else:
        main()
