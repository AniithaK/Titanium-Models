from __future__ import annotations

import base64
import importlib.util
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss


HERE = Path(__file__).resolve()
RUN_DIR = HERE.parents[1]
MODEL_A_DIR = RUN_DIR.parent
IMPROVED_SCRIPT = MODEL_A_DIR / "improved_v2" / "code" / "train_titanium_model_a_improved_v2.py"
SOURCE_DATA_PATH = RUN_DIR / "data" / "titanium_model_a_production_v3_training.csv"

RESULTS_DIR = RUN_DIR / "results"
MODELS_DIR = RUN_DIR / "models"
VIS_DIR = RUN_DIR / "visualizations"
MONITORING_DIR = RUN_DIR / "monitoring"
DOCS_DIR = RUN_DIR / "documentation"


def load_improved_module():
    original_module_from_spec = importlib.util.module_from_spec

    def registered_module_from_spec(spec):
        module = original_module_from_spec(spec)
        sys.modules[spec.name] = module
        return module

    importlib.util.module_from_spec = registered_module_from_spec
    try:
        spec = importlib.util.spec_from_file_location("model_a_v3_engine", IMPROVED_SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {IMPROVED_SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        importlib.util.module_from_spec = original_module_from_spec


engine = load_improved_module()
base = engine.base

FEATURES = [
    "price_log_change_gbp_1m",
    "price_log_change_gbp_2m",
    "price_log_change_gbp_3m",
    "price_log_change_gbp_6m",
    "price_log_change_gbp_12m",
    "price_volatility_6m",
    "price_volatility_12m",
    "quantity_log_change_1m",
    "quantity_log_change_3m",
    "quantity_log_change_12m",
    "quantity_volatility_6m",
    "quantity_volatility_12m",
    "log_net_weight_kg",
    "volume_ratio_to_prior_12m_median",
    "low_volume_history_flag",
    "supplier_share_usa",
    "supplier_share_japan",
    "supplier_share_france",
    "supplier_share_russia",
    "supplier_share_china",
    "supplier_share_other",
    "supplier_hhi_weight",
    "top_supplier_share_weight",
    "supplier_concentration_change_3m",
    "top_supplier_share_change_3m",
    "log_partner_count",
    "fx_log_change_1m",
    "fx_log_change_3m",
    "fx_log_change_6m",
    "fx_log_change_12m",
    "fx_volatility_6m",
    "eruc_log_change_1m",
    "eruc_log_change_3m",
    "eruc_log_change_6m",
    "eruc_log_change_12m",
    "eruc_volatility_6m",
    "world_bank_aluminum_log_change_1m",
    "world_bank_aluminum_log_change_3m",
    "world_bank_aluminum_log_change_6m",
    "world_bank_nickel_log_change_1m",
    "world_bank_nickel_log_change_3m",
    "world_bank_nickel_log_change_6m",
    "world_bank_source_age_months",
    "world_bank_stale_flag",
    "usgs_titanium_sponge_proxy_usd_per_kg",
    "usgs_titanium_sponge_proxy_log_change_yoy",
    "usgs_source_age_years",
    "usgs_stale_flag",
    "month_sin",
    "month_cos",
    "price_shock_history_flag",
    "world_weight_estimated_by_un_flag",
    "eu_collection_change_flag",
]


def configure_engine() -> None:
    engine.RUN_DIR = RUN_DIR
    engine.SOURCE_DATA_PATH = SOURCE_DATA_PATH
    engine.FEATURES = FEATURES

    def v3_sample_weights(train: pd.DataFrame, horizon: int) -> np.ndarray:
        weights = train["training_sample_weight_v3"].to_numpy(dtype=float).copy()
        target_quality = f"target_quality_flag_h{horizon}"
        if target_quality in train:
            weights *= np.where(train[target_quality].fillna(0).astype(float).gt(0), 0.70, 1.0)
        return np.clip(weights, 0.10, 2.0)

    engine.sample_weights = v3_sample_weights
    base.RUN_DIR = RUN_DIR
    base.SOURCE_DIR = RUN_DIR
    base.SOURCE_DATA_PATH = SOURCE_DATA_PATH
    base.DATA_DIR = RUN_DIR / "data"
    base.MODELS_DIR = MODELS_DIR
    base.RESULTS_DIR = RESULTS_DIR / "validation"
    base.VIS_DIR = VIS_DIR
    base.DOCS_DIR = DOCS_DIR
    base.DATA_COPY_PATH = base.DATA_DIR / SOURCE_DATA_PATH.name
    base.FEATURES = FEATURES
    base.CANDIDATES = engine.CANDIDATES
    base.fit_candidate = engine.fit_candidate
    base.predict_candidate = engine.predict_candidate
    base.feature_importance = engine.feature_importance
    base.MODEL_COLORS = {
        "linear_quantile": "#0d7c80",
        "gradient_boosting_quantile": "#d05c2b",
        "random_forest_distribution": "#d39b28",
        "elastic_net_residual": "#355c9a",
        "huber_residual": "#8a4f7d",
        "adaptive_no_change": "#54824b",
        base.BASELINE_NAME: "#66757d",
    }


def ensure_directories() -> None:
    for path in (
        RESULTS_DIR / "validation",
        RESULTS_DIR / "stress_2022_2023",
        RESULTS_DIR / "post_2024_2025",
        RESULTS_DIR / "latest",
        MODELS_DIR,
        VIS_DIR,
        MONITORING_DIR,
        DOCS_DIR,
        base.DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def build_validation_history(
    validation_predictions: pd.DataFrame,
    selected: dict[tuple[int, int], base.Candidate],
    data: pd.DataFrame,
) -> dict[tuple[int, int, str], list[dict[str, object]]]:
    history: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    volatility = data[["hs_code", "period_m", "price_volatility_6m"]].copy()
    volatility["forecast_origin"] = volatility["period_m"].astype(str)
    thresholds = (
        data[data["period_m"] <= pd.Period("2022-01", "M")]
        .groupby("hs_code")["price_volatility_6m"]
        .median()
        .to_dict()
    )
    for (hs_code, horizon), candidate in selected.items():
        group = validation_predictions[
            (validation_predictions["hs_code"] == hs_code)
            & (validation_predictions["horizon_months"] == horizon)
            & (validation_predictions["model"] == candidate.name)
        ].merge(
            volatility[["hs_code", "forecast_origin", "price_volatility_6m"]],
            on=["hs_code", "forecast_origin"],
            how="left",
        )
        records = []
        for row in group.itertuples(index=False):
            score = max(
                float(row.raw_p05_log_change) - float(row.actual_log_change),
                float(row.actual_log_change) - float(row.raw_p95_log_change),
                0.0,
            )
            records.append(
                {
                    "score": score,
                    "regime": "high"
                    if float(row.price_volatility_6m) > float(thresholds[hs_code])
                    else "normal",
                    "available_date": pd.Timestamp("2022-01-31"),
                }
            )
        history[(hs_code, horizon, candidate.name)] = records
    return history


def rolling_offset(history: list[dict[str, object]], regime: str, cutoff: pd.Timestamp) -> float:
    available = [item for item in history if pd.Timestamp(item["available_date"]) <= cutoff]
    same_regime = [float(item["score"]) for item in available if item["regime"] == regime]
    pool = same_regime[-36:] if len(same_regime) >= 12 else [float(item["score"]) for item in available[-36:]]
    if not pool:
        return 0.0
    return base.conformal_quantile(np.asarray(pool, dtype=float))


def calculate_metrics(predictions: pd.DataFrame, recovery_deadline: str | None) -> pd.DataFrame:
    records = []
    for keys, group in predictions.groupby(["hs_code", "horizon_months", "model"]):
        group = group.sort_values("forecast_origin").copy()
        expected = len(group)
        valid = group[group["model_output_available"]]
        coverage = float(group["covered_p05_p95"].sum() / expected)
        mis = float(group["interval_score_alpha_010"].fillna(np.inf).mean())
        pinball = float(
            np.mean(
                [
                    mean_pinball_loss(
                        valid["actual_log_change_gbp_per_kg"],
                        valid[f"{base.quantile_suffix(q)}_log_change"],
                        alpha=q,
                    )
                    for q in base.QUANTILES
                ]
            )
        )
        mae = float(mean_absolute_error(valid["actual_log_change_gbp_per_kg"], valid["p50_log_change"]))
        group["rolling_6m_coverage"] = group["covered_p05_p95"].astype(float).rolling(6, min_periods=6).mean()
        if recovery_deadline is None:
            recovery_month = None
            recovered = True
        else:
            recovered_rows = group[
                (group["forecast_origin"] <= recovery_deadline)
                & (group["rolling_6m_coverage"] >= 0.90)
            ]
            recovery_month = None if recovered_rows.empty else str(recovered_rows.iloc[0]["forecast_origin"])
            recovered = recovery_month is not None
        records.append(
            {
                "hs_code": keys[0],
                "horizon_months": keys[1],
                "model": keys[2],
                "expected_origins": expected,
                "model_outputs": int(group["model_output_available"].sum()),
                "no_output_misses": expected - int(group["model_output_available"].sum()),
                "p05_p95_coverage": coverage,
                "mean_interval_score_alpha_010": mis,
                "median_mae_log_change": mae,
                "mean_pinball_loss": pinball,
                "coverage_requirement_met": coverage >= 0.90,
                "first_six_month_coverage_recovery": recovery_month,
                "coverage_recovery_requirement_met": recovered,
            }
        )
    metrics = pd.DataFrame(records)
    baseline = metrics[metrics["model"] == base.BASELINE_NAME][
        ["hs_code", "horizon_months", "mean_interval_score_alpha_010"]
    ].rename(columns={"mean_interval_score_alpha_010": "baseline_mean_interval_score_alpha_010"})
    metrics = metrics.merge(baseline, on=["hs_code", "horizon_months"], how="left")
    metrics["mis_ratio_to_baseline"] = (
        metrics["mean_interval_score_alpha_010"] / metrics["baseline_mean_interval_score_alpha_010"]
    )
    metrics["mis_beats_baseline"] = metrics["mis_ratio_to_baseline"] < 1.0
    metrics.loc[metrics["model"] == base.BASELINE_NAME, "mis_beats_baseline"] = False
    metrics["passes_acceptance"] = (
        metrics["horizon_months"].isin([3, 6])
        & metrics["coverage_requirement_met"]
        & metrics["mis_beats_baseline"]
        & metrics["no_output_misses"].eq(0)
        & metrics["coverage_recovery_requirement_met"]
    )
    return metrics


def run_selected_evaluation(
    label: str,
    data: pd.DataFrame,
    validation_history: dict[tuple[int, int, str], list[dict[str, object]]],
    selected: dict[tuple[int, int], base.Candidate],
    start: str,
    end: str,
    recovery_deadline: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = RESULTS_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = {key: list(value) for key, value in validation_history.items()}
    thresholds = (
        data[data["period_m"] <= pd.Period("2022-01", "M")]
        .groupby("hs_code")["price_volatility_6m"]
        .median()
        .to_dict()
    )
    records = []
    for hs_code in base.HS_CODES:
        hs_data = data[data["hs_code"] == hs_code].sort_values("date")
        test = hs_data[
            (hs_data["period_m"] >= pd.Period(start, "M"))
            & (hs_data["period_m"] <= pd.Period(end, "M"))
        ]
        target_availability = hs_data.set_index("period_m")["trade_data_available_date_assumed"].to_dict()
        for horizon in base.HORIZONS:
            candidate = selected[(hs_code, horizon)]
            key = (hs_code, horizon, candidate.name)
            model_history = histories[key]
            baseline_history = [dict(item) for item in model_history]
            for _, row in test.iterrows():
                cutoff = row["trade_data_available_date_assumed"]
                train = base.eligible_training_rows(hs_data, horizon, cutoff)
                actual = row[base.target_column(horizon)]
                output_available = bool(
                    pd.notna(actual)
                    and row[FEATURES].notna().all()
                    and len(train) >= base.MIN_TRAINING_ROWS
                )
                regime = (
                    "high"
                    if float(row["price_volatility_6m"]) > float(thresholds[hs_code])
                    else "normal"
                )
                common = {
                    "evaluation": label,
                    "hs_code": hs_code,
                    "hs_product": row["hs_product"],
                    "horizon_months": horizon,
                    "forecast_origin": str(row["period_m"]),
                    "target_month": str(row["period_m"] + horizon),
                    "forecast_data_available_date": cutoff.date().isoformat(),
                    "actual_log_change_gbp_per_kg": actual,
                    "actual_origin_price_gbp_per_kg": row["import_unit_value_gbp_per_kg"],
                    "actual_target_price_gbp_per_kg": row[base.target_price_column(horizon)],
                    "training_rows": len(train),
                    "model_output_available": output_available,
                    "volatility_regime": regime,
                    "target_quality_status": row["target_quality_status"],
                    "world_bank_stale_flag": int(row["world_bank_stale_flag"]),
                    "usgs_stale_flag": int(row["usgs_stale_flag"]),
                }
                for model_name in (candidate.name, base.BASELINE_NAME):
                    record = {**common, "model": model_name, "selected_model": model_name == candidate.name}
                    if output_available:
                        if model_name == candidate.name:
                            fitted = engine.fit_candidate(candidate, train, horizon)
                            raw = engine.predict_candidate(fitted, row.to_frame().T)
                            history = model_history
                        else:
                            raw = base.baseline_quantiles(train[base.target_column(horizon)])
                            history = baseline_history
                        offset = rolling_offset(history, regime, cutoff)
                        prediction = base.apply_calibration(raw, offset)
                        record["rolling_conformal_expansion"] = offset
                        record.update(base.prediction_record_values(prediction))
                        for quantile in base.QUANTILES:
                            suffix = base.quantile_suffix(quantile)
                            record[f"{suffix}_price_gbp_per_kg"] = float(
                                row["import_unit_value_gbp_per_kg"] * np.exp(prediction[quantile])
                            )
                    records.append(record)

                if output_available:
                    target_period = row["period_m"] + horizon
                    available_date = target_availability.get(target_period, pd.Timestamp.max)
                    model_score = max(raw_value for raw_value in [0.0])
                    fitted = engine.fit_candidate(candidate, train, horizon)
                    candidate_raw = engine.predict_candidate(fitted, row.to_frame().T)
                    model_score = max(
                        candidate_raw[0.05] - float(actual),
                        float(actual) - candidate_raw[0.95],
                        0.0,
                    )
                    model_history.append(
                        {"score": model_score, "regime": regime, "available_date": available_date}
                    )
                    baseline_raw = base.baseline_quantiles(train[base.target_column(horizon)])
                    baseline_score = max(
                        baseline_raw[0.05] - float(actual),
                        float(actual) - baseline_raw[0.95],
                        0.0,
                    )
                    baseline_history.append(
                        {"score": baseline_score, "regime": regime, "available_date": available_date}
                    )

    predictions = pd.DataFrame(records)
    predictions["covered_p05_p95"] = (
        predictions["p05_log_change"].notna()
        & (predictions["actual_log_change_gbp_per_kg"] >= predictions["p05_log_change"])
        & (predictions["actual_log_change_gbp_per_kg"] <= predictions["p95_log_change"])
    )
    predictions["interval_score_alpha_010"] = predictions.apply(
        lambda row: base.interval_score(
            row["actual_log_change_gbp_per_kg"], row["p05_log_change"], row["p95_log_change"]
        )
        if pd.notna(row.get("p05_log_change"))
        else np.nan,
        axis=1,
    )
    metrics = calculate_metrics(predictions, recovery_deadline)
    predictions.to_csv(output_dir / "selected_and_baseline_predictions.csv", index=False)
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    return predictions, metrics


def selected_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["model"] != base.BASELINE_NAME].copy()


def plot_evaluation(metrics: pd.DataFrame, title: str, output: Path) -> Path:
    focus = selected_metric_table(metrics)
    labels = [f"{int(row.hs_code)} H{int(row.horizon_months)}" for row in focus.itertuples()]
    x = np.arange(len(focus))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(x, focus["p05_p95_coverage"] * 100, color="#0d7c80")
    axes[0].axhline(90, color="#9b2f2f", linestyle="--")
    axes[0].set_xticks(x, labels, rotation=25)
    axes[0].set_ylabel("P05-P95 coverage (%)")
    axes[0].set_title("Coverage")
    axes[1].bar(x, focus["mis_ratio_to_baseline"], color="#d05c2b")
    axes[1].axhline(1.0, color="#9b2f2f", linestyle="--")
    axes[1].set_xticks(x, labels, rotation=25)
    axes[1].set_ylabel("MIS / baseline MIS")
    axes[1].set_title("Interval quality")
    fig.suptitle(title, fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_post_predictions(predictions: pd.DataFrame) -> Path:
    selected = predictions[
        predictions["selected_model"] & predictions["horizon_months"].isin([3, 6])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (hs_code, horizon) in zip(
        axes.flat, [(810820, 3), (810820, 6), (810890, 3), (810890, 6)]
    ):
        group = selected[
            (selected["hs_code"] == hs_code) & (selected["horizon_months"] == horizon)
        ].sort_values("target_month")
        x = pd.to_datetime(group["target_month"], format="%Y-%m")
        ax.fill_between(x, group["p05_price_gbp_per_kg"], group["p95_price_gbp_per_kg"], color="#8ec9c3", alpha=0.35)
        ax.plot(x, group["actual_target_price_gbp_per_kg"], color="#173a42", linewidth=2, label="Actual")
        ax.plot(x, group["p50_price_gbp_per_kg"], color="#d05c2b", linewidth=1.8, label="P50")
        ax.set_title(f"HS {hs_code} H{horizon}")
        ax.set_ylabel("GBP/kg")
        ax.grid(axis="both")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("2024-2025 post-disruption evaluation", fontsize=17, fontweight="bold")
    fig.tight_layout()
    path = VIS_DIR / "post_2024_2025_actual_vs_forecast.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_quality(data: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, hs_code in zip(axes, base.HS_CODES):
        group = data[data["hs_code"] == hs_code]
        ax.plot(group["date"], group["target_reliability_score"], color="#0d7c80")
        ax.axhline(0.50, color="#9b2f2f", linestyle="--")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"HS {hs_code}")
        ax.set_ylabel("Reliability score")
        ax.grid(axis="both")
    fig.suptitle("Monthly target reliability diagnostics", fontsize=17, fontweight="bold")
    fig.tight_layout()
    path = VIS_DIR / "target_reliability_over_time.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def save_monitoring(
    data: pd.DataFrame,
    stress_predictions: pd.DataFrame,
    post_predictions: pd.DataFrame,
) -> None:
    selected = pd.concat(
        [
            stress_predictions[stress_predictions["selected_model"]],
            post_predictions[post_predictions["selected_model"]],
        ],
        ignore_index=True,
    )
    selected["median_absolute_error_gbp_per_kg"] = (
        selected["p50_price_gbp_per_kg"] - selected["actual_target_price_gbp_per_kg"]
    ).abs()
    selected["interval_width_gbp_per_kg"] = (
        selected["p95_price_gbp_per_kg"] - selected["p05_price_gbp_per_kg"]
    )
    selected.to_csv(MONITORING_DIR / "historical_forecast_monitoring.csv", index=False)

    latest = data.sort_values("date").groupby("hs_code").tail(1)
    latest[
        [
            "date",
            "hs_code",
            "import_unit_value_gbp_per_kg",
            "target_reliability_score",
            "target_quality_status",
            "world_bank_source_age_months",
            "world_bank_stale_flag",
            "usgs_source_age_years",
            "usgs_stale_flag",
        ]
    ].to_csv(MONITORING_DIR / "latest_data_status.csv", index=False)

    feature_reference = data[data["period_m"] <= pd.Period("2023-12", "M")][FEATURES].agg(["mean", "std"]).T
    drift_records = []
    for row in latest.itertuples(index=False):
        for feature in FEATURES:
            value = float(getattr(row, feature))
            mean = float(feature_reference.loc[feature, "mean"])
            std = float(feature_reference.loc[feature, "std"])
            z_score = 0.0 if not np.isfinite(std) or std == 0 else (value - mean) / std
            drift_records.append(
                {
                    "hs_code": int(row.hs_code),
                    "feature": feature,
                    "latest_value": value,
                    "reference_mean": mean,
                    "reference_std": std,
                    "absolute_z_score": abs(z_score),
                    "drift_alert": abs(z_score) >= 3.0,
                }
            )
    pd.DataFrame(drift_records).to_csv(MONITORING_DIR / "latest_feature_drift.csv", index=False)


def format_metrics(metrics: pd.DataFrame, stress: bool) -> str:
    display = selected_metric_table(metrics)
    display = display[display["horizon_months"].isin([3, 6])].copy()
    display["coverage"] = display["p05_p95_coverage"].map(lambda x: f"{x:.1%}")
    display["mis_ratio"] = display["mis_ratio_to_baseline"].map(lambda x: f"{x:.3f}")
    display["result"] = display["passes_acceptance"].map(lambda x: "PASS" if x else "FAIL")
    columns = ["hs_code", "horizon_months", "model", "coverage", "mis_ratio"]
    if stress:
        display["recovery"] = display["coverage_recovery_requirement_met"].map(lambda x: "Yes" if x else "No")
        columns.append("recovery")
    columns.append("result")
    return display[columns].to_html(index=False, border=0, classes="data-table")


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_dashboard(
    stress_metrics: pd.DataFrame,
    post_metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    charts: list[Path],
) -> Path:
    stress_pass = int(selected_metric_table(stress_metrics)["passes_acceptance"].sum())
    post_pass = int(selected_metric_table(post_metrics)["passes_acceptance"].sum())
    latest_table = forecasts[
        ["hs_code", "horizon_months", "forecast_origin", "target_month", "selected_model", "p05_price_gbp_per_kg", "p50_price_gbp_per_kg", "p95_price_gbp_per_kg"]
    ].copy()
    for column in ["p05_price_gbp_per_kg", "p50_price_gbp_per_kg", "p95_price_gbp_per_kg"]:
        latest_table[column] = latest_table[column].map(lambda x: f"GBP {x:,.2f}/kg")
    figures = "".join(
        f'<figure><img src="{image_uri(path)}" alt="{path.stem}"><figcaption>{path.stem.replace("_", " ")}</figcaption></figure>'
        for path in charts
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Titanium Model A production V3</title><style>
:root{{--ink:#173a42;--teal:#0d7c80;--orange:#d05c2b;--paper:#fffdf8;--sand:#f1ece2;--line:#d8cdbd;}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#f1ece2,#e3efec);color:var(--ink);font-family:"Trebuchet MS",sans-serif}}
header{{background:var(--ink);color:white;padding:50px 6vw}}h1,h2{{font-family:Georgia,serif;font-weight:500}}h1{{font-size:clamp(2.4rem,5vw,4.6rem);margin:.2em 0}}header p{{max-width:960px;line-height:1.65;color:#d3e1df}}
main{{max-width:1500px;margin:auto;padding:30px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:-58px}}.card,section{{background:var(--paper);border:1px solid var(--line);box-shadow:0 12px 30px #173a4215}}
.card{{padding:22px}}.card span{{display:block;text-transform:uppercase;letter-spacing:.12em;font-size:.75rem}}.card strong{{display:block;font:2rem Georgia;color:var(--teal);margin-top:8px}}section{{padding:28px;margin-top:24px;overflow-x:auto}}
.note{{border-left:5px solid var(--orange);padding:14px 18px;background:#fff2e9;line-height:1.55}}.data-table{{width:100%;border-collapse:collapse}}.data-table th{{background:var(--ink);color:white;padding:9px;text-align:left}}.data-table td{{padding:9px;border-bottom:1px solid var(--line)}}figure{{margin:22px 0}}figure img{{width:100%;border:1px solid var(--line)}}figcaption{{color:#69777d;font-size:.82rem}}
@media(max-width:850px){{.cards{{grid-template-columns:1fr;margin-top:15px}}main{{padding:16px}}}}
</style></head><body><header><div>SCI MODEL A / PRODUCTION V3 RESEARCH GATE</div><h1>Titanium import-price forecast evaluation</h1><p>Direct H3, H6 and H12 GBP/kg probabilistic forecasts for HS 810820 and HS 810890. Selection remains pre-February-2022. Uncertainty is calibrated using only residuals available at each forecast origin.</p></header><main>
<div class="cards"><div class="card"><span>Stress passes</span><strong>{stress_pass}/4</strong></div><div class="card"><span>Post-period passes</span><strong>{post_pass}/4</strong></div><div class="card"><span>Latest data month</span><strong>2026-06</strong></div><div class="card"><span>Forecast quantiles</span><strong>7</strong></div></div>
<section><h2>Decision boundary</h2><p class="note">The World Bank command-line refresh was blocked and the local USGS source ends in 2020. V3 exposes source-age and stale-data flags. It does not claim production approval while required external data are stale or acceptance tests fail.</p></section>
<section><h2>2022-2023 disruption stress test</h2>{format_metrics(stress_metrics, True)}</section>
<section><h2>2024-2025 post-disruption evaluation</h2>{format_metrics(post_metrics, False)}<p>The post-disruption period is additional evidence, not a pristine holdout, because the dataset has already been explored during development.</p></section>
<section><h2>Latest reference forecasts</h2>{latest_table.to_html(index=False,border=0,classes="data-table")}</section>
<section><h2>Evidence and diagnostics</h2>{figures}</section>
</main></body></html>"""
    path = VIS_DIR / "titanium_model_a_production_v3_dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


def save_production_response(
    data: pd.DataFrame,
    forecasts: pd.DataFrame,
    post_metrics: pd.DataFrame,
) -> Path:
    latest_rows = data.sort_values("date").groupby("hs_code").tail(1).set_index("hs_code")
    selected_post = selected_metric_table(post_metrics).set_index(["hs_code", "horizon_months"])
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
    for hs_code in base.HS_CODES:
        origin = latest_rows.loc[hs_code]
        hs_forecasts = []
        for row in forecasts[forecasts["hs_code"] == hs_code].sort_values("horizon_months").itertuples(index=False):
            evaluation = selected_post.loc[(hs_code, row.horizon_months)]
            hs_forecasts.append(
                {
                    "horizon_months": int(row.horizon_months),
                    "target_month": row.target_month,
                    "p05_gbp_per_kg": round(float(row.p05_price_gbp_per_kg), 4),
                    "p50_gbp_per_kg": round(float(row.p50_price_gbp_per_kg), 4),
                    "p95_gbp_per_kg": round(float(row.p95_price_gbp_per_kg), 4),
                    "selected_model": row.selected_model,
                    "fallback_used": row.selected_model == "adaptive_no_change",
                    "post_evaluation_pass": bool(evaluation.passes_acceptance),
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
                "forecasts": hs_forecasts,
            }
        )
    path = RESULTS_DIR / "latest" / "model_a_production_response.json"
    path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return path


def write_documentation(stress_metrics: pd.DataFrame, post_metrics: pd.DataFrame) -> None:
    stress_pass = int(selected_metric_table(stress_metrics)["passes_acceptance"].sum())
    post_pass = int(selected_metric_table(post_metrics)["passes_acceptance"].sum())
    text = f"""# Titanium Model A Production V3

## Data directories

- `raw/`: immutable snapshots of the two upstream feature matrices used by the V3 preparation step.
- `prepared/`: engineered V3 matrix plus source-audit and target-quality diagnostics.
- `data/`: synchronized training snapshot consumed by model training and reusable prediction.

`RUN_PRODUCTION_V3.cmd` rebuilds `prepared/` from `raw/`, copies the exact training matrix into `data/`, and then trains and evaluates the models.

## Implemented

- Separate HS 810820 and 810890 direct H3, H6 and H12 models; HS 810830 remains excluded.
- GBP/kg target with Bank of England conversion.
- Price, quantity, supplier-share, supplier-concentration, FX, ERUC, World Bank and USGS features.
- Explicit low-volume, partner-anomaly, estimated-weight and target-reliability diagnostics.
- Quality-weighted model fitting.
- Six validation candidates, including robust Huber and Elastic Net models and a safe no-change fallback.
- Candidate selection using blocked validation ending January 2022.
- Rolling retraining with target-availability checks.
- Volatility-regime-aware rolling conformal P05-P95 calibration.
- Separate 2022-2023 disruption and 2024-2025 post-disruption evaluations.
- Saved model bundles, production-style JSON, feature-drift monitoring and an HTML dashboard.

## Current result

- 2022-2023 stress combinations passing: **{stress_pass}/4**.
- 2024-2025 post-disruption combinations passing: **{post_pass}/4**.

## Blocking limitations

- The official World Bank September 2026 workbook URL was identified, but command-line DNS prevented download. The local World Bank source is used with source-age and stale flags.
- The official USGS titanium-metal workbook available locally ends in 2020. Later rows carry the last known annual observation with a source-age feature and stale flag.
- The 2024-2025 evaluation is useful additional evidence but is not pristine because the dataset was previously explored.
- Deployment remains blocked until required sources are refreshed and a genuinely prospective holdout confirms performance.

## Reproduce

```powershell
& "{RUN_DIR / 'RUN_PRODUCTION_V3.cmd'}"
```
"""
    (RUN_DIR / "README.md").write_text(text, encoding="utf-8")
    (DOCS_DIR / "MODEL_CARD.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_engine()
    ensure_directories()
    base.setup_plot_style()
    data = base.load_data()
    print(f"Loaded {len(data)} V3 rows with {len(FEATURES)} model features")

    base.RESULTS_DIR = RESULTS_DIR / "validation"
    validation_predictions, comparison, calibration, _ = base.run_pretest_validation(data)
    selected = engine.select_with_validation_gates(comparison)
    for key, candidate in sorted(selected.items()):
        print(f"Selected HS {key[0]} H{key[1]}: {candidate.name}")
    history = build_validation_history(validation_predictions, selected, data)

    print("Running 2022-2023 disruption stress evaluation...")
    stress_predictions, stress_metrics = run_selected_evaluation(
        "stress_2022_2023", data, history, selected, "2022-02", "2023-12", "2022-08"
    )
    print("Running 2024-2025 post-disruption evaluation...")
    post_predictions, post_metrics = run_selected_evaluation(
        "post_2024_2025", data, history, selected, "2024-01", "2025-12", None
    )

    base.RESULTS_DIR = RESULTS_DIR / "latest"
    base.MODELS_DIR = MODELS_DIR
    forecasts, importance, manifest = base.fit_reference_models(data, calibration, selected)
    manifest["model_version"] = "production_v3_research"
    manifest["external_data_policy"] = "stale sources are flagged; no unverified values are fabricated"
    (RESULTS_DIR / "latest" / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    stress_chart = plot_evaluation(
        stress_metrics[stress_metrics["horizon_months"].isin([3, 6])],
        "2022-2023 disruption stress test",
        VIS_DIR / "stress_2022_2023_coverage_and_mis.png",
    )
    post_chart = plot_evaluation(
        post_metrics[post_metrics["horizon_months"].isin([3, 6])],
        "2024-2025 post-disruption evaluation",
        VIS_DIR / "post_2024_2025_coverage_and_mis.png",
    )
    charts = [stress_chart, post_chart, plot_post_predictions(post_predictions), plot_target_quality(data)]
    dashboard = build_dashboard(stress_metrics, post_metrics, forecasts, charts)
    save_monitoring(data, stress_predictions, post_predictions)
    response = save_production_response(data, forecasts, post_metrics)
    write_documentation(stress_metrics, post_metrics)

    summary = {
        "stress_h3_h6_passes": int(selected_metric_table(stress_metrics)["passes_acceptance"].sum()),
        "post_h3_h6_passes": int(selected_metric_table(post_metrics)["passes_acceptance"].sum()),
        "dashboard": str(dashboard),
        "production_response": str(response),
    }
    (RESULTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
