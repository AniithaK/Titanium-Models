from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


ROOT = Path(
    r"C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM"
    r"\TRAIN READY Processed data- Titanium 2010-2026"
)
MODEL_A = ROOT / "training titanium model A" / "production_v6_titanium_core_corrected"
MODEL_B = ROOT / "Training Titanium Model B"
REVIEW = ROOT / "Titanium Model A and B Performance Review"
OUTPUT = REVIEW / "RESULT ANALYSIS"

SPLIT_LABELS = {
    "validation": "Validation",
    "disruption_test": "Disruption test",
    "post_disruption_test": "Post-disruption test",
}
CLASSES = ["collapse", "fall", "stable", "rise"]
COLORS = {
    "actual": "#087D80",
    "predicted": "#CC5D2B",
    "interval": "#E9B79F",
    "good": "#337653",
    "warning": "#A97018",
    "bad": "#A43B32",
    "muted": "#617579",
}


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "a_detail": pd.read_csv(REVIEW / "results" / "model_a_ground_truth_vs_predictions.csv"),
        "a_metrics": pd.read_csv(REVIEW / "results" / "model_a_performance_summary.csv"),
        "b_detail": pd.read_csv(REVIEW / "results" / "model_b_ground_truth_vs_predictions.csv"),
        "b_metrics": pd.read_csv(REVIEW / "results" / "model_b_performance_summary.csv"),
        "a_future": pd.read_csv(MODEL_A / "results" / "latest" / "latest_reference_forecasts.csv"),
        "b_future": pd.read_csv(MODEL_B / "results" / "latest_country_supply_shift_forecasts.csv"),
    }


def prepare_model_a(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = detail.copy()
    data = data[data["model_output_available"].astype(bool)].copy()
    data = data[data["actual_price_gbp_per_kg"].notna()].copy()
    data["split_label"] = data["split"].map(SPLIT_LABELS)
    data["forecast_origin_date"] = pd.to_datetime(data["forecast_origin"] + "-01")
    data["target_month_date"] = pd.to_datetime(data["target_month"] + "-01")
    data["signed_deviation_pct"] = (
        (data["p50_price_gbp_per_kg"] - data["actual_price_gbp_per_kg"])
        / data["actual_price_gbp_per_kg"]
        * 100
    )
    data["absolute_deviation_pct"] = data["signed_deviation_pct"].abs()
    data["squared_error"] = (
        data["p50_price_gbp_per_kg"] - data["actual_price_gbp_per_kg"]
    ) ** 2
    data["interval_width_gbp_per_kg"] = (
        data["p95_price_gbp_per_kg"] - data["p05_price_gbp_per_kg"]
    )
    data["interval_width_pct_of_actual"] = (
        data["interval_width_gbp_per_kg"] / data["actual_price_gbp_per_kg"] * 100
    )
    actual_move = np.sign(data["actual_price_gbp_per_kg"] - data["origin_price_gbp_per_kg"])
    predicted_move = np.sign(data["p50_price_gbp_per_kg"] - data["origin_price_gbp_per_kg"])
    data["actual_direction"] = np.select(
        [actual_move > 0, actual_move < 0], ["up", "down"], default="unchanged"
    )
    data["predicted_direction"] = np.select(
        [predicted_move > 0, predicted_move < 0], ["up", "down"], default="unchanged"
    )
    data["direction_correct"] = data["actual_direction"] == data["predicted_direction"]
    data["within_5_pct"] = data["absolute_deviation_pct"] <= 5
    data["within_10_pct"] = data["absolute_deviation_pct"] <= 10
    data["within_20_pct"] = data["absolute_deviation_pct"] <= 20

    groups = ["split", "split_label", "hs_code", "hs_product", "horizon_months", "model"]
    summary = (
        data.groupby(groups, dropna=False)
        .agg(
            rows=("actual_price_gbp_per_kg", "size"),
            mae_gbp_per_kg=("absolute_error_gbp_per_kg", "mean"),
            rmse_gbp_per_kg=("squared_error", lambda x: float(np.sqrt(x.mean()))),
            mean_signed_deviation_pct=("signed_deviation_pct", "mean"),
            mean_absolute_deviation_pct=("absolute_deviation_pct", "mean"),
            median_absolute_deviation_pct=("absolute_deviation_pct", "median"),
            within_5_pct_rate=("within_5_pct", "mean"),
            within_10_pct_rate=("within_10_pct", "mean"),
            within_20_pct_rate=("within_20_pct", "mean"),
            direction_accuracy=("direction_correct", "mean"),
            p05_p95_coverage=("covered_p05_p95", "mean"),
            mean_interval_width_gbp_per_kg=("interval_width_gbp_per_kg", "mean"),
        )
        .reset_index()
    )
    return data, summary


def prepare_model_b(detail: pd.DataFrame, saved_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = detail.copy()
    data["split_label"] = data["split"].map(SPLIT_LABELS)
    data["forecast_origin_date"] = pd.to_datetime(data["forecast_origin"] + "-01")
    data["target_month_date"] = pd.to_datetime(data["target_month"] + "-01")
    data["correct_numeric"] = data["correct"].astype(int)
    probability_columns = [f"probability_{label}" for label in CLASSES]
    matrix = data[probability_columns].to_numpy(dtype=float)
    data["probability_sum"] = matrix.sum(axis=1)
    sorted_probs = np.sort(matrix, axis=1)
    data["second_highest_probability"] = sorted_probs[:, -2]
    data["prediction_margin"] = data["confidence"] - data["second_highest_probability"]
    data["actual_class_probability"] = [
        row[f"probability_{row['actual_class']}"] for _, row in data.iterrows()
    ]
    data["confidence_minus_outcome"] = data["confidence"] - data["correct_numeric"]

    grouped = (
        data.groupby(["split", "split_label", "horizon_months"], dropna=False)
        .agg(
            rows=("correct_numeric", "size"),
            accuracy=("correct_numeric", "mean"),
            mean_model_confidence=("confidence", "mean"),
            median_model_confidence=("confidence", "median"),
            mean_actual_class_probability=("actual_class_probability", "mean"),
            mean_prediction_margin=("prediction_margin", "mean"),
        )
        .reset_index()
    )
    grouped["confidence_minus_accuracy"] = grouped["mean_model_confidence"] - grouped["accuracy"]
    summary = grouped.merge(
        saved_metrics[
            [
                "split",
                "horizon_months",
                "balanced_accuracy",
                "macro_f1",
                "brier_score",
                "log_loss",
                "expected_calibration_error",
                "max_calibration_gap",
                "selected_model",
                "historical_frequency_brier",
                "persistence_brier",
                "passes_acceptance",
            ]
        ],
        on=["split", "horizon_months"],
        how="left",
        validate="one_to_one",
    )
    return data, summary


def add_future_fields(a_future: pd.DataFrame, b_future: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = a_future.copy()
    a["forecast_origin_date"] = pd.to_datetime(a["forecast_origin"] + "-01")
    a["target_month_date"] = pd.to_datetime(a["target_month"] + "-01")
    a["p50_change_from_origin_pct"] = (
        (a["p50_price_gbp_per_kg"] - a["origin_price_gbp_per_kg"])
        / a["origin_price_gbp_per_kg"]
        * 100
    )
    a["interval_width_gbp_per_kg"] = a["p95_price_gbp_per_kg"] - a["p05_price_gbp_per_kg"]
    a["ground_truth_status"] = "Not available in source data ending June 2026"

    b = b_future.copy()
    b["forecast_origin_date"] = pd.to_datetime(b["forecast_origin"] + "-01")
    b["target_month_date"] = pd.to_datetime(b["target_month"] + "-01")
    probability_columns = [f"probability_{label}" for label in CLASSES]
    b["confidence"] = b[probability_columns].max(axis=1)
    b["probability_sum"] = b[probability_columns].sum(axis=1)
    b["ground_truth_status"] = "Not available in source data ending June 2026"
    return a, b


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBF8F1",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CFC5B4",
            "axes.labelcolor": "#12353B",
            "text.color": "#12353B",
            "xtick.color": "#617579",
            "ytick.color": "#617579",
            "axes.titleweight": "bold",
        }
    )


def plot_model_a_split(data: pd.DataFrame, split: str, path: Path) -> None:
    subset = data[data["split"] == split].copy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), sharex=False)
    for row_idx, hs in enumerate([810820, 810890]):
        for col_idx, horizon in enumerate([3, 6, 12]):
            ax = axes[row_idx, col_idx]
            frame = subset[(subset["hs_code"] == hs) & (subset["horizon_months"] == horizon)].sort_values("target_month_date")
            if frame.empty:
                ax.axis("off")
                continue
            x = frame["target_month_date"]
            ax.fill_between(
                x,
                frame["p05_price_gbp_per_kg"],
                frame["p95_price_gbp_per_kg"],
                color=COLORS["interval"],
                alpha=0.35,
                label="P05-P95",
            )
            ax.plot(x, frame["actual_price_gbp_per_kg"], color=COLORS["actual"], lw=2.0, label="Actual")
            ax.plot(x, frame["p50_price_gbp_per_kg"], color=COLORS["predicted"], lw=1.8, label="P50")
            ax.set_title(f"HS {hs} - H{horizon}")
            ax.set_ylabel("GBP/kg")
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, ncol=3, fontsize=8)
    title = SPLIT_LABELS[split]
    fig.suptitle(f"Model A actual price versus P50 forecast - {title}", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_model_a_deviation(summary: pd.DataFrame, path: Path) -> None:
    plot = summary.copy()
    plot["label"] = (
        plot["split_label"].str.replace(" test", "", regex=False)
        + "\n" + plot["hs_code"].astype(str)
        + " H" + plot["horizon_months"].astype(str)
    )
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    colors = [COLORS["good"] if v <= 10 else COLORS["warning"] if v <= 20 else COLORS["bad"] for v in plot["mean_absolute_deviation_pct"]]
    axes[0].bar(np.arange(len(plot)), plot["mean_absolute_deviation_pct"], color=colors)
    axes[0].axhline(10, color=COLORS["warning"], ls="--", lw=1.3, label="10% deviation")
    axes[0].set_ylabel("Mean absolute deviation (%)")
    axes[0].set_title("P50 percentage deviation from ground truth")
    axes[0].legend(frameon=False)
    axes[1].bar(np.arange(len(plot)), plot["p05_p95_coverage"] * 100, color="#087D80")
    axes[1].axhline(90, color=COLORS["predicted"], ls="--", lw=1.3, label="90% coverage target")
    axes[1].set_ylabel("Actual inside P05-P95 (%)")
    axes[1].set_title("Prediction interval coverage")
    axes[1].set_xticks(np.arange(len(plot)), plot["label"], rotation=48, ha="right", fontsize=8)
    axes[1].legend(frameon=False)
    fig.suptitle("Model A error and uncertainty performance", fontsize=18, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_model_b_performance(summary: pd.DataFrame, path: Path) -> None:
    plot = summary.copy()
    plot["label"] = (
        plot["split_label"].str.replace(" test", "", regex=False)
        + " H" + plot["horizon_months"].astype(str)
    )
    x = np.arange(len(plot))
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.7), sharex=True)
    width = 0.34
    axes[0].bar(x - width / 2, plot["accuracy"] * 100, width, color=COLORS["actual"], label="Observed accuracy")
    axes[0].bar(x + width / 2, plot["mean_model_confidence"] * 100, width, color=COLORS["predicted"], label="Mean confidence")
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Observed correctness versus predicted-class confidence")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].bar(x - width, plot["brier_score"], width, color="#087D80", label="Model")
    axes[1].bar(x, plot["historical_frequency_brier"], width, color="#A97018", label="Historical frequency")
    axes[1].bar(x + width, plot["persistence_brier"], width, color="#A43B32", label="Persistence")
    axes[1].set_ylabel("Brier score (lower is better)")
    axes[1].set_title("Probability prediction versus baselines")
    axes[1].set_xticks(x, plot["label"], rotation=40, ha="right", fontsize=8)
    axes[1].legend(frameon=False, ncol=3)
    fig.suptitle("Model B classification and probability performance", fontsize=18, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_model_b_confusions(data: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    splits = ["validation", "disruption_test", "post_disruption_test"]
    for row_idx, split in enumerate(splits):
        for col_idx, horizon in enumerate([3, 6, 12]):
            ax = axes[row_idx, col_idx]
            frame = data[(data["split"] == split) & (data["horizon_months"] == horizon)]
            cm = confusion_matrix(frame["actual_class"], frame["predicted_class"], labels=CLASSES)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="GnBu",
                cbar=False,
                xticklabels=CLASSES,
                yticklabels=CLASSES,
                ax=ax,
            )
            ax.set_title(f"{SPLIT_LABELS[split]} - H{horizon}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.suptitle("Model B ground-truth versus predicted classes", fontsize=18, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_model_b_reliability(data: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, horizon in zip(axes, [3, 6, 12]):
        for split, color in zip(
            ["validation", "disruption_test", "post_disruption_test"],
            ["#A97018", "#087D80", "#CC5D2B"],
        ):
            frame = data[(data["horizon_months"] == horizon) & (data["split"] == split)].copy()
            frame["confidence_bin"] = pd.cut(
                frame["confidence"], bins=np.linspace(0, 1, 11), include_lowest=True
            )
            reliability = frame.groupby("confidence_bin", observed=True).agg(
                mean_confidence=("confidence", "mean"),
                observed_accuracy=("correct_numeric", "mean"),
                rows=("correct_numeric", "size"),
            )
            reliability = reliability[reliability["rows"] >= 3]
            ax.plot(
                reliability["mean_confidence"],
                reliability["observed_accuracy"],
                marker="o",
                lw=1.8,
                color=color,
                label=SPLIT_LABELS[split],
            )
        ax.plot([0, 1], [0, 1], ls="--", color="#617579", lw=1)
        ax.set_title(f"H{horizon}")
        ax.set_xlabel("Mean model confidence")
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Model B reliability: confidence versus observed correctness", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_future(a_future: pd.DataFrame, b_future: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9.5), gridspec_kw={"height_ratios": [1.05, 1.2]})
    for hs, color, marker in [(810820, "#087D80", "o"), (810890, "#CC5D2B", "s")]:
        frame = a_future[a_future["hs_code"] == hs].sort_values("target_month_date")
        x = frame["target_month_date"]
        axes[0].plot(x, frame["p50_price_gbp_per_kg"], color=color, marker=marker, lw=2, label=f"HS {hs} P50")
        axes[0].fill_between(x, frame["p05_price_gbp_per_kg"], frame["p95_price_gbp_per_kg"], color=color, alpha=0.14)
    axes[0].set_title("Model A future P50 forecasts with P05-P95 ranges")
    axes[0].set_ylabel("GBP/kg")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].tick_params(axis="x", rotation=25)

    pivot = b_future.pivot(index="country", columns="horizon_months", values="confidence").reindex(
        ["USA", "Japan", "France", "Russian Federation", "China", "Other countries"]
    )
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Predicted-class confidence (%)"}, ax=axes[1])
    axes[1].set_title("Model B future predicted-class confidence")
    axes[1].set_xlabel("Horizon (months)")
    axes[1].set_ylabel("Supplier")
    fig.suptitle("Future forecasts from June 2026 origin", fontsize=18, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_outputs(inputs: dict[str, pd.DataFrame]) -> dict[str, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    a_detail, a_summary = prepare_model_a(inputs["a_detail"])
    b_detail, b_summary = prepare_model_b(inputs["b_detail"], inputs["b_metrics"])
    a_future, b_future = add_future_fields(inputs["a_future"], inputs["b_future"])

    paths = {
        "a_detail": OUTPUT / "model_a_ground_truth_vs_predictions_analysis.csv",
        "a_summary": OUTPUT / "model_a_performance_analysis.csv",
        "b_detail": OUTPUT / "model_b_ground_truth_vs_predictions_analysis.csv",
        "b_summary": OUTPUT / "model_b_performance_analysis.csv",
        "a_future": OUTPUT / "model_a_future_forecasts_2026_2027.csv",
        "b_future": OUTPUT / "model_b_future_forecasts_2026_2027.csv",
    }
    a_detail.to_csv(paths["a_detail"], index=False)
    a_summary.to_csv(paths["a_summary"], index=False)
    b_detail.to_csv(paths["b_detail"], index=False)
    b_summary.to_csv(paths["b_summary"], index=False)
    a_future.to_csv(paths["a_future"], index=False)
    b_future.to_csv(paths["b_future"], index=False)

    set_plot_style()
    chart_paths = {
        "a_validation": OUTPUT / "model_a_validation_actual_vs_predicted.png",
        "a_disruption": OUTPUT / "model_a_disruption_actual_vs_predicted.png",
        "a_post": OUTPUT / "model_a_post_disruption_actual_vs_predicted.png",
        "a_deviation": OUTPUT / "model_a_percentage_deviation_and_coverage.png",
        "b_performance": OUTPUT / "model_b_accuracy_confidence_and_brier.png",
        "b_confusion": OUTPUT / "model_b_ground_truth_confusion_matrices.png",
        "b_reliability": OUTPUT / "model_b_probability_reliability.png",
        "future": OUTPUT / "future_forecasts_2026_2027.png",
    }
    plot_model_a_split(a_detail, "validation", chart_paths["a_validation"])
    plot_model_a_split(a_detail, "disruption_test", chart_paths["a_disruption"])
    plot_model_a_split(a_detail, "post_disruption_test", chart_paths["a_post"])
    plot_model_a_deviation(a_summary, chart_paths["a_deviation"])
    plot_model_b_performance(b_summary, chart_paths["b_performance"])
    plot_model_b_confusions(b_detail, chart_paths["b_confusion"])
    plot_model_b_reliability(b_detail, chart_paths["b_reliability"])
    plot_future(a_future, b_future, chart_paths["future"])

    manifest = {
        "source_cutoff": "2026-06",
        "model_a_future_targets": sorted(a_future["target_month"].unique().tolist()),
        "model_b_future_targets": sorted(b_future["target_month"].unique().tolist()),
        "model_a_historical_rows": int(len(a_detail)),
        "model_b_historical_rows": int(len(b_detail)),
        "outputs": {key: str(value) for key, value in {**paths, **chart_paths}.items()},
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**paths, **chart_paths}


def main() -> None:
    outputs = save_outputs(load_inputs())
    print("Result analysis data and visualizations created")
    print(f"Output folder: {OUTPUT}")
    for key, path in outputs.items():
        print(f"{key}: {path.name}")


if __name__ == "__main__":
    main()
