from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "results" / "latest" / "selected_model_feature_importance.csv"
OUTPUT_PATH = ROOT / "visualizations" / "titanium_model_a_selected_feature_importance.png"

PRODUCT_NAMES = {
    "810820": "HS 810820: Unwrought titanium and powders",
    "810890": "HS 810890: Bars, sheet, forgings and other articles",
}


def readable_feature_name(name: str) -> str:
    replacements = {
        "gbp": "GBP",
        "fx": "FX",
        "eruc": "ERUC",
        "usgs": "USGS",
        "usd": "USD",
        "yoy": "YoY",
    }
    words = name.split("_")
    return " ".join(replacements.get(word, word) for word in words)


def main() -> None:
    importance = pd.read_csv(INPUT_PATH, dtype={"hs_code": str})
    importance["horizon_months"] = importance["horizon_months"].astype(int)
    importance["relative_importance_pct"] = pd.to_numeric(
        importance["relative_importance_pct"], errors="coerce"
    ).fillna(0.0)

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    panel_labels = ["A", "B", "C", "D", "E", "F"]

    for row, hs_code in enumerate(("810820", "810890")):
        for col, horizon in enumerate((3, 6, 12)):
            ax = axes[row, col]
            subset = importance[
                (importance["hs_code"] == hs_code)
                & (importance["horizon_months"] == horizon)
            ].copy()
            model_name = subset["selected_model"].iloc[0] if not subset.empty else "unknown"
            nonzero = subset[subset["relative_importance_pct"] > 0].nlargest(
                12, "relative_importance_pct"
            )

            ax.set_title(
                f"HS {hs_code} | H{horizon} | {model_name.replace('_', ' ').title()}",
                fontsize=12.5,
                fontweight="bold",
                color="#12343b",
                pad=12,
            )
            ax.text(
                -0.08,
                1.04,
                panel_labels[row * 3 + col],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color="#c95d2e",
            )

            if nonzero.empty:
                ax.text(
                    0.5,
                    0.56,
                    "No learned feature weights",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="#12343b",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.5,
                    0.42,
                    "Adaptive no-change is a rules-based baseline.\n"
                    "It does not estimate feature importance.",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#5d6b70",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#d7cdbf")
                continue

            nonzero = nonzero.sort_values("relative_importance_pct")
            labels = [readable_feature_name(value) for value in nonzero["feature"]]
            values = nonzero["relative_importance_pct"]
            colors = ["#117c82"] * len(values)
            colors[-1] = "#c95d2e"
            bars = ax.barh(labels, values, color=colors, height=0.68)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, color="#314a52")
            ax.set_xlabel("Relative importance (%)", fontsize=10, color="#405860")
            ax.grid(axis="x", color="#ddd4c8", linewidth=0.8, alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=8.7, colors="#263f47")
            ax.tick_params(axis="x", labelsize=9, colors="#526970")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#bdb3a6")
            ax.spines["bottom"].set_color("#bdb3a6")
            ax.set_xlim(0, max(values.max() * 1.23, 1))

    fig.patch.set_facecolor("#f7f4ee")
    for ax in axes.flat:
        ax.set_facecolor("#fffdf8")

    fig.suptitle(
        "Titanium Model A: Selected-Model Feature Importance",
        fontsize=24,
        fontweight="bold",
        color="#0d3942",
        y=0.988,
    )
    fig.text(
        0.5,
        0.958,
        "Top 12 non-zero features for each saved HS-code and forecast-horizon model",
        ha="center",
        fontsize=12,
        color="#5d6b70",
    )
    fig.text(
        0.01,
        0.012,
        "Source: results/latest/selected_model_feature_importance.csv. "
        "Importance is model-specific and should not be interpreted as causality.",
        fontsize=9,
        color="#68787d",
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.91,
        bottom=0.065,
        wspace=0.62,
        hspace=0.36,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved feature-importance plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
