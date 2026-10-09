from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE
SOURCE_PATH = EXPERIMENT_DIR.parents[1] / "titanium_model_a_training_with_fx_eruc_2010_2026.csv"
PREPARED_DIR = EXPERIMENT_DIR / "prepared"
OUTPUT_PATH = PREPARED_DIR / "titanium_model_a_improved_training_2010_2026.csv"
QUALITY_REPORT_PATH = PREPARED_DIR / "improved_data_quality_summary.csv"


def add_history_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()

    price_change = group["price_log_change_gbp_1m"]
    quantity_change = group["quantity_log_change_1m"]
    fx_change = group["fx_log_change_1m"]
    eruc_change = group["eruc_log_change_1m"]

    for window in (6, 12):
        minimum = max(3, window // 2)
        group[f"price_volatility_{window}m"] = price_change.rolling(
            window, min_periods=minimum
        ).std(ddof=0)
        group[f"quantity_volatility_{window}m"] = quantity_change.rolling(
            window, min_periods=minimum
        ).std(ddof=0)

    group["fx_volatility_6m"] = fx_change.rolling(6, min_periods=3).std(ddof=0)
    group["eruc_volatility_6m"] = eruc_change.rolling(6, min_periods=3).std(ddof=0)
    group["log_net_weight_kg"] = np.log1p(group["net_weight_kg"].clip(lower=0))
    group["log_partner_count"] = np.log1p(group["partner_count"].clip(lower=0))

    historical_volume_median = group["net_weight_kg"].shift(1).rolling(
        12, min_periods=6
    ).median()
    group["volume_ratio_to_prior_12m_median"] = (
        group["net_weight_kg"] / historical_volume_median.replace(0, np.nan)
    )
    group["volume_ratio_to_prior_12m_median"] = group[
        "volume_ratio_to_prior_12m_median"
    ].clip(0.05, 20.0)
    group["low_volume_history_flag"] = (
        group["volume_ratio_to_prior_12m_median"] < 0.25
    ).astype(int)

    prior_volatility = price_change.shift(1).rolling(12, min_periods=6).std(ddof=0)
    group["price_shock_history_flag"] = (
        price_change.abs() > (3.0 * prior_volatility.clip(lower=0.03))
    ).astype(int)

    # Weight only historical observations during fitting. Test outcomes remain untouched.
    weight = np.sqrt(group["volume_ratio_to_prior_12m_median"].clip(0.10, 4.0))
    weight = weight.fillna(1.0).clip(0.35, 2.0)
    weight *= np.where(group["low_volume_history_flag"].eq(1), 0.65, 1.0)
    weight *= np.where(group["price_shock_history_flag"].eq(1), 0.75, 1.0)
    weight *= np.where(group["weight_estimated_by_un_flag"].eq(1), 0.75, 1.0)
    weight *= np.where(group["partner_anomaly_count"].fillna(0).gt(0), 0.75, 1.0)
    group["training_sample_weight"] = weight.clip(0.20, 2.0)
    return group


def main() -> None:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    required = {
        "hs_code",
        "net_weight_kg",
        "partner_count",
        "price_log_change_gbp_1m",
        "quantity_log_change_1m",
        "fx_log_change_1m",
        "eruc_log_change_1m",
        "weight_estimated_by_un_flag",
        "partner_anomaly_count",
        "supplier_hhi_weight",
        "top_supplier_share_weight",
        "month_sin",
        "month_cos",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Source training data is missing columns: {missing}")

    improved = (
        data.groupby("hs_code", group_keys=False, sort=False)
        .apply(add_history_features, include_groups=True)
        .reset_index(drop=True)
        .sort_values(["date", "hs_code"])
        .reset_index(drop=True)
    )
    improved["feature_set_id"] = "titanium_model_a_improved_v2"

    numeric = improved.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Improved training data contains infinite numeric values.")

    output = improved.copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output.to_csv(OUTPUT_PATH, index=False, float_format="%.10g")

    feature_columns = [
        "price_volatility_6m",
        "price_volatility_12m",
        "quantity_volatility_6m",
        "quantity_volatility_12m",
        "fx_volatility_6m",
        "eruc_volatility_6m",
        "log_net_weight_kg",
        "log_partner_count",
        "volume_ratio_to_prior_12m_median",
        "low_volume_history_flag",
        "price_shock_history_flag",
        "training_sample_weight",
    ]
    report = []
    for hs_code, group in improved.groupby("hs_code"):
        report.append(
            {
                "hs_code": int(hs_code),
                "rows": len(group),
                "start_month": group["date"].min().strftime("%Y-%m"),
                "end_month": group["date"].max().strftime("%Y-%m"),
                "low_volume_months": int(group["low_volume_history_flag"].sum()),
                "price_shock_months": int(group["price_shock_history_flag"].sum()),
                "median_training_weight": float(group["training_sample_weight"].median()),
                "minimum_training_weight": float(group["training_sample_weight"].min()),
                "new_features_complete_rows": int(group[feature_columns].notna().all(axis=1).sum()),
            }
        )
    pd.DataFrame(report).to_csv(QUALITY_REPORT_PATH, index=False)

    print("Improved Titanium Model A training data prepared")
    print(f"Rows: {len(improved)}")
    print(f"Columns: {len(improved.columns)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Quality summary: {QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
