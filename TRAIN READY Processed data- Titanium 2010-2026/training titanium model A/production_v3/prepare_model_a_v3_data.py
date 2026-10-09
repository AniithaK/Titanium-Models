from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
DATA_DIR = HERE / "data"
IMPROVED_V2_PATH = RAW_DIR / "upstream_improved_v2_training_2010_2026.csv"
FULL_SPEC_MATRIX_PATH = RAW_DIR / "upstream_external_features_2010_2026.csv"
OUTPUT_DIR = HERE / "prepared"
OUTPUT_PATH = OUTPUT_DIR / "titanium_model_a_production_v3_training.csv"
TRAINING_SNAPSHOT_PATH = DATA_DIR / OUTPUT_PATH.name
SOURCE_AUDIT_PATH = OUTPUT_DIR / "external_source_audit.csv"
QUALITY_PATH = OUTPUT_DIR / "target_quality_diagnostics.csv"

WORLD_BANK_FEATURES = [
    "world_bank_aluminum_log_change_1m",
    "world_bank_aluminum_log_change_3m",
    "world_bank_aluminum_log_change_6m",
    "world_bank_nickel_log_change_1m",
    "world_bank_nickel_log_change_3m",
    "world_bank_nickel_log_change_6m",
]
USGS_FEATURES = [
    "usgs_titanium_sponge_proxy_usd_per_kg",
    "usgs_titanium_sponge_proxy_log_change_yoy",
    "usgs_source_age_years",
]


def month_distance(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later.dt.year - earlier.dt.year) * 12 + later.dt.month - earlier.dt.month


def load_external_features() -> pd.DataFrame:
    external = pd.read_csv(FULL_SPEC_MATRIX_PATH)
    external["date"] = pd.to_datetime(external["date"], errors="raise")
    required = {
        "date",
        "hs_code",
        "world_bank_input_observed_flag",
        *WORLD_BANK_FEATURES,
        *USGS_FEATURES,
    }
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"Full-spec matrix is missing external columns: {missing}")

    keep = [
        "date",
        "hs_code",
        "world_bank_input_observed_flag",
        *WORLD_BANK_FEATURES,
        *USGS_FEATURES,
    ]
    external = external[keep].copy()
    if external.duplicated(["date", "hs_code"]).any():
        raise ValueError("External feature matrix contains duplicate HS/month rows.")
    return external


def add_external_availability(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["hs_code", "date"]).copy()
    parts = []
    for _, group in data.groupby("hs_code", sort=False):
        group = group.copy()
        wb_observed = group["world_bank_input_observed_flag"].fillna(0).astype(bool)
        group.loc[~wb_observed, WORLD_BANK_FEATURES] = np.nan
        group["world_bank_observed_flag"] = wb_observed.astype(int)
        group["world_bank_last_observed_date"] = group["date"].where(wb_observed).ffill()
        group["world_bank_source_age_months"] = month_distance(
            group["date"], group["world_bank_last_observed_date"]
        ).astype(float)
        group["world_bank_stale_flag"] = (
            group["world_bank_source_age_months"].fillna(999) > 2
        ).astype(int)

        # Missing external changes remain explicit through flags. Neutral zero values
        # allow the fitted pipelines to produce a guarded fallback forecast.
        for column in WORLD_BANK_FEATURES:
            group[f"{column}_missing_flag"] = group[column].isna().astype(int)
            group[column] = group[column].fillna(0.0)
        group = group.drop(columns=["world_bank_input_observed_flag"])

        usgs_observed = group["usgs_titanium_sponge_proxy_usd_per_kg"].notna()
        group["usgs_observed_flag"] = usgs_observed.astype(int)
        group["usgs_stale_flag"] = (
            group["usgs_source_age_years"].fillna(999) > 3
        ).astype(int)
        for column in USGS_FEATURES:
            group[f"{column}_missing_flag"] = group[column].isna().astype(int)
            if column == "usgs_source_age_years":
                group[column] = group[column].fillna(99.0)
            else:
                group[column] = group[column].ffill().fillna(group[column].median())
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "hs_code"])


def add_target_quality(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    volume_score = data["volume_ratio_to_prior_12m_median"].fillna(1.0).clip(0.0, 1.0)
    anomaly_penalty = np.where(data["partner_anomaly_count"].fillna(0).gt(0), 0.20, 0.0)
    estimated_penalty = np.where(data["world_weight_estimated_by_un_flag"].eq(1), 0.20, 0.0)
    outlier_penalty = np.where(data["price_shock_history_flag"].eq(1), 0.20, 0.0)
    low_volume_penalty = np.where(data["low_volume_history_flag"].eq(1), 0.25, 0.0)
    data["target_reliability_score"] = np.clip(
        0.40 + 0.60 * volume_score
        - anomaly_penalty
        - estimated_penalty
        - outlier_penalty
        - low_volume_penalty,
        0.10,
        1.00,
    )
    conditions = [
        data["target_reliability_score"] < 0.50,
        data["target_reliability_score"] < 0.80,
    ]
    data["target_quality_status"] = np.select(
        conditions, ["review", "usable_with_caution"], default="reliable"
    )
    data["training_sample_weight_v3"] = np.clip(
        data["training_sample_weight"] * data["target_reliability_score"], 0.10, 2.0
    )
    data["supplier_concentration_change_3m"] = data.groupby("hs_code")[
        "supplier_hhi_weight"
    ].diff(3)
    data["top_supplier_share_change_3m"] = data.groupby("hs_code")[
        "top_supplier_share_weight"
    ].diff(3)
    return data


def main() -> None:
    missing_inputs = [path for path in (IMPROVED_V2_PATH, FULL_SPEC_MATRIX_PATH) if not path.exists()]
    if missing_inputs:
        missing_text = "\n".join(f"- {path}" for path in missing_inputs)
        raise FileNotFoundError(f"Production V3 raw input snapshot(s) missing:\n{missing_text}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (IMPROVED_V2_PATH, FULL_SPEC_MATRIX_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    core = pd.read_csv(IMPROVED_V2_PATH)
    core["date"] = pd.to_datetime(core["date"], errors="raise")
    external = load_external_features()
    data = core.merge(
        external,
        on=["date", "hs_code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_external"),
    )
    data = add_external_availability(data)
    data = add_target_quality(data)
    data["feature_set_id"] = "titanium_model_a_production_v3"

    numeric = data.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("V3 training data contains infinite numeric values.")
    if data.duplicated(["date", "hs_code"]).any():
        raise ValueError("V3 training data contains duplicate HS/month rows.")

    output = data.copy()
    for column in output.select_dtypes(include=["datetime", "datetimetz"]).columns:
        output[column] = output[column].dt.strftime("%Y-%m-%d")
    output.to_csv(OUTPUT_PATH, index=False, float_format="%.10g")
    shutil.copy2(OUTPUT_PATH, TRAINING_SNAPSHOT_PATH)

    audit = pd.DataFrame(
        [
            {
                "source": "UN Comtrade UK monthly imports",
                "role": "target, quantities and supplier mix",
                "first_month": data["date"].min().strftime("%Y-%m"),
                "last_month": data["date"].max().strftime("%Y-%m"),
                "status": "available",
            },
            {
                "source": "World Bank Pink Sheet local workbook",
                "role": "aluminium and nickel monthly changes",
                "first_month": data.loc[data["world_bank_observed_flag"].eq(1), "date"].min().strftime("%Y-%m"),
                "last_month": data.loc[data["world_bank_observed_flag"].eq(1), "date"].max().strftime("%Y-%m"),
                "status": "stale after last observed month; official refresh download blocked",
            },
            {
                "source": "USGS Data Series 140 titanium metal",
                "role": "annual titanium sponge price proxy",
                "first_month": "as-of joined",
                "last_month": "2020 source year",
                "status": "official series available locally but stale",
            },
            {
                "source": "Bank of England XUMAUSS",
                "role": "USD to GBP conversion",
                "first_month": data["date"].min().strftime("%Y-%m"),
                "last_month": data["date"].max().strftime("%Y-%m"),
                "status": "available",
            },
        ]
    )
    audit.to_csv(SOURCE_AUDIT_PATH, index=False)

    quality_columns = [
        "date",
        "hs_code",
        "import_unit_value_gbp_per_kg",
        "net_weight_kg",
        "volume_ratio_to_prior_12m_median",
        "partner_count",
        "supplier_hhi_weight",
        "top_supplier_name",
        "top_supplier_share_weight",
        "partner_anomaly_count",
        "world_weight_estimated_by_un_flag",
        "price_shock_history_flag",
        "target_reliability_score",
        "target_quality_status",
        "world_bank_source_age_months",
        "world_bank_stale_flag",
        "usgs_source_age_years",
        "usgs_stale_flag",
    ]
    data[quality_columns].to_csv(QUALITY_PATH, index=False)

    print("Titanium Model A production V3 data prepared")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"World Bank last observed: {audit.loc[1, 'last_month']}")
    print(f"Review-quality months: {int(data['target_quality_status'].eq('review').sum())}")
    print(f"Prepared output: {OUTPUT_PATH}")
    print(f"Training snapshot: {TRAINING_SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
