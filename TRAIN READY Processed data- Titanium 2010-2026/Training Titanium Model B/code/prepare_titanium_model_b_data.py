from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT.parent
ENRICHED_PATH = SOURCE_DIR / "titanium_model_a_training_with_fx_eruc_2010_2026.csv"
TRADE_CORE_PATH = SOURCE_DIR / "titanium_model_a_trade_core_2010_2026.csv"
OUTPUT_PATH = ROOT / "data" / "titanium_model_b_country_month_training_2010_2026.csv"
THRESHOLD_PATH = ROOT / "data" / "model_b_fixed_class_thresholds.json"

COUNTRIES = ["usa", "japan", "france", "russia", "china", "other"]
COUNTRY_LABELS = {
    "usa": "USA",
    "japan": "Japan",
    "france": "France",
    "russia": "Russian Federation",
    "china": "China",
    "other": "Other countries",
}
HORIZONS = (3, 6, 12)
LAGS = (1, 3, 6, 12)
CLASSES = ["collapse", "fall", "stable", "rise"]
FORBIDDEN_TOKENS = ("usgs", "world_bank", "aluminium", "nickel")

FEATURE_COLUMNS = [
    "country_share",
    "log1p_country_quantity_kg",
    *[f"country_share_change_{lag}m" for lag in LAGS],
    *[f"country_quantity_log1p_change_{lag}m" for lag in LAGS],
    *[f"total_import_log_change_{lag}m" for lag in LAGS],
    *[f"titanium_price_log_change_{lag}m" for lag in LAGS],
    *[f"fx_log_change_{lag}m" for lag in LAGS],
    *[f"eruc_log_change_{lag}m" for lag in LAGS],
    "supplier_hhi_weight",
    "top_supplier_share_weight",
    "partner_count_total",
    "month_sin",
    "month_cos",
    "eu_collection_change_flag",
    "weight_reconstructed_any_flag",
    "world_weight_estimated_any_flag",
]


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def assert_permitted_sources(frame: pd.DataFrame, source: Path) -> None:
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Forbidden external-source columns in {source}: {forbidden}")


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = pd.read_csv(ENRICHED_PATH, low_memory=False)
    trade_core = pd.read_csv(TRADE_CORE_PATH, low_memory=False)
    assert_permitted_sources(enriched, ENRICHED_PATH)
    assert_permitted_sources(trade_core, TRADE_CORE_PATH)

    required = [
        "date",
        "hs_code",
        "trade_data_available_date_assumed",
        "trade_value_gbp",
        "net_weight_kg",
        "import_unit_value_gbp_per_kg",
        "usd_per_gbp_monthly_average",
        "fx_available_date_assumed",
        "eruc_index_2015_100",
        "eruc_available_date_assumed",
        "partner_count",
        "weight_reconstructed_flag",
        "world_weight_estimated_by_un_flag",
        "eu_collection_change_flag",
        *[f"fx_log_change_{lag}m" for lag in LAGS],
        *[f"eruc_log_change_{lag}m" for lag in LAGS],
        *[f"supplier_share_{country}" for country in COUNTRIES],
    ]
    require_columns(enriched, required, ENRICHED_PATH)
    require_columns(
        trade_core,
        ["date", "hs_code", "net_weight_kg", *[f"supplier_share_{c}" for c in COUNTRIES]],
        TRADE_CORE_PATH,
    )

    for frame in (enriched, trade_core):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["hs_code"] = frame["hs_code"].astype(int)
    for column in (
        "trade_data_available_date_assumed",
        "fx_available_date_assumed",
        "eruc_available_date_assumed",
    ):
        enriched[column] = pd.to_datetime(enriched[column], errors="raise")

    if set(enriched["hs_code"]) != {810820, 810890}:
        raise ValueError("Model B expects only HS 810820 and 810890; HS 810830 must remain excluded.")
    if enriched.duplicated(["date", "hs_code"]).any():
        raise ValueError("The enriched source contains duplicate HS/month rows.")
    if trade_core.duplicated(["date", "hs_code"]).any():
        raise ValueError("The trade-core source contains duplicate HS/month rows.")

    check_columns = ["net_weight_kg", *[f"supplier_share_{c}" for c in COUNTRIES]]
    check = enriched[["date", "hs_code", *check_columns]].merge(
        trade_core[["date", "hs_code", *check_columns]],
        on=["date", "hs_code"],
        how="outer",
        suffixes=("_enriched", "_core"),
        validate="one_to_one",
        indicator=True,
    )
    if not (check["_merge"] == "both").all():
        raise ValueError("The two source files do not contain identical HS/month keys.")
    for column in check_columns:
        if not np.allclose(
            check[f"{column}_enriched"],
            check[f"{column}_core"],
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(f"Source files disagree for {column}.")

    shares = enriched[[f"supplier_share_{c}" for c in COUNTRIES]].sum(axis=1)
    if not np.allclose(shares, 1.0, atol=1e-8):
        raise ValueError("Supplier shares do not sum to one.")
    return enriched.sort_values(["date", "hs_code"]), trade_core


def create_monthly_totals(enriched: pd.DataFrame) -> pd.DataFrame:
    work = enriched.copy()
    for country in COUNTRIES:
        work[f"country_weight_{country}_kg"] = (
            work[f"supplier_share_{country}"] * work["net_weight_kg"]
        )

    records: list[dict] = []
    for date, group in work.groupby("date", sort=True):
        if set(group["hs_code"]) != {810820, 810890}:
            raise ValueError(f"Month {date:%Y-%m} does not contain both non-scrap HS codes.")
        total_weight = float(group["net_weight_kg"].sum())
        total_value_gbp = float(group["trade_value_gbp"].sum())
        if total_weight <= 0 or total_value_gbp <= 0:
            raise ValueError(f"Non-positive aggregate trade value or weight in {date:%Y-%m}.")

        record = {
            "date": date,
            "year": int(date.year),
            "month": int(date.month),
            "period": date.strftime("%Y%m"),
            "total_import_weight_kg": total_weight,
            "total_trade_value_gbp": total_value_gbp,
            "titanium_import_unit_value_gbp_per_kg": total_value_gbp / total_weight,
            "usd_per_gbp_monthly_average": float(group["usd_per_gbp_monthly_average"].iloc[0]),
            "eruc_index_2015_100": float(group["eruc_index_2015_100"].iloc[0]),
            "partner_count_total": int(group["partner_count"].sum()),
            "weight_reconstructed_any_flag": int(group["weight_reconstructed_flag"].max()),
            "world_weight_estimated_any_flag": int(group["world_weight_estimated_by_un_flag"].max()),
            "eu_collection_change_flag": int(group["eu_collection_change_flag"].max()),
            "trade_data_available_date_assumed": group["trade_data_available_date_assumed"].max(),
            "fx_available_date_assumed": group["fx_available_date_assumed"].max(),
            "eruc_available_date_assumed": group["eruc_available_date_assumed"].max(),
        }
        record["origin_data_available_date_assumed"] = max(
            record["trade_data_available_date_assumed"],
            record["fx_available_date_assumed"],
            record["eruc_available_date_assumed"],
        )
        for lag in LAGS:
            fx_values = group[f"fx_log_change_{lag}m"].dropna().unique()
            eruc_values = group[f"eruc_log_change_{lag}m"].dropna().unique()
            record[f"fx_log_change_{lag}m"] = float(fx_values[0]) if len(fx_values) else np.nan
            record[f"eruc_log_change_{lag}m"] = float(eruc_values[0]) if len(eruc_values) else np.nan
        for country in COUNTRIES:
            record[f"country_weight_{country}_kg"] = float(group[f"country_weight_{country}_kg"].sum())
        records.append(record)

    monthly = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    observed = pd.PeriodIndex(monthly["date"].dt.to_period("M"), freq="M")
    expected = pd.period_range(observed.min(), observed.max(), freq="M")
    if not observed.equals(expected):
        raise ValueError(f"Missing aggregate months: {expected.difference(observed).astype(str).tolist()}")

    for country in COUNTRIES:
        monthly[f"supplier_share_{country}"] = (
            monthly[f"country_weight_{country}_kg"] / monthly["total_import_weight_kg"]
        )
    share_columns = [f"supplier_share_{country}" for country in COUNTRIES]
    if not np.allclose(monthly[share_columns].sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Combined supplier shares do not sum to one.")
    monthly["supplier_hhi_weight"] = (monthly[share_columns] ** 2).sum(axis=1)
    monthly["top_supplier_share_weight"] = monthly[share_columns].max(axis=1)
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12.0)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12.0)
    for lag in LAGS:
        monthly[f"total_import_log_change_{lag}m"] = np.log(
            monthly["total_import_weight_kg"] / monthly["total_import_weight_kg"].shift(lag)
        )
        monthly[f"titanium_price_log_change_{lag}m"] = np.log(
            monthly["titanium_import_unit_value_gbp_per_kg"]
            / monthly["titanium_import_unit_value_gbp_per_kg"].shift(lag)
        )
    return monthly


def to_country_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    panels = []
    base_columns = [
        column
        for column in monthly.columns
        if not column.startswith("country_weight_") and not column.startswith("supplier_share_")
    ]
    for country in COUNTRIES:
        panel = monthly[base_columns].copy()
        panel["country_key"] = country
        panel["country"] = COUNTRY_LABELS[country]
        panel["country_quantity_kg"] = monthly[f"country_weight_{country}_kg"]
        panel["country_share"] = monthly[f"supplier_share_{country}"]
        panel["log1p_country_quantity_kg"] = np.log1p(panel["country_quantity_kg"])
        panels.append(panel)
    data = pd.concat(panels, ignore_index=True).sort_values(["country_key", "date"]).reset_index(drop=True)

    parts = []
    for _, group in data.groupby("country_key", sort=False):
        group = group.sort_values("date").copy()
        for lag in LAGS:
            group[f"country_share_lag_{lag}m"] = group["country_share"].shift(lag)
            group[f"country_share_change_{lag}m"] = (
                group["country_share"] - group["country_share"].shift(lag)
            )
            group[f"country_quantity_log1p_change_{lag}m"] = (
                np.log1p(group["country_quantity_kg"])
                - np.log1p(group["country_quantity_kg"].shift(lag))
            )
        for horizon in HORIZONS:
            group[f"target_month_h{horizon}"] = group["date"].shift(-horizon)
            group[f"target_country_share_h{horizon}"] = group["country_share"].shift(-horizon)
            group[f"target_share_change_h{horizon}"] = (
                group[f"target_country_share_h{horizon}"] - group["country_share"]
            )
            group[f"target_data_available_date_assumed_h{horizon}"] = group[
                "origin_data_available_date_assumed"
            ].shift(-horizon)
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "country_key"]).reset_index(drop=True)


def class_from_change(values: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    valid = values.notna()
    result.loc[valid & (values < thresholds["collapse_upper"])] = "collapse"
    result.loc[
        valid
        & (values >= thresholds["collapse_upper"])
        & (values < thresholds["fall_upper"])
    ] = "fall"
    result.loc[
        valid
        & (values >= thresholds["fall_upper"])
        & (values <= thresholds["stable_upper"])
    ] = "stable"
    result.loc[valid & (values > thresholds["stable_upper"])] = "rise"
    return result


def add_targets_and_controls(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    threshold_sample = data.loc[
        data["target_month_h3"].notna()
        & (data["target_month_h3"] <= pd.Timestamp("2021-06-01")),
        "target_share_change_h3",
    ].dropna()
    if len(threshold_sample) < 100:
        raise ValueError("Insufficient pre-test H3 share changes to define classes.")
    thresholds = {
        "collapse_upper": float(threshold_sample.quantile(0.05)),
        "fall_upper": float(threshold_sample.quantile(0.33)),
        "stable_upper": float(threshold_sample.quantile(0.67)),
        "source_start": "2010-01",
        "source_end": "2021-06",
        "requested_start_unavailable": "2005-01",
        "sample_rows": int(len(threshold_sample)),
    }
    for horizon in HORIZONS:
        data[f"target_class_h{horizon}"] = class_from_change(
            data[f"target_share_change_h{horizon}"], thresholds
        )
    data["last_observed_3m_class"] = class_from_change(
        data["country_share_change_3m"], thresholds
    )
    data["evaluation_partition"] = np.select(
        [
            data["date"] <= pd.Timestamp("2021-06-01"),
            data["date"].between(pd.Timestamp("2021-07-01"), pd.Timestamp("2023-06-01")),
        ],
        ["model_selection_and_tuning", "disruption_test_2021_2023"],
        default="post_disruption_monitoring",
    )
    data["feature_complete"] = data[FEATURE_COLUMNS].notna().all(axis=1)
    for horizon in HORIZONS:
        data[f"model_b_training_eligible_h{horizon}"] = (
            data["feature_complete"]
            & data[f"target_class_h{horizon}"].notna()
            & data[f"target_data_available_date_assumed_h{horizon}"].notna()
        )
    return data, thresholds


def format_for_csv(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return output


def main() -> None:
    for path in (ENRICHED_PATH, TRADE_CORE_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    ROOT.joinpath("data").mkdir(parents=True, exist_ok=True)

    enriched, _ = load_sources()
    monthly = create_monthly_totals(enriched)
    data = to_country_panel(monthly)
    data, thresholds = add_targets_and_controls(data)

    if data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).isna().all(axis=1).any():
        raise ValueError("A country-month row has no usable numeric values.")
    if any(token in " ".join(data.columns).lower() for token in FORBIDDEN_TOKENS):
        raise ValueError("A forbidden external-source field was created unexpectedly.")

    format_for_csv(data).to_csv(OUTPUT_PATH, index=False, float_format="%.10g")
    THRESHOLD_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    print("Titanium Model B country-month data prepared")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Dates: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Countries: {data['country'].nunique()}")
    print(f"Class thresholds: {thresholds}")
    for horizon in HORIZONS:
        print(
            f"H{horizon} eligible rows: "
            f"{int(data[f'model_b_training_eligible_h{horizon}'].sum())}"
        )
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
