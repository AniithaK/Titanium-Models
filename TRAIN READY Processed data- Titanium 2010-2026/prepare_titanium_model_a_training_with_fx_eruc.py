from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent

TRADE_PATH = HERE / "titanium_model_a_trade_core_2010_2026.csv"
FX_PATH = (
    PROJECT_DIR
    / "Titanium UK monthly imports data 2010-2026"
    / "Bank of Englad database GBP to USD"
    / "Bank of England  Database.csv"
)
PPI_PATH = (
    PROJECT_DIR
    / "Titanium UK monthly imports data 2010-2026"
    / "Inflation Data"
    / "ppi.csv"
)
OUTPUT_PATH = HERE / "titanium_model_a_training_with_fx_eruc_2010_2026.csv"

HORIZONS = (3, 6, 12)
PRICE_LAGS = (1, 2, 3, 6, 12)
ERUC_LAGS = (1, 3, 6, 12)
FX_LAGS = (1, 3, 6, 12)
ERUC_SERIES_ID = "ERUC"
ERUC_ASSUMED_PUBLICATION_LAG_DAYS = 31
FX_ASSUMED_PUBLICATION_LAG_DAYS = 7

# The default predictors follow Model A's trade-history structure and add ERUC.
# FX is retained for conversion and audit, not duplicated as a default predictor.
MODEL_FEATURE_COLUMNS = [
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


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def period_end(periods: pd.Series) -> pd.Series:
    return periods.dt.to_timestamp(how="end").dt.normalize()


def load_trade() -> pd.DataFrame:
    trade = pd.read_csv(TRADE_PATH)
    required = [
        "date",
        "period",
        "year",
        "month",
        "hs_code",
        "hs_product",
        "trade_data_available_date_assumed",
        "trade_value_usd",
        "net_weight_kg",
        "import_unit_value_usd_per_kg",
        "quantity_log_change_1m",
        "quantity_log_change_3m",
        "quantity_log_change_12m",
        "supplier_share_usa",
        "supplier_share_japan",
        "supplier_share_france",
        "supplier_share_russia",
        "supplier_share_china",
        "supplier_share_other",
        "eu_collection_change_flag",
    ]
    require_columns(trade, required, str(TRADE_PATH))

    trade["date"] = pd.to_datetime(trade["date"], errors="raise")
    trade["period_m"] = trade["date"].dt.to_period("M")
    trade["trade_data_available_date_assumed"] = pd.to_datetime(
        trade["trade_data_available_date_assumed"], errors="raise"
    )
    for horizon in HORIZONS:
        column = f"target_data_available_date_assumed_h{horizon}"
        trade[column] = pd.to_datetime(trade[column], errors="coerce")

    if trade.duplicated(["hs_code", "period_m"]).any():
        raise ValueError("Trade data contains duplicate HS-code/month rows.")
    if set(trade["hs_code"].astype(int)) != {810820, 810890}:
        raise ValueError("Expected only HS 810820 and 810890 in Model A trade data.")

    for hs_code, group in trade.groupby("hs_code"):
        observed = pd.PeriodIndex(group["period_m"].sort_values(), freq="M")
        expected = pd.period_range(observed.min(), observed.max(), freq="M")
        if not observed.equals(expected):
            missing = expected.difference(observed).astype(str).tolist()
            raise ValueError(f"HS {hs_code} has missing months: {missing}")

    return trade.sort_values(["hs_code", "date"]).reset_index(drop=True)


def load_fx() -> pd.DataFrame:
    fx = pd.read_csv(FX_PATH)
    if fx.shape[1] != 2:
        raise ValueError(f"Expected two columns in Bank of England file, got {fx.shape[1]}.")

    fx.columns = ["fx_date_raw", "usd_per_gbp_monthly_average"]
    fx["fx_date"] = pd.to_datetime(fx["fx_date_raw"], format="%d %b %y", errors="raise")
    fx["period_m"] = fx["fx_date"].dt.to_period("M")
    fx["usd_per_gbp_monthly_average"] = pd.to_numeric(
        fx["usd_per_gbp_monthly_average"], errors="raise"
    )
    if fx["period_m"].duplicated().any():
        raise ValueError("Bank of England data contains duplicate months.")
    if (fx["usd_per_gbp_monthly_average"] <= 0).any():
        raise ValueError("Bank of England exchange rates must be positive.")

    fx = fx.sort_values("period_m").reset_index(drop=True)
    fx["fx_period"] = fx["period_m"].astype(str)
    fx["fx_available_date_assumed"] = period_end(fx["period_m"]) + pd.Timedelta(
        days=FX_ASSUMED_PUBLICATION_LAG_DAYS
    )
    for lag in FX_LAGS:
        fx[f"fx_log_change_{lag}m"] = np.log(
            fx["usd_per_gbp_monthly_average"]
            / fx["usd_per_gbp_monthly_average"].shift(lag)
        )
    return fx.drop(columns=["fx_date_raw", "fx_date"])


def load_eruc() -> pd.DataFrame:
    with PPI_PATH.open(encoding="utf-8-sig", errors="strict", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 8 or rows[0][0].strip().lower() != "title":
        raise ValueError("Unexpected ONS PPI CSV structure.")
    try:
        eruc_column = rows[1].index(ERUC_SERIES_ID)
    except ValueError as exc:
        raise ValueError(f"ONS PPI file does not contain series {ERUC_SERIES_ID}.") from exc

    title = rows[0][eruc_column].strip()
    release_date_raw = rows[4][eruc_column].strip()
    release_date = pd.to_datetime(release_date_raw, format="%d-%m-%Y", errors="raise")
    extraction_date = pd.Timestamp(
        datetime.fromtimestamp(PPI_PATH.stat().st_mtime).date()
    )

    observations: list[tuple[str, float]] = []
    for row in rows[7:]:
        if len(row) <= eruc_column:
            continue
        period = row[0].strip()
        value = row[eruc_column].strip()
        if not re.fullmatch(r"\d{4} [A-Z]{3}", period) or not value:
            continue
        observations.append((period, float(value.replace(",", ""))))

    eruc = pd.DataFrame(observations, columns=["period_raw", "eruc_index_2015_100"])
    eruc["date"] = pd.to_datetime(eruc["period_raw"], format="%Y %b", errors="raise")
    eruc["period_m"] = eruc["date"].dt.to_period("M")
    eruc = eruc.sort_values("period_m").reset_index(drop=True)
    if eruc["period_m"].duplicated().any():
        raise ValueError("ERUC data contains duplicate months.")
    expected = pd.period_range(eruc["period_m"].min(), eruc["period_m"].max(), freq="M")
    observed = pd.PeriodIndex(eruc["period_m"], freq="M")
    if not observed.equals(expected):
        missing = expected.difference(observed).astype(str).tolist()
        raise ValueError(f"ERUC data has missing months: {missing}")
    if (eruc["eruc_index_2015_100"] <= 0).any():
        raise ValueError("ERUC index values must be positive.")

    eruc["eruc_series_id"] = ERUC_SERIES_ID
    eruc["eruc_title"] = title
    eruc["eruc_period"] = eruc["period_m"].astype(str)
    eruc["eruc_source_release_date"] = release_date
    eruc["eruc_source_extraction_date"] = extraction_date
    eruc["eruc_available_date_assumed"] = period_end(
        eruc["period_m"]
    ) + pd.Timedelta(days=ERUC_ASSUMED_PUBLICATION_LAG_DAYS)
    latest_period = eruc["period_m"].max()
    eruc["eruc_provisional_latest_two_months_flag"] = (
        eruc["period_m"] >= latest_period - 1
    ).astype(int)
    eruc["eruc_latest_12_months_revision_risk_flag"] = (
        eruc["period_m"] >= latest_period - 11
    ).astype(int)
    for lag in ERUC_LAGS:
        eruc[f"eruc_log_change_{lag}m"] = np.log(
            eruc["eruc_index_2015_100"]
            / eruc["eruc_index_2015_100"].shift(lag)
        )

    return eruc.drop(columns=["period_raw", "date"])


def add_gbp_features_and_targets(data: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in data.groupby("hs_code", sort=False):
        group = group.sort_values("date").copy()
        for lag in PRICE_LAGS:
            group[f"price_log_change_gbp_{lag}m"] = np.log(
                group["import_unit_value_gbp_per_kg"]
                / group["import_unit_value_gbp_per_kg"].shift(lag)
            )
        for horizon in HORIZONS:
            future_date = group["date"].shift(-horizon)
            future_price = group["import_unit_value_gbp_per_kg"].shift(-horizon)
            group[f"target_month_gbp_h{horizon}"] = future_date.dt.to_period("M").astype(
                "string"
            )
            group[f"target_unit_value_gbp_per_kg_h{horizon}"] = future_price
            group[f"target_log_change_gbp_per_kg_h{horizon}"] = np.log(
                future_price / group["import_unit_value_gbp_per_kg"]
            )
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["date", "hs_code"]
    ).reset_index(drop=True)


def build_training_data() -> pd.DataFrame:
    trade = load_trade()
    fx = load_fx()
    eruc = load_eruc()

    data = trade.merge(fx, on="period_m", how="left", validate="many_to_one")
    data = data.merge(eruc, on="period_m", how="left", validate="many_to_one")

    data["fx_known_by_forecast_date"] = (
        data["fx_available_date_assumed"] <= data["trade_data_available_date_assumed"]
    )
    data["eruc_known_by_forecast_date"] = (
        data["eruc_available_date_assumed"]
        <= data["trade_data_available_date_assumed"]
    )

    fx_columns = ["usd_per_gbp_monthly_average"] + [
        f"fx_log_change_{lag}m" for lag in FX_LAGS
    ]
    eruc_columns = ["eruc_index_2015_100"] + [
        f"eruc_log_change_{lag}m" for lag in ERUC_LAGS
    ]
    data.loc[~data["fx_known_by_forecast_date"], fx_columns] = np.nan
    data.loc[~data["eruc_known_by_forecast_date"], eruc_columns] = np.nan

    data["trade_value_gbp"] = data["trade_value_usd"] / data[
        "usd_per_gbp_monthly_average"
    ]
    data["import_unit_value_gbp_per_kg"] = data[
        "import_unit_value_usd_per_kg"
    ] / data["usd_per_gbp_monthly_average"]
    data["log_unit_value_gbp"] = np.log(data["import_unit_value_gbp_per_kg"])
    data = add_gbp_features_and_targets(data)

    require_columns(data, MODEL_FEATURE_COLUMNS, "Prepared training data")
    data["feature_set_id"] = "titanium_model_a_fx_eruc_v1"
    data["model_a_selected_feature_complete"] = data[
        MODEL_FEATURE_COLUMNS
    ].notna().all(axis=1)
    data["model_a_source_inputs_complete"] = data[
        [
            "usd_per_gbp_monthly_average",
            "eruc_index_2015_100",
            "import_unit_value_gbp_per_kg",
        ]
    ].notna().all(axis=1)

    for horizon in HORIZONS:
        trade_eligible = f"trade_core_eligible_h{horizon}"
        require_columns(data, [trade_eligible], "Prepared training data")
        data[f"model_a_training_eligible_h{horizon}"] = (
            data["model_a_selected_feature_complete"]
            & data["model_a_source_inputs_complete"]
            & data[trade_eligible].astype(bool)
            & data[f"target_log_change_gbp_per_kg_h{horizon}"].notna()
            & data[f"target_data_available_date_assumed_h{horizon}"].notna()
        )

    numeric = data.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Prepared data contains infinite numeric values.")
    if data[["usd_per_gbp_monthly_average", "eruc_index_2015_100"]].isna().any().any():
        raise ValueError("FX or ERUC failed to join to one or more trade rows.")

    derived_unit_value = data["trade_value_gbp"] / data["net_weight_kg"]
    if not np.allclose(
        data["import_unit_value_gbp_per_kg"],
        derived_unit_value,
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True,
    ):
        raise ValueError("GBP/kg conversion does not reconcile to GBP trade value / kg.")

    priority = [
        "date",
        "period",
        "year",
        "month",
        "hs_code",
        "hs_product",
        "trade_data_available_date_assumed",
        "feature_set_id",
        "trade_value_usd",
        "trade_value_gbp",
        "net_weight_kg",
        "import_unit_value_usd_per_kg",
        "usd_per_gbp_monthly_average",
        "import_unit_value_gbp_per_kg",
        "log_unit_value_gbp",
        "fx_period",
        "fx_available_date_assumed",
        "fx_known_by_forecast_date",
        *[f"fx_log_change_{lag}m" for lag in FX_LAGS],
        "eruc_series_id",
        "eruc_title",
        "eruc_period",
        "eruc_index_2015_100",
        "eruc_available_date_assumed",
        "eruc_known_by_forecast_date",
        *[f"eruc_log_change_{lag}m" for lag in ERUC_LAGS],
        *MODEL_FEATURE_COLUMNS,
        "model_a_selected_feature_complete",
        "model_a_source_inputs_complete",
    ]
    for horizon in HORIZONS:
        priority.extend(
            [
                f"target_month_gbp_h{horizon}",
                f"target_unit_value_gbp_per_kg_h{horizon}",
                f"target_log_change_gbp_per_kg_h{horizon}",
                f"target_data_available_date_assumed_h{horizon}",
                f"model_a_training_eligible_h{horizon}",
            ]
        )
    priority = list(dict.fromkeys(priority))
    remaining = [column for column in data.columns if column not in priority and column != "period_m"]
    data = data[priority + remaining]
    return data


def format_dates_for_csv(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return output


def main() -> None:
    for path in (TRADE_PATH, FX_PATH, PPI_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    data = build_training_data()
    format_dates_for_csv(data).to_csv(OUTPUT_PATH, index=False, float_format="%.10g")

    print("Titanium Model A training data prepared")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Rows per HS code: {data['hs_code'].value_counts().sort_index().to_dict()}")
    print(f"Missing FX values: {int(data['usd_per_gbp_monthly_average'].isna().sum())}")
    print(f"Missing ERUC values: {int(data['eruc_index_2015_100'].isna().sum())}")
    print(
        "Feature-complete rows: "
        f"{int(data['model_a_selected_feature_complete'].sum())}/{len(data)}"
    )
    for horizon in HORIZONS:
        column = f"model_a_training_eligible_h{horizon}"
        print(f"H{horizon} eligible rows: {int(data[column].sum())}")
    print("Selected model features:")
    for feature in MODEL_FEATURE_COLUMNS:
        print(f"  - {feature}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
