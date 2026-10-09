from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanium_model_a_production_v6_titanium_core_corrected_training.csv"
MANIFEST_PATH = ROOT / "results" / "latest" / "model_manifest.json"
FEATURE_IMPORTANCE_PATH = ROOT / "visualizations" / "selected_model_feature_importance.png"
OUTPUT_PATH = ROOT / "documentation" / "titanium_model_a_v6_training_data_record.pdf"
TEMP_DIR = ROOT.parents[3] / "tmp" / "pdfs" / "titanium_model_a_v6_data_record"

INK = colors.HexColor("#14343B")
TEAL = colors.HexColor("#087F83")
TEAL_SOFT = colors.HexColor("#DCEFED")
ORANGE = colors.HexColor("#D25D2B")
ORANGE_SOFT = colors.HexColor("#F7E5DA")
SAND = colors.HexColor("#F5F1E7")
MUTED = colors.HexColor("#60777B")
LINE = colors.HexColor("#D8D4C8")
GREEN = colors.HexColor("#2B7D58")


def load_manifest() -> dict:
    import json

    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def source_group(column: str) -> str:
    if column.startswith("fx_") or column == "usd_per_gbp_monthly_average":
        return "Bank of England"
    if column.startswith("eruc_"):
        return "ONS ERUC/PPI"
    if column in {
        "trade_value_gbp",
        "import_unit_value_gbp_per_kg",
        "log_unit_value_gbp",
    } or "_gbp_" in column:
        return "Derived: Comtrade + FX"
    if column.startswith("model_a_") or column.startswith("target_quality"):
        return "Derived control"
    if column.startswith("target_") or column.startswith("price_") or column.startswith("quantity_"):
        return "Derived from trade/FX"
    if column.startswith("supplier_") or column.startswith("top_supplier"):
        return "Derived from UN Comtrade"
    if column in {
        "month_sin",
        "month_cos",
        "eu_collection_change_flag",
        "latest_year_partial_flag",
        "no_change_baseline_log_change",
        "price_volatility_6m",
        "price_volatility_12m",
        "quantity_volatility_6m",
        "quantity_volatility_12m",
        "log_net_weight_kg",
        "volume_ratio_to_prior_12m_median",
        "low_volume_history_flag",
        "supplier_concentration_change_3m",
        "top_supplier_share_change_3m",
        "log_partner_count",
        "price_shock_history_flag",
    }:
        return "Derived feature"
    return "UN Comtrade"


EXACT_DESCRIPTIONS = {
    "date": "Calendar month represented by the row, stored as the first day of the month.",
    "period": "UN Comtrade monthly period identifier.",
    "year": "Calendar year extracted from the monthly period.",
    "month": "Calendar month number, 1 to 12.",
    "hs_code": "Titanium product category. V6 uses 810820 and 810890 separately.",
    "hs_product": "Readable product name for the HS code.",
    "source_product_description": "Commodity description supplied in the UN Comtrade export.",
    "classification_code": "Trade classification/version recorded by UN Comtrade.",
    "source_file": "Annual UN Comtrade CSV from which the monthly record originated.",
    "source_extraction_date": "Date the UN Comtrade source export was obtained.",
    "month_end_date": "Last calendar date of the observation month.",
    "trade_data_available_date_assumed": "Conservative month-end plus 42-day date used to prevent publication-timing leakage.",
    "evaluation_partition": "Time-ordered selection, disruption-test or post-disruption partition.",
    "feature_set_id": "Version label identifying the prepared feature specification.",
    "trade_value_usd": "UK monthly import value reported by UN Comtrade, in USD.",
    "trade_value_gbp": "Trade value converted to GBP using the monthly USD-per-GBP rate.",
    "world_net_weight_kg_original": "Net weight on the World aggregate record before reconstruction.",
    "partner_weight_sum_kg": "Sum of partner-country net weights for the same HS code and month.",
    "net_weight_kg": "Selected monthly import weight used in unit-value calculations.",
    "weight_source": "Indicates whether weight came from the World row or partner reconstruction.",
    "weight_reconstructed_flag": "1 when missing/zero World weight was replaced by summed partner weights.",
    "world_weight_estimated_by_un_flag": "UN Comtrade flag indicating that the World weight was estimated.",
    "partner_value_sum_usd": "Sum of partner-country import values for reconciliation.",
    "world_value_partner_difference_pct": "Percentage difference between World value and summed partner values.",
    "partner_count": "Number of partner countries with trade in the month.",
    "import_unit_value_usd_per_kg": "Trade value USD divided by selected net weight. This is an import unit value, not a quoted market price.",
    "import_unit_value_gbp_per_kg": "Primary observed price proxy: GBP trade value divided by net weight in kg.",
    "log_unit_value": "Natural logarithm of the USD/kg import unit value.",
    "log_unit_value_gbp": "Natural logarithm of the GBP/kg import unit value.",
    "unit_value_robust_z": "Robust outlier score for the monthly unit value.",
    "unit_value_outlier_flag": "Flags extreme unit values without silently replacing them.",
    "partner_anomaly_count": "Count of suspicious partner-level value/weight combinations in the month.",
    "partner_anomaly_max_weight_multiple": "Largest detected partner-weight anomaly multiple.",
    "data_quality_status": "Summary label for source and unit-value quality checks.",
    "usd_per_gbp_monthly_average": "Bank of England monthly average: US dollars per one pound sterling.",
    "fx_period": "Month matched to the Bank of England FX observation.",
    "fx_available_date_assumed": "Month-end plus seven days, used as the assumed FX availability date.",
    "fx_known_by_forecast_date": "True when the FX observation would have been available at the simulated forecast date.",
    "eruc_series_id": "ONS time-series identifier ERUC.",
    "eruc_title": "ONS description of the broad basic-metals import-price index.",
    "eruc_period": "Month of the ERUC/PPI observation.",
    "eruc_index_2015_100": "ONS import producer-price index for C24 basic metals, with 2015=100.",
    "eruc_source_release_date": "Release date recorded in the downloaded ONS source file.",
    "eruc_source_extraction_date": "Local extraction/download date for the ONS source.",
    "eruc_available_date_assumed": "Month-end plus 31 days, used to avoid publication-timing leakage.",
    "eruc_known_by_forecast_date": "True when the ERUC value would have been available at forecast time.",
    "eruc_provisional_latest_two_months_flag": "Marks the latest two ONS observations as potentially provisional.",
    "eruc_latest_12_months_revision_risk_flag": "Marks recent ONS observations that may be revised.",
    "supplier_share_sum_check": "Check that supplier weight shares sum to approximately one.",
    "supplier_hhi_weight": "Herfindahl-Hirschman concentration index of supplier weight shares.",
    "top_supplier_name": "Partner country with the largest monthly import-weight share.",
    "top_supplier_share_weight": "Weight share supplied by the largest partner country.",
    "month_sin": "Sine encoding of calendar month for smooth annual seasonality.",
    "month_cos": "Cosine encoding of calendar month for smooth annual seasonality.",
    "eu_collection_change_flag": "Indicator for the January 2021 UK/EU trade-data collection change.",
    "latest_year_partial_flag": "Indicates that the latest calendar year is incomplete.",
    "no_change_baseline_log_change": "Zero log-change benchmark used by the no-change baseline.",
    "log_net_weight_kg": "Natural logarithm of monthly import weight, reducing scale skew.",
    "volume_ratio_to_prior_12m_median": "Current import weight divided by the previous 12-month median.",
    "low_volume_history_flag": "Flags months with unusually low volume and less reliable unit values.",
    "supplier_concentration_change_3m": "Three-month change in supplier concentration.",
    "top_supplier_share_change_3m": "Three-month change in the largest supplier's share.",
    "log_partner_count": "Natural logarithm of active supplier-country count.",
    "price_shock_history_flag": "Flags a recent price shock in the observable history.",
    "model_a_selected_feature_complete": "True when every selected predictor is populated.",
    "model_a_source_inputs_complete": "True when required trade, FX and ERUC source inputs are present.",
}


def describe_column(column: str) -> str:
    if column in EXACT_DESCRIPTIONS:
        return EXACT_DESCRIPTIONS[column]
    match = re.fullmatch(r"price_log_change(?:_gbp)?_(\d+)m", column)
    if match:
        currency = "GBP/kg" if "_gbp_" in column else "USD/kg"
        return f"Log change in {currency} import unit value versus {match.group(1)} month(s) earlier."
    match = re.fullmatch(r"quantity_log_change_(\d+)m", column)
    if match:
        return f"Log change in imported net weight versus {match.group(1)} month(s) earlier."
    match = re.fullmatch(r"fx_log_change_(\d+)m", column)
    if match:
        return f"Log change in USD per GBP over {match.group(1)} month(s); captures currency movement."
    match = re.fullmatch(r"eruc_log_change_(\d+)m", column)
    if match:
        return f"Log change in the ONS ERUC basic-metals import-price index over {match.group(1)} month(s)."
    match = re.fullmatch(r"supplier_share_(.+)", column)
    if match:
        supplier = match.group(1).replace("_", " ").title()
        return f"Share of monthly import weight supplied by {supplier}."
    match = re.fullmatch(r"(price|quantity|fx|eruc)_volatility_(\d+)m", column)
    if match:
        return f"Rolling {match.group(2)}-month volatility of {match.group(1)} changes."
    match = re.fullmatch(r"target_month(?:_gbp)?_h(3|6|12)", column)
    if match:
        return f"Future target month {match.group(1)} months after the forecast origin. Audit field only."
    match = re.fullmatch(r"target_unit_value_(usd|gbp)_per_kg_h(3|6|12)", column)
    if match:
        return f"Observed future {match.group(1).upper()}/kg unit value at H{match.group(2)}. Target/audit field, never an input."
    match = re.fullmatch(r"target_log_change_(usd|gbp)_per_kg_h(3|6|12)", column)
    if match:
        return f"Direct H{match.group(2)} training label: log future/current {match.group(1).upper()}/kg unit value. Never an input."
    match = re.fullmatch(r"target_data_available_date_assumed_h(3|6|12)", column)
    if match:
        return f"Assumed publication date of the H{match.group(1)} outcome, used to enforce target availability."
    match = re.fullmatch(r"target_quality_flag_h(3|6|12)", column)
    if match:
        return f"Quality status for the H{match.group(1)} target observation."
    match = re.fullmatch(r"trade_core_eligible_h(3|6|12)", column)
    if match:
        return f"Trade-core eligibility flag for direct H{match.group(1)} modelling."
    match = re.fullmatch(r"model_a_training_eligible_h(3|6|12)", column)
    if match:
        return f"Final V6 training eligibility flag for horizon H{match.group(1)}."
    return "Supporting provenance, preparation, quality-control or audit field retained in the V6 snapshot."


def usage_role(column: str, feature_set: set[str]) -> str:
    if column in feature_set:
        return "Model input"
    if column.startswith("target_") and (
        "log_change" in column or "unit_value" in column or "month" in column
    ):
        return "Target / audit only"
    if column.startswith("model_a_training_eligible") or column.startswith("trade_core_eligible"):
        return "Row-selection control"
    if column in {
        "hs_code",
        "hs_product",
        "date",
        "period",
        "evaluation_partition",
        "trade_data_available_date_assumed",
    }:
        return "Identity / split control"
    if "flag" in column or "status" in column or "check" in column or "known_by" in column:
        return "Quality / timing control"
    return "Audit / supporting field"


def feature_category(feature: str) -> str:
    if feature.startswith("price_"):
        return "Price history"
    if feature.startswith("quantity_") or feature in {
        "log_net_weight_kg",
        "volume_ratio_to_prior_12m_median",
        "low_volume_history_flag",
    }:
        return "Import volume"
    if feature.startswith("supplier_") or feature.startswith("top_supplier") or feature == "log_partner_count":
        return "Supplier structure"
    if feature.startswith("fx_"):
        return "Exchange rate"
    if feature.startswith("eruc_"):
        return "Basic-metals inflation"
    if feature.startswith("month_"):
        return "Seasonality"
    return "Quality / regime"


def feature_use(feature: str) -> str:
    category = feature_category(feature)
    uses = {
        "Price history": "Learns momentum, mean reversion and recent instability in the GBP/kg target.",
        "Import volume": "Captures demand/supply intensity and warns when low volume makes unit values unstable.",
        "Supplier structure": "Represents sourcing mix, concentration and shifts among major supplying countries.",
        "Exchange rate": "Separates sterling movement from changes in the underlying USD-denominated trade value.",
        "Basic-metals inflation": "Provides a broad imported-basic-metals cost signal; it is not titanium-specific.",
        "Seasonality": "Allows recurring month-of-year effects without treating December and January as far apart.",
        "Quality / regime": "Marks shocks, estimated weights and structural collection changes that affect reliability.",
    }
    return uses[category]


class NumberedDocTemplate(BaseDocTemplate):
    pass


def footer(canvas, doc):
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.5 * mm, "Titanium Model A V6 - Training Data Record")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="Times-Bold", fontSize=28, leading=31, textColor=INK, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11, leading=16, textColor=MUTED),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Times-Bold", fontSize=20, leading=23, textColor=INK, spaceBefore=4, spaceAfter=10),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=ORANGE, spaceBefore=8, spaceAfter=6, uppercase=True),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=13.2, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.3, leading=9.5, textColor=INK),
        "tiny": ParagraphStyle("Tiny", parent=base["BodyText"], fontName="Helvetica", fontSize=6.3, leading=7.7, textColor=INK),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=INK),
        "center": ParagraphStyle("Center", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12, textColor=INK, alignment=TA_CENTER),
    }


def p(text: object, style) -> Paragraph:
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def table(data, widths, header=True, font_size=7.5, row_backs=None, repeat=1):
    data = [list(row) for row in data]
    if header and data:
        header_style = ParagraphStyle(
            "TableHeader",
            fontName="Helvetica-Bold",
            fontSize=font_size,
            leading=font_size + 1.7,
            textColor=colors.white,
        )
        data[0] = [
            Paragraph(cell.getPlainText(), header_style)
            if isinstance(cell, Paragraph)
            else Paragraph(str(cell), header_style)
            for cell in data[0]
        ]
    result = LongTable(data, colWidths=widths, repeatRows=repeat if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F8F6EF")))
    if row_backs:
        for row, colour in row_backs.items():
            commands.append(("BACKGROUND", (0, row), (-1, row), colour))
    result.setStyle(TableStyle(commands))
    return result


def create_timeline_chart(data: pd.DataFrame) -> Path:
    output = TEMP_DIR / "data_timeline.png"
    start = pd.Timestamp(data["date"].min())
    end = pd.Timestamp(data["date"].max())
    fig, ax = plt.subplots(figsize=(11.2, 2.8), dpi=180)
    ax.axvspan(start, pd.Timestamp("2022-01-31"), color="#DCEFED", alpha=1)
    ax.axvspan(pd.Timestamp("2022-02-01"), pd.Timestamp("2023-12-31"), color="#F7E5DA", alpha=1)
    ax.axvspan(pd.Timestamp("2024-01-01"), end, color="#E8E9DF", alpha=1)
    ys = [3, 2, 1]
    labels = ["UN Comtrade", "Bank of England FX", "ONS ERUC/PPI"]
    colours = ["#087F83", "#D25D2B", "#55717A"]
    for y, label, colour in zip(ys, labels, colours):
        ax.plot([start, end], [y, y], lw=10, color=colour, solid_capstyle="butt")
        ax.text(start - pd.DateOffset(months=5), y, label, va="center", ha="right", fontsize=9, color="#14343B")
    ax.text(pd.Timestamp("2016-01-01"), 3.55, "Model selection / validation", ha="center", fontsize=9, weight="bold")
    ax.text(pd.Timestamp("2022-12-01"), 3.55, "Disruption test", ha="center", fontsize=9, weight="bold")
    ax.text(pd.Timestamp("2025-02-01"), 3.55, "Post-disruption", ha="center", fontsize=9, weight="bold")
    ax.set_xlim(start - pd.DateOffset(months=18), end + pd.DateOffset(months=5))
    ax.set_ylim(.4, 4)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D8D4C8")
    ax.tick_params(axis="x", colors="#60777B", labelsize=8)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def create_feature_group_chart(features: list[str]) -> Path:
    output = TEMP_DIR / "feature_groups.png"
    counts = Counter(feature_category(feature) for feature in features)
    order = [
        "Price history",
        "Import volume",
        "Supplier structure",
        "Exchange rate",
        "Basic-metals inflation",
        "Seasonality",
        "Quality / regime",
    ]
    values = [counts[key] for key in order]
    fig, ax = plt.subplots(figsize=(10.7, 3.4), dpi=180)
    bars = ax.barh(order[::-1], values[::-1], color=["#14343B", "#087F83", "#3A9898", "#D25D2B", "#DD825C", "#8CA9A5", "#A9A99D"][::-1])
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + .15, bar.get_y() + bar.get_height()/2, str(value), va="center", fontsize=9, weight="bold", color="#14343B")
    ax.set_xlim(0, max(values) + 2)
    ax.set_xlabel("Number of selected V6 model inputs", fontsize=9, color="#60777B")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D8D4C8")
    ax.tick_params(axis="y", length=0, labelsize=9, colors="#14343B")
    ax.tick_params(axis="x", labelsize=8, colors="#60777B")
    ax.grid(axis="x", color="#ECE9DF", linewidth=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def build_pdf() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    manifest = load_manifest()
    features = manifest["features"]
    feature_set = set(features)
    missing_features = sorted(feature_set - set(data.columns))
    if missing_features:
        raise ValueError(f"Manifest features missing from data: {missing_features}")

    forbidden_tokens = ("usgs", "world_bank", "aluminium", "nickel")
    forbidden_columns = [c for c in data.columns if any(t in c.lower() for t in forbidden_tokens)]
    if forbidden_columns:
        raise ValueError(f"Excluded proxy columns unexpectedly remain: {forbidden_columns}")

    timeline_path = create_timeline_chart(data)
    feature_chart_path = create_feature_group_chart(features)
    st = styles()
    width, height = landscape(A4)
    doc = NumberedDocTemplate(
        str(OUTPUT_PATH),
        pagesize=(width, height),
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="Titanium Model A V6 Training Data Record",
        author="Supply Chain Efficiency - Model A",
        subject="Dataset lineage, feature use, targets and quality controls",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=footer)])
    story = []

    story += [Spacer(1, 11 * mm), p("DATA RECORD", st["h2"]), p("Titanium Model A V6", st["title"])]
    story += [p("Training dataset lineage, field purpose and use in prediction", st["subtitle"]), Spacer(1, 9 * mm)]
    summary = [
        [p("ROWS", st["small"]), p("COLUMNS", st["small"]), p("COVERAGE", st["small"]), p("PRODUCTS", st["small"])],
        [p(f"{len(data):,}", st["callout"]), p(f"{len(data.columns):,}", st["callout"]), p(f"{data['date'].min():%b %Y} to {data['date'].max():%b %Y}", st["callout"]), p("HS 810820 and 810890", st["callout"])],
    ]
    summary_table = Table(summary, colWidths=[42*mm, 42*mm, 65*mm, 75*mm], rowHeights=[9*mm, 15*mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), TEAL_SOFT), ("BACKGROUND", (0,1), (-1,1), colors.white),
        ("BOX", (0,0), (-1,-1), .6, LINE), ("INNERGRID", (0,0), (-1,-1), .4, LINE),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story += [summary_table, Spacer(1, 9 * mm)]
    story += [p("Purpose", st["h1"]), p(
        "This PDF records exactly what the selected V6 training CSV contains, where each data family originates, and how it supports the direct 3, 6 and 12-month GBP/kg forecasts. The file is a reproducible training snapshot, not a raw-source export. It combines source observations, engineered model inputs, future training labels, publication-timing controls and audit fields.", st["body"])]
    story += [p("Important interpretation", st["h2"]), p(
        "The observed price is an import unit value: UK import value divided by imported net weight. It reflects product mix, supplier mix, contract timing and data quality as well as underlying titanium-market conditions. It must not be described as a spot-market quotation.", st["body"])]
    story += [Spacer(1, 4 * mm), Table([[p("41", st["callout"]), p("Selected V6 model inputs", st["body"]), p("134", st["callout"]), p("Total columns including targets, provenance and controls", st["body"]) ]], colWidths=[18*mm, 75*mm, 18*mm, 105*mm], style=TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), ORANGE_SOFT), ("BOX", (0,0), (-1,-1), .5, colors.HexColor("#E4B79F")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))]

    story += [PageBreak(), p("1. Source lineage", st["h1"])]
    source_rows = [
        [p("Source", st["small"]), p("Supplied file(s)", st["small"]), p("Fields contributed", st["small"]), p("Use", st["small"])],
        [p("UN Comtrade", st["small"]), p("17 annual UK monthly-import CSV exports", st["small"]), p("USD trade value, net weight, partner countries, HS identity and source-quality metadata", st["small"]), p("Creates GBP/kg target, volume history, supplier mix and concentration features", st["small"])],
        [p("Bank of England", st["small"]), p("Bank of England Database.csv", st["small"]), p("Monthly average USD per GBP and derived FX changes", st["small"]), p("Converts USD trade value to GBP and captures sterling movement", st["small"])],
        [p("ONS", st["small"]), p("ppi.csv, ERUC series", st["small"]), p("C24 basic-metals import-price index, 2015=100, and lagged changes", st["small"]), p("Broad non-ferrous imported-cost signal; not titanium-specific", st["small"])],
        [p("Preparation pipeline", st["small"]), p("Deterministic derived fields", st["small"]), p("Lags, volatility, supplier structure, seasonality, quality flags and direct future labels", st["small"]), p("Transforms time-ordered source data into leak-controlled training examples", st["small"])],
    ]
    story += [table(source_rows, [33*mm, 52*mm, 83*mm, 77*mm], font_size=7.2), Spacer(1, 5*mm)]
    story += [Image(str(timeline_path), width=245*mm, height=58*mm), Spacer(1, 2*mm)]
    story += [p("Time boundary", st["h2"]), p(
        "Each observation is one month and one HS code. Model selection and blocked validation end in January 2022. February 2022 through December 2023 is the frozen disruption test. January 2024 onward is post-disruption monitoring. Availability dates prevent source values or labels from entering a simulated forecast before they would have been published.", st["body"])]
    story += [p("Explicit exclusions", st["h2"]), p(
        "The V6 snapshot contains no USGS titanium-sponge proxy and no World Bank aluminium or nickel price features. HS 810830 titanium waste and scrap is also excluded. These exclusions keep the selected dataset focused on titanium trade, FX and the documented broad ERUC/PPI proxy.", st["body"])]

    story += [PageBreak(), p("2. From source records to GBP/kg", st["h1"])]
    formulas = [
        [p("Step", st["small"]), p("Calculation", st["small"]), p("Why it is needed", st["small"])],
        [p("Weight selection", st["small"]), p("Use positive World net weight; otherwise sum partner-country weights and set reconstruction flag", st["small"]), p("Preserves months where the aggregate World weight is missing or zero", st["small"])],
        [p("GBP value", st["small"]), p("trade_value_gbp = trade_value_usd / usd_per_gbp_monthly_average", st["small"]), p("Places import expenditure in the model's required currency", st["small"])],
        [p("GBP/kg unit value", st["small"]), p("import_unit_value_gbp_per_kg = trade_value_gbp / net_weight_kg", st["small"]), p("Creates the observed monthly target-price proxy", st["small"])],
        [p("Direct H3/H6/H12 label", st["small"]), p("ln(price at t+h) - ln(price at t)", st["small"]), p("Learns each horizon directly rather than recursively chaining H1 predictions", st["small"])],
    ]
    story += [table(formulas, [43*mm, 104*mm, 98*mm], font_size=7.5), Spacer(1, 6*mm)]
    story += [p("Two separate product series", st["h2"])]
    product_counts = data.groupby(["hs_code", "hs_product"], dropna=False).size().reset_index(name="rows")
    product_rows = [[p("HS code", st["small"]), p("Product", st["small"]), p("Rows", st["small"]), p("Reason for separation", st["small"])]]
    for _, row in product_counts.iterrows():
        product_rows.append([p(str(int(row["hs_code"])), st["small"]), p(row["hs_product"], st["small"]), p(str(int(row["rows"])), st["small"]), p("Different product form, price level and supplier behaviour", st["small"])])
    story += [table(product_rows, [30*mm, 90*mm, 25*mm, 100*mm], font_size=7.5), Spacer(1, 5*mm)]
    story += [p("Data-quality handling", st["h2"]), p(
        "Flagged observations are retained for traceability rather than silently overwritten. The training pipeline uses explicit weight-reconstruction, estimated-weight, low-volume, price-shock and structural-change indicators. Eligibility flags and assumed publication dates control whether a row can train a given horizon.", st["body"])]

    story += [PageBreak(), p("3. The 41 predictors used by V6", st["h1"]), p(
        "Only the following manifest-recorded columns are supplied to the selected models. Future targets, future prices, filenames and evaluation labels are not model inputs.", st["body"])]
    story += [Image(str(feature_chart_path), width=238*mm, height=73*mm), Spacer(1, 4*mm)]
    feature_rows = [[p("Feature", st["tiny"]), p("Group", st["tiny"]), p("Source", st["tiny"]), p("Training/prediction role", st["tiny"])]]
    for feature in features:
        feature_rows.append([
            p(feature, st["tiny"]),
            p(feature_category(feature), st["tiny"]),
            p(source_group(feature), st["tiny"]),
            p(feature_use(feature), st["tiny"]),
        ])
    story += [table(feature_rows, [56*mm, 34*mm, 41*mm, 114*mm], font_size=6.4)]

    story += [PageBreak(), p("4. Targets, controls and leakage boundary", st["h1"])]
    target_rows = [[p("Field family", st["small"]), p("Purpose", st["small"]), p("Allowed use", st["small"])],
        [p("target_log_change_gbp_per_kg_h3/h6/h12", st["small"]), p("Direct future log-change labels for each horizon", st["small"]), p("Training outcome only; never a predictor", st["small"])],
        [p("target_unit_value_gbp_per_kg_h3/h6/h12", st["small"]), p("Observed future GBP/kg values for audit and evaluation", st["small"]), p("Target/audit only", st["small"])],
        [p("target_month_gbp_h3/h6/h12", st["small"]), p("Identifies the future month associated with each label", st["small"]), p("Audit and reporting", st["small"])],
        [p("target_data_available_date_assumed_h*", st["small"]), p("Models when the outcome would become observable", st["small"]), p("Training-window eligibility", st["small"])],
        [p("model_a_training_eligible_h*", st["small"]), p("Combines feature completeness, source completeness, quality and target availability", st["small"]), p("Row selection, not a predictive signal", st["small"])],
        [p("evaluation_partition", st["small"]), p("Separates selection, stress-test and monitoring periods", st["small"]), p("Evaluation control only", st["small"])],
    ]
    story += [table(target_rows, [65*mm, 105*mm, 75*mm], font_size=7.4), Spacer(1, 6*mm)]
    story += [p("Leakage controls", st["h2"])]
    leakage = [
        "Source publication dates are compared with the simulated forecast date.",
        "A training row is used only after its H3, H6 or H12 outcome is assumed available.",
        "Targets and future prices remain in the CSV for auditability but are excluded from the manifest feature list.",
        "Validation is blocked in time; the 2022-2023 disruption period does not select the model family.",
        "Each HS product and horizon is modelled separately, avoiding price-level averaging across unlike titanium forms.",
    ]
    for item in leakage:
        story.append(p(f"- {item}", st["body"]))
    story += [p("Availability at the end of the dataset", st["h2"]), p(
        "The latest rows naturally have missing future H3, H6 or H12 labels because those target months have not yet occurred or were not yet assumed published. This is expected and is not source-data missingness. Such rows can be forecast origins but cannot yet be completed training examples for the affected horizon.", st["body"])]

    story += [PageBreak(), p("5. How the data supports prediction", st["h1"])]
    role_rows = [[p("Question", st["small"]), p("Data used", st["small"]), p("What the model can learn", st["small"])],
        [p("Is price momentum changing?", st["small"]), p("Price log changes and rolling volatility", st["small"]), p("Recent direction, persistence, mean reversion and instability", st["small"])],
        [p("Is the unit value based on normal trade volume?", st["small"]), p("Net weight, quantity changes, volume ratio and low-volume flag", st["small"]), p("Whether a price movement coincides with a demand/supply shift or thin-volume noise", st["small"])],
        [p("Has the sourcing mix changed?", st["small"]), p("Country shares, HHI, top-supplier share and their changes", st["small"]), p("Supplier concentration, substitution and exposure to dominant partners", st["small"])],
        [p("Did sterling move?", st["small"]), p("Bank of England FX changes and volatility", st["small"]), p("Currency contribution to GBP-denominated import unit values", st["small"])],
        [p("Are imported basic-metal costs shifting?", st["small"]), p("ERUC/PPI changes and volatility", st["small"]), p("Broad non-ferrous import-cost pressure; treated as a proxy, not titanium causality", st["small"])],
        [p("Are there recurring or structural effects?", st["small"]), p("Month sine/cosine, EU collection-change and quality flags", st["small"]), p("Seasonality, regime changes and data-reliability conditions", st["small"])],
    ]
    story += [table(role_rows, [55*mm, 80*mm, 110*mm], font_size=7.5), Spacer(1, 6*mm)]
    if FEATURE_IMPORTANCE_PATH.exists():
        story += [p("Selected-model feature influence", st["h2"]), Image(str(FEATURE_IMPORTANCE_PATH), width=232*mm, height=91*mm)]
    story += [p("What the dataset does not establish", st["h2"]), p(
        "Feature importance and predictive association do not establish causality. ERUC is broad, UN Comtrade unit values reflect product mix, and supplier shares may proxy changes in specification or contracting. The resulting forecast should support procurement planning together with domain review, not replace it.", st["body"])]

    story += [Spacer(1, 5*mm), p("6. Dataset controls and reproducibility", st["h1"])]
    feature_missing = int(data[features].isna().any(axis=1).sum())
    quality_rows = [[p("Control", st["small"]), p("Observed in V6 snapshot", st["small"]), p("Interpretation", st["small"])],
        [p("Rows / columns", st["small"]), p(f"{len(data):,} / {len(data.columns):,}", st["small"]), p("One row per HS code and month", st["small"])],
        [p("Date range", st["small"]), p(f"{data['date'].min():%Y-%m} to {data['date'].max():%Y-%m}", st["small"]), p("Continuous monthly modelling span", st["small"])],
        [p("Selected features", st["small"]), p(str(len(features)), st["small"]), p("Must match model_manifest.json", st["small"])],
        [p("Rows with at least one unavailable selected feature", st["small"]), p(f"{feature_missing:,}", st["small"]), p("Expected mainly in early lag windows; eligibility controls fitting", st["small"])],
        [p("Excluded proxy columns detected", st["small"]), p("0", st["small"]), p("No USGS or World Bank cross-metal inputs remain", st["small"])],
    ]
    story += [table(quality_rows, [78*mm, 52*mm, 115*mm], font_size=7.5), Spacer(1, 5*mm)]
    story += [p("Reproduction files", st["h2"]), p(
        "The training snapshot is consumed by code/train_model_a_production_v6_titanium_core_corrected.py. Selected feature names, model files, horizons and calibration metadata are recorded in results/latest/model_manifest.json. RUN_TRAINING_V6.cmd reproduces training and evaluation from the isolated V6 package.", st["body"])]
    story += [p("Known limitations", st["h2"])]
    for item in [
        "ERUC/PPI is a broad C24 basic-metals import-price proxy, not a titanium benchmark.",
        "The assumed 42-day trade, 7-day FX and 31-day ERUC publication lags should be validated against production release calendars.",
        "Import unit values may change because of product grade and contract mix even if the underlying market is unchanged.",
        "The 2024-2025 assessment is useful but not pristine because this dataset was previously explored.",
        "A genuinely prospective holdout is still required before production approval.",
    ]:
        story.append(p(f"- {item}", st["body"]))

    story += [PageBreak(), p("Appendix A. Full 134-column data dictionary", st["h1"]), p(
        "Every column in the delivered CSV is listed below. 'Model input' means the field appears in the saved V6 manifest. Other columns remain for conversion, row selection, evaluation, quality control or auditability.", st["body"])]
    dictionary_rows = [[p("Column", st["tiny"]), p("Source", st["tiny"]), p("Role", st["tiny"]), p("Definition", st["tiny"]), p("Missing", st["tiny"])]]
    for column in data.columns:
        missing = int(data[column].isna().sum())
        missing_text = f"{missing} ({missing/len(data):.1%})"
        dictionary_rows.append([
            p(column, st["tiny"]),
            p(source_group(column), st["tiny"]),
            p(usage_role(column, feature_set), st["tiny"]),
            p(describe_column(column), st["tiny"]),
            p(missing_text, st["tiny"]),
        ])
    story += [table(dictionary_rows, [51*mm, 34*mm, 34*mm, 111*mm, 25*mm], font_size=6.1)]

    story += [PageBreak(), p("Appendix B. Source file record", st["h1"])]
    source_record = [
        [p("Record", st["small"]), p("Location", st["small"])],
        [p("Dataset", st["small"]), p(str(DATA_PATH), st["small"])],
        [p("UN Comtrade source location", st["small"]), p(str(ROOT.parents[2] / "Titanium UK monthly imports data 2010-2026" / "UN ComTrade"), st["small"])],
        [p("Bank of England source", st["small"]), p(str(ROOT.parents[2] / "Titanium UK monthly imports data 2010-2026" / "Bank of Englad database GBP to USD" / "Bank of England  Database.csv"), st["small"])],
        [p("ONS source", st["small"]), p(str(ROOT.parents[2] / "Titanium UK monthly imports data 2010-2026" / "Inflation Data" / "ppi.csv"), st["small"])],
        [p("Model manifest", st["small"]), p(str(MANIFEST_PATH), st["small"])],
        [p("Training code", st["small"]), p(str(ROOT / "code" / "train_model_a_production_v6_titanium_core_corrected.py"), st["small"])],
    ]
    story += [table(source_record, [55*mm, 190*mm], font_size=7.1), Spacer(1, 7*mm)]
    story += [p("Source institutions", st["h2"]), p(
        "UN Comtrade supplies UK import value, weight and partner-country records. The Bank of England supplies the monthly USD-per-GBP conversion series. The UK Office for National Statistics supplies ERUC, the C24 basic-metals import-price index. All model-ready lags, volatility measures, concentration measures, targets and controls are derived locally by the documented preparation and training pipelines.", st["body"])]
    story += [p("Record status", st["h2"]), p(
        "This document describes the selected V6 research package and the exact CSV present when the report was generated. Any future refresh should regenerate this PDF so row counts, date coverage, missingness and the manifest feature set remain synchronized.", st["body"])]

    doc.build(story)
    print(f"Created: {OUTPUT_PATH}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Selected model inputs: {len(features)}")


if __name__ == "__main__":
    build_pdf()
