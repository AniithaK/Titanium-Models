from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(
    r"C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM"
    r"\TRAIN READY Processed data- Titanium 2010-2026"
    r"\Titanium Model A and B Performance Review\RESULT ANALYSIS"
)
OUTPUT = ROOT / "Titanium_Model_A_B_Result_Analysis_Report.pdf"

NAVY = colors.HexColor("#12353B")
TEAL = colors.HexColor("#087D80")
ORANGE = colors.HexColor("#CC5D2B")
PALE_TEAL = colors.HexColor("#E6F1EF")
PALE_ORANGE = colors.HexColor("#F8E5D8")
SAND = colors.HexColor("#F3EBDD")
INK = colors.HexColor("#17363C")
MUTED = colors.HexColor("#617579")
GRID = colors.HexColor("#D8D0C2")
GREEN = colors.HexColor("#337653")
RED = colors.HexColor("#A43B32")


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8))
styles.add(ParagraphStyle(name="ReportSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=12, leading=17, textColor=MUTED, spaceAfter=16))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, spaceBefore=4, spaceAfter=9))
styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=TEAL, spaceBefore=7, spaceAfter=5))
styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="Note", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=5))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=NAVY, backColor=PALE_TEAL, borderColor=TEAL, borderWidth=0.7, borderPadding=8, spaceAfter=10))
styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.2, leading=9, textColor=INK))
styles.add(ParagraphStyle(name="TableHeader", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=9, textColor=colors.white, alignment=TA_CENTER))


def p(text: object, style: str = "TableCell") -> Paragraph:
    return Paragraph(str(text), styles[style])


def fmt_pct(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{decimals}f}%"


def fmt_num(value: object, decimals: int = 3) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def fmt_gbp(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return f"GBP {float(value):,.2f}/kg"


def make_table(rows: list[list[object]], widths: list[float], header=True, font_size=7.2) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        style = "TableHeader" if header and row_index == 0 else "TableCell"
        converted.append([p(value, style) for value in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F8F6F0")))
    table.setStyle(TableStyle(commands))
    return table


def scaled_image(filename: str, width_mm: float, max_height_mm: float | None = None) -> Image:
    image = Image(str(ROOT / filename))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width_mm * mm
    image.drawHeight = width_mm * ratio * mm
    if max_height_mm and image.drawHeight > max_height_mm * mm:
        factor = (max_height_mm * mm) / image.drawHeight
        image.drawWidth *= factor
        image.drawHeight *= factor
    image.hAlign = "CENTER"
    return image


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    usable = frame.dropna(subset=[column, "rows"])
    return float((usable[column] * usable["rows"]).sum() / usable["rows"].sum())


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setStrokeColor(ORANGE)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8 * mm, "Titanium Models A and B performance review")
    canvas.drawRightString(width - 18 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


a_summary = pd.read_csv(ROOT / "model_a_performance_analysis.csv")
a_detail = pd.read_csv(ROOT / "model_a_ground_truth_vs_predictions_analysis.csv")
b_summary = pd.read_csv(ROOT / "model_b_performance_analysis.csv")
b_detail = pd.read_csv(ROOT / "model_b_ground_truth_vs_predictions_analysis.csv")
a_future = pd.read_csv(ROOT / "model_a_future_forecasts_2026_2027.csv")
b_future = pd.read_csv(ROOT / "model_b_future_forecasts_2026_2027.csv")

split_order = ["validation", "disruption_test", "post_disruption_test"]
split_label = {
    "validation": "Validation",
    "disruption_test": "Disruption test",
    "post_disruption_test": "Post-disruption test",
}

doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=landscape(A4),
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=19 * mm,
    bottomMargin=15 * mm,
    title="Titanium Models A and B Result Analysis",
    author="Supply Chain Efficiency",
)
story = []

story.extend(
    [
        Spacer(1, 18 * mm),
        Paragraph("Titanium Models A and B", styles["ReportTitle"]),
        Paragraph("Ground-truth comparison, probability review and future forecast analysis", styles["ReportSubtitle"]),
        Spacer(1, 5 * mm),
        Paragraph(
            "This report compares available historical observations with predictions for both titanium models. "
            "It also presents the genuine future forecasts produced from the June 2026 source cutoff. "
            "Future target months do not yet have ground truth and therefore cannot be scored today.",
            styles["Callout"],
        ),
        Spacer(1, 4 * mm),
        make_table(
            [
                ["Model", "Prediction task", "Outputs", "How performance is judged"],
                ["Model A", "UK titanium import unit price", "P05, P50 and P95 GBP/kg at H3, H6 and H12", "P50 deviation, MAE/RMSE, direction accuracy, interval coverage and interval score"],
                ["Model B", "Supplier-share movement", "Collapse, fall, stable or rise probabilities at H3, H6 and H12", "Accuracy, balanced accuracy, macro F1, Brier score, log loss and calibration"],
            ],
            [32 * mm, 55 * mm, 78 * mm, 88 * mm],
        ),
        Spacer(1, 7 * mm),
        Paragraph("Prepared from saved model evaluation outputs and future forecast files. Source data ends June 2026.", styles["Note"]),
        PageBreak(),
    ]
)

story.extend(
    [
        Paragraph("1. Data, training and evaluation periods", styles["Section"]),
        Paragraph(
            "The two models use expanding-time evaluation, but their saved production pipelines do not use identical date blocks. "
            "The table below records the exact implementation rather than forcing one split definition onto both models.",
            styles["BodySmall"],
        ),
        make_table(
            [
                ["Model", "Stage", "Dates used", "Purpose"],
                ["Model A", "Development and blocked validation", "Development through Jan 2022; blocked forecast origins Jan 2017-Jan 2022", "Fit and select price algorithms without using future target values"],
                ["Model A", "Disruption test", "Forecast origins Feb 2022-Dec 2023", "Stress-test price forecasts across the disruption period"],
                ["Model A", "Post-disruption test", "Forecast origins Jan 2024-Dec 2025", "Test recent performance; H12 availability is limited by the Jun 2026 cutoff"],
                ["Model B", "Initial training", "2010-2017, expanding forward", "Fit the supplier-share movement classifier"],
                ["Model B", "Validation", "Jan 2018-Jun 2021", "Historical model selection and probability evaluation"],
                ["Model B", "Disruption test", "Jul 2021-Jun 2023", "Test class predictions under structural change"],
                ["Model B", "Post-disruption test", "Jul 2023-latest completed target", "Test recent generalisation"],
            ],
            [27 * mm, 43 * mm, 72 * mm, 110 * mm],
        ),
        Spacer(1, 5 * mm),
        Paragraph("Historical comparison versus future prediction", styles["Subsection"]),
        Paragraph(
            "Validation and test forecasts are generated as if the later observations were unknown, then compared with the values that subsequently occurred. "
            "This provides ground truth. Separately, both models contain forecasts from a June 2026 origin for September 2026, December 2026 and June 2027. "
            "Those future rows cannot yet be evaluated because the source dataset stops at June 2026.",
            styles["BodySmall"],
        ),
        Paragraph(
            "The original statement 'training 2010-2017, then validation 2018-2021' accurately describes Model B. "
            "Model A's production_v6 saved evaluation uses the model-specific blocks shown above.",
            styles["Callout"],
        ),
        PageBreak(),
    ]
)

a_overview_rows = [["Period", "Rows", "Mean abs. deviation", "Within 10%", "Direction accuracy", "P05-P95 coverage"]]
for split in split_order:
    frame = a_summary[a_summary["split"] == split]
    a_overview_rows.append(
        [
            split_label[split],
            int(frame["rows"].sum()),
            fmt_pct(weighted_mean(frame, "mean_absolute_deviation_pct") / 100),
            fmt_pct(weighted_mean(frame, "within_10_pct_rate")),
            fmt_pct(weighted_mean(frame, "direction_accuracy")),
            fmt_pct(weighted_mean(frame, "p05_p95_coverage")),
        ]
    )

story.extend(
    [
        Paragraph("2. Model A price forecast performance", styles["Section"]),
        Paragraph(
            "Model A predicts an import-price distribution for HS 810820 and HS 810890. P50 is the central prediction. "
            "Percentage deviation measures how far P50 is from the actual price. Coverage reports whether the actual price fell between P05 and P95.",
            styles["BodySmall"],
        ),
        make_table(a_overview_rows, [42 * mm, 24 * mm, 48 * mm, 36 * mm, 42 * mm, 42 * mm]),
        Spacer(1, 5 * mm),
        Paragraph(
            "Overall point accuracy improves in later periods: mean absolute deviation falls from 29.0% in validation to 16.7% during disruption and 12.3% post-disruption. "
            "The broad intervals cover 93.3%-99.3% of actual outcomes. This strong coverage is useful, but intervals that are very wide can be conservative rather than precise.",
            styles["Callout"],
        ),
        scaled_image("model_a_percentage_deviation_and_coverage.png", 235, 100),
        PageBreak(),
        Paragraph("Model A validation: actual price versus P50", styles["Section"]),
        scaled_image("model_a_validation_actual_vs_predicted.png", 245, 165),
        Paragraph("Each panel is one HS code and horizon. The shaded area is P05-P95; the teal line is actual price and the orange line is P50.", styles["Note"]),
        PageBreak(),
        Paragraph("Model A disruption test: actual price versus P50", styles["Section"]),
        scaled_image("model_a_disruption_actual_vs_predicted.png", 245, 165),
        Paragraph("This period checks whether the model remains useful when market behaviour differs from the earlier development period.", styles["Note"]),
        PageBreak(),
        Paragraph("Model A post-disruption test: actual price versus P50", styles["Section"]),
        scaled_image("model_a_post_disruption_actual_vs_predicted.png", 245, 165),
        Paragraph("This is the most recent historical evidence available for judging current price-forecast behaviour.", styles["Note"]),
        PageBreak(),
    ]
)

b_overview_rows = [["Period", "Rows", "Accuracy", "Mean confidence", "Balanced accuracy", "Macro F1", "Brier", "ECE"]]
for split in split_order:
    frame = b_summary[b_summary["split"] == split]
    b_overview_rows.append(
        [
            split_label[split],
            int(frame["rows"].sum()),
            fmt_pct(weighted_mean(frame, "accuracy")),
            fmt_pct(weighted_mean(frame, "mean_model_confidence")),
            fmt_pct(weighted_mean(frame, "balanced_accuracy")),
            fmt_pct(weighted_mean(frame, "macro_f1")),
            fmt_num(weighted_mean(frame, "brier_score")),
            fmt_num(weighted_mean(frame, "expected_calibration_error")),
        ]
    )

story.extend(
    [
        Paragraph("3. Model B supplier-shift performance", styles["Section"]),
        Paragraph(
            "Model B classifies each supplier's future share movement as collapse, fall, stable or rise. "
            "The predicted-class confidence is the largest of the four class probabilities. It is not the same as observed accuracy and should not be presented as a guaranteed chance of correctness.",
            styles["BodySmall"],
        ),
        make_table(b_overview_rows, [40 * mm, 20 * mm, 31 * mm, 38 * mm, 38 * mm, 31 * mm, 27 * mm, 27 * mm]),
        Spacer(1, 5 * mm),
        Paragraph(
            "Observed accuracy rises from 44.8% in validation to 47.5% in disruption and 56.1% post-disruption. "
            "The model's Brier score is lower than the saved historical-frequency and persistence baselines across the evaluated blocks. "
            "Calibration is still imperfect, especially in some horizon-specific reliability bins, so probability claims require caution.",
            styles["Callout"],
        ),
        scaled_image("model_b_accuracy_confidence_and_brier.png", 180, 80),
        PageBreak(),
        Paragraph("Model B ground truth versus predicted classes", styles["Section"]),
        scaled_image("model_b_ground_truth_confusion_matrices.png", 160, 128),
        Paragraph("Diagonal cells are correct classifications. Off-diagonal cells show which movement classes the model confuses.", styles["Note"]),
        PageBreak(),
        Paragraph("Model B probability reliability", styles["Section"]),
        scaled_image("model_b_probability_reliability.png", 245, 118),
        Paragraph(
            "A perfectly calibrated model follows the dashed diagonal: predictions made with 60% confidence would be correct about 60% of the time. "
            "Visible departures from this line explain why calibration remains a limitation even when the classifier beats its probability baselines.",
            styles["BodySmall"],
        ),
        PageBreak(),
    ]
)

a_future_rows = [["HS code", "Target", "Horizon", "Model", "P05", "P50", "P95", "P50 change"]]
for _, row in a_future.sort_values(["hs_code", "horizon_months"]).iterrows():
    a_future_rows.append(
        [
            str(row["hs_code"]),
            row["target_month"],
            f"H{int(row['horizon_months'])}",
            row["selected_model"],
            fmt_gbp(row["p05_price_gbp_per_kg"]),
            fmt_gbp(row["p50_price_gbp_per_kg"]),
            fmt_gbp(row["p95_price_gbp_per_kg"]),
            f"{float(row['p50_change_from_origin_pct']):.1f}%",
        ]
    )

b_future_rows = [["Supplier", "Target", "Horizon", "Predicted class", "Confidence", "Collapse", "Fall", "Stable", "Rise"]]
for _, row in b_future.sort_values(["horizon_months", "country"]).iterrows():
    b_future_rows.append(
        [
            row["country"],
            row["target_month"],
            f"H{int(row['horizon_months'])}",
            row["predicted_class"],
            fmt_pct(row["confidence"]),
            fmt_pct(row["probability_collapse"]),
            fmt_pct(row["probability_fall"]),
            fmt_pct(row["probability_stable"]),
            fmt_pct(row["probability_rise"]),
        ]
    )

story.extend(
    [
        Paragraph("4. Future forecasts after the source cutoff", styles["Section"]),
        Paragraph(
            "Yes, both saved pipelines generate later 2026 and 2027 forecasts. The current files use June 2026 as the forecast origin and produce H3, H6 and H12 targets: September 2026, December 2026 and June 2027. "
            "These rows are predictions only. They do not yet have actual values in the dataset and are therefore excluded from performance scoring.",
            styles["Callout"],
        ),
        scaled_image("future_forecasts_2026_2027.png", 225, 112),
        PageBreak(),
        Paragraph("Model A future price forecasts", styles["Section"]),
        make_table(a_future_rows, [23 * mm, 27 * mm, 22 * mm, 48 * mm, 31 * mm, 31 * mm, 31 * mm, 30 * mm]),
        Spacer(1, 5 * mm),
        Paragraph(
            "The P05-P95 width grows materially for HS 810890, reflecting substantial uncertainty. P50 is not required to rise with the forecast horizon, and interval bounds are not required to be monotonic. "
            "They describe different horizon-specific forecast distributions.",
            styles["BodySmall"],
        ),
        Paragraph("Model B future supplier-shift probabilities", styles["Subsection"]),
        make_table(b_future_rows, [36 * mm, 24 * mm, 18 * mm, 31 * mm, 28 * mm, 25 * mm, 24 * mm, 25 * mm, 24 * mm]),
        PageBreak(),
    ]
)

latest_a = a_detail.sort_values(["target_month", "hs_code", "horizon_months"], ascending=[False, True, True]).head(12)
latest_a_rows = [["Split", "HS", "H", "Origin", "Target", "Actual", "P50", "Abs. deviation", "Covered"]]
for _, row in latest_a.iterrows():
    latest_a_rows.append(
        [
            row["split_label"], str(row["hs_code"]), f"H{int(row['horizon_months'])}", row["forecast_origin"], row["target_month"],
            fmt_gbp(row["actual_price_gbp_per_kg"]), fmt_gbp(row["p50_price_gbp_per_kg"]), f"{row['absolute_deviation_pct']:.1f}%", "Yes" if row["covered_p05_p95"] else "No",
        ]
    )

latest_b = b_detail.sort_values(["target_month", "country", "horizon_months"], ascending=[False, True, True]).head(12)
latest_b_rows = [["Split", "Supplier", "H", "Origin", "Target", "Actual", "Predicted", "Correct", "Confidence"]]
for _, row in latest_b.iterrows():
    latest_b_rows.append(
        [
            row["split_label"], row["country"], f"H{int(row['horizon_months'])}", row["forecast_origin"], row["target_month"],
            row["actual_class"], row["predicted_class"], "Yes" if row["correct"] else "No", fmt_pct(row["confidence"]),
        ]
    )

story.extend(
    [
        Paragraph("5. Recent completed ground-truth examples", styles["Section"]),
        Paragraph("Model A: latest available completed comparisons", styles["Subsection"]),
        make_table(latest_a_rows, [35 * mm, 18 * mm, 14 * mm, 25 * mm, 25 * mm, 31 * mm, 31 * mm, 34 * mm, 22 * mm]),
        Spacer(1, 6 * mm),
        Paragraph("Model B: latest available completed comparisons", styles["Subsection"]),
        make_table(latest_b_rows, [35 * mm, 35 * mm, 14 * mm, 25 * mm, 25 * mm, 25 * mm, 28 * mm, 22 * mm, 28 * mm]),
        Spacer(1, 5 * mm),
        Paragraph(
            "The Excel workbook contains all 630 scored Model A rows and all 1,584 Model B rows, not only these recent examples. "
            "Use the detail sheets to filter by evaluation period, HS code, supplier, origin month, target month and horizon.",
            styles["Note"],
        ),
        PageBreak(),
    ]
)

story.extend(
    [
        Paragraph("6. Conclusions and limitations", styles["Section"]),
        make_table(
            [
                ["Question", "Evidence-based conclusion"],
                ["Can the models be compared with ground truth?", "Yes. Historical validation, disruption and post-disruption rows include actual outcomes and predictions."],
                ["Is there a percentage deviation for Model A?", "Yes. Signed and absolute deviation percentages are calculated per row. The workbook also reports within-5%, within-10% and within-20% rates."],
                ["Is Model B's probability the chance it is correct?", "It is model confidence, not a guaranteed chance. Observed accuracy and calibration plots must be read beside it."],
                ["Are future 2026/2027 forecasts evaluated?", "No. The future target months occur after the June 2026 source cutoff, so ground truth is not available yet."],
                ["What is strongest in Model A?", "Post-disruption point deviation is lower than validation and P05-P95 coverage remains high."],
                ["What is strongest in Model B?", "Post-disruption accuracy and balanced accuracy improve, and Brier score beats the saved baselines."],
                ["What remains a limitation?", "Model A intervals can be wide; Model B probability calibration is uneven and future forecasts require later outcome monitoring."],
            ],
            [70 * mm, 180 * mm],
        ),
        Spacer(1, 7 * mm),
        Paragraph(
            "Demo interpretation: Model A is suitable for demonstrating probabilistic price forecasts with explicit uncertainty. Model B is suitable for research/demo use when the calibration warning remains visible. "
            "Neither model's unobserved future outputs should be described as validated outcomes until actual data arrives.",
            styles["Callout"],
        ),
        Paragraph("Files created with this report", styles["Subsection"]),
        Paragraph(
            "Titanium_Model_A_B_Result_Analysis.xlsx contains the summary, all ground-truth rows, future forecasts and metric definitions. "
            "The RESULT ANALYSIS folder also contains the analysis CSV files and standalone PNG visualizations used in this report.",
            styles["BodySmall"],
        ),
    ]
)

doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"Report saved: {OUTPUT}")
