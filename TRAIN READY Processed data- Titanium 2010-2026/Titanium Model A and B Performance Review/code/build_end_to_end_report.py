from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
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
)
MODEL_A = ROOT / "training titanium model A" / "production_v6_titanium_core_corrected"
MODEL_B = ROOT / "Training Titanium Model B"
REVIEW = ROOT / "Titanium Model A and B Performance Review"
OUTPUT = REVIEW / "visualizations" / "Titanium_Model_A_and_B_End_to_End_Report.pdf"
TMP = Path(r"C:\Anitha\SupplyChainEfficiency\tmp\pdfs\titanium_end_to_end_report")

INK = colors.HexColor("#12353B")
TEAL = colors.HexColor("#087D80")
TEAL_LIGHT = colors.HexColor("#DDEDEA")
ORANGE = colors.HexColor("#CC5D2B")
ORANGE_LIGHT = colors.HexColor("#F5E2D7")
GREEN = colors.HexColor("#337653")
GREEN_LIGHT = colors.HexColor("#DDEBE2")
GOLD = colors.HexColor("#A97018")
GOLD_LIGHT = colors.HexColor("#F5E8C9")
RED = colors.HexColor("#A43B32")
RED_LIGHT = colors.HexColor("#F2DCD9")
MUTED = colors.HexColor("#617579")
LINE = colors.HexColor("#D8D0C1")
PAPER = colors.HexColor("#FBF8F1")
WHITE = colors.white


def load_data() -> dict:
    a_data = pd.read_csv(
        MODEL_A / "data" / "titanium_model_a_production_v6_titanium_core_corrected_training.csv",
        low_memory=False,
    )
    b_data = pd.read_csv(
        MODEL_B / "data" / "titanium_model_b_country_month_training_2010_2026.csv",
        low_memory=False,
    )
    a_metrics = pd.read_csv(REVIEW / "results" / "model_a_performance_summary.csv")
    b_metrics = pd.read_csv(REVIEW / "results" / "model_b_performance_summary.csv")
    a_compare = pd.read_csv(
        MODEL_A / "results" / "validation" / "pretest_validation_model_comparison.csv"
    )
    b_selected = json.loads((MODEL_B / "results" / "selected_models.json").read_text())
    a_manifest = json.loads(
        (MODEL_A / "results" / "latest" / "model_manifest.json").read_text()
    )
    b_manifest = json.loads((MODEL_B / "results" / "model_manifest.json").read_text())
    training_diag = pd.read_csv(REVIEW / "results" / "training_diagnostics.csv")
    a_predictions = pd.read_csv(REVIEW / "results" / "model_a_ground_truth_vs_predictions.csv")
    b_predictions = pd.read_csv(REVIEW / "results" / "model_b_ground_truth_vs_predictions.csv")
    return {
        "a_data": a_data,
        "b_data": b_data,
        "a_metrics": a_metrics,
        "b_metrics": b_metrics,
        "a_compare": a_compare,
        "b_selected": b_selected,
        "a_manifest": a_manifest,
        "b_manifest": b_manifest,
        "training_diag": training_diag,
        "a_predictions": a_predictions,
        "b_predictions": b_predictions,
    }


def pct(value: float, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def money(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"GBP {float(value):.{digits}f}/kg"


def create_charts(data: dict) -> dict[str, Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#A8B3B0",
            "axes.labelcolor": "#12353B",
            "text.color": "#12353B",
            "xtick.color": "#617579",
            "ytick.color": "#617579",
            "axes.titleweight": "bold",
        }
    )
    paths: dict[str, Path] = {}

    # End-to-end lineage diagram.
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.16, 0.25, "UN Comtrade\nUK monthly imports\n2010-Jun 2026"),
        (0.02, 0.18, 0.16, 0.25, "Bank of England FX\n+ ONS ERUC/PPI"),
        (0.24, 0.38, 0.18, 0.28, "Monthly preparation\nGBP/kg, quantities,\nsupplier shares, lags"),
        (0.49, 0.58, 0.20, 0.25, "Model A\nDirect H3/H6/H12\nprice distributions"),
        (0.49, 0.18, 0.20, 0.25, "Model B\nCountry share-change\nclass probabilities"),
        (0.76, 0.38, 0.21, 0.28, "Evidence\nValidation, disruption,\npost-disruption, monitoring"),
    ]
    for x, y, w, h, label in boxes:
        face = "#DDEDEA" if "Model" in label else "#FBF8F1"
        ax.add_patch(
            plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=face,
                          edgecolor="#087D80", linewidth=1.6)
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                transform=ax.transAxes, fontsize=10.5, linespacing=1.35)
    arrows = [
        ((0.18, 0.70), (0.24, 0.55)),
        ((0.18, 0.30), (0.24, 0.49)),
        ((0.42, 0.53), (0.49, 0.70)),
        ((0.42, 0.48), (0.49, 0.30)),
        ((0.69, 0.70), (0.76, 0.56)),
        ((0.69, 0.30), (0.76, 0.48)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#CC5D2B", lw=2))
    ax.set_title("Titanium forecasting data and evidence flow", fontsize=17, pad=12)
    p = TMP / "workflow.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["workflow"] = p

    # Time splits for both models.
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.set_xlim(2009.8, 2026.7)
    ax.set_ylim(-0.7, 2.4)
    ax.set_yticks([1.6, 0.4], ["Model A", "Model B"])
    segments = {
        "Model A": [
            (2010, 2022.08, "Train / tune", "#55A9A4"),
            (2017, 2022.08, "Blocked validation", "#A97018"),
            (2022.08, 2024.0, "Disruption test", "#CC5D2B"),
            (2024.0, 2026.0, "Post-disruption", "#337653"),
        ],
        "Model B": [
            (2010, 2021.5, "Train / tune", "#55A9A4"),
            (2018, 2021.5, "Blocked validation", "#A97018"),
            (2021.5, 2023.5, "Disruption test", "#CC5D2B"),
            (2023.5, 2026.5, "Post-disruption", "#337653"),
        ],
    }
    ymap = {"Model A": 1.6, "Model B": 0.4}
    for model, items in segments.items():
        y = ymap[model]
        for start, end, label, color in items:
            height = 0.32 if "validation" not in label.lower() else 0.16
            ypos = y - height / 2 if height > 0.2 else y + 0.20
            ax.barh(ypos, end - start, left=start, height=height, color=color,
                    edgecolor="white", linewidth=1)
            if end - start > 2.2:
                ax.text((start + end) / 2, ypos, label, ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
    ax.set_xlabel("Forecast origin year")
    ax.set_title("Expanding-window training and out-of-sample evaluation periods")
    ax.grid(axis="x", color="#E4DED3", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    p = TMP / "time_splits.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["splits"] = p

    # Model A validation gates.
    a = data["a_metrics"]
    av = a[a["split"] == "validation"].copy()
    av["label"] = av["hs_code"].astype(str) + " H" + av["horizon_months"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors_a = ["#087D80" if x else "#A43B32" for x in av["passes_acceptance"]]
    axes[0].barh(av["label"], av["canonical_coverage"] * 100, color=colors_a)
    axes[0].axvline(90, color="#CC5D2B", linestyle="--", label="90% gate")
    axes[0].set_xlabel("P05-P95 coverage (%)")
    axes[0].set_xlim(0, 105)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_title("Validation interval coverage")
    axes[1].barh(av["label"], av["mis_ratio_to_baseline"], color=colors_a)
    axes[1].axvline(1.0, color="#CC5D2B", linestyle="--", label="Baseline")
    axes[1].set_xlabel("MIS / no-change baseline MIS")
    axes[1].set_xlim(0, 1.1)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].set_title("Interval-score improvement")
    fig.suptitle("Model A selected configurations pass validation gates", fontsize=16,
                 fontweight="bold")
    fig.tight_layout()
    p = TMP / "model_a_validation.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["a_validation"] = p

    # Model A primary evaluation comparison.
    ae = a[(a["horizon_months"].isin([3, 6])) & (a["split"] != "validation")].copy()
    ae["label"] = (
        ae["split"].map({"disruption_test": "Disruption", "post_disruption_test": "Post"})
        + "\n" + ae["hs_code"].astype(str) + " H" + ae["horizon_months"].astype(str)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    c = ["#337653" if x else "#A43B32" for x in ae["passes_acceptance"]]
    axes[0].bar(range(len(ae)), ae["canonical_coverage"] * 100, color=c)
    axes[0].axhline(90, color="#CC5D2B", linestyle="--")
    axes[0].set_xticks(range(len(ae)), ae["label"], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_title("P05-P95 coverage")
    axes[1].bar(range(len(ae)), ae["mis_ratio_to_baseline"], color=c)
    axes[1].axhline(1, color="#CC5D2B", linestyle="--")
    axes[1].set_xticks(range(len(ae)), ae["label"], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("MIS ratio")
    axes[1].set_title("MIS relative to no-change")
    fig.suptitle("Model A primary H3/H6 out-of-sample evidence", fontsize=16,
                 fontweight="bold")
    fig.tight_layout()
    p = TMP / "model_a_out_of_sample.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["a_oos"] = p

    # Model B Brier and calibration across periods.
    bm = data["b_metrics"].copy()
    order = {"validation": 0, "disruption_test": 1, "post_disruption_test": 2}
    bm["sort"] = bm["split"].map(order) * 10 + bm["horizon_months"]
    bm = bm.sort_values("sort")
    bm["label"] = (
        bm["split"].map({"validation": "Val", "disruption_test": "Disrupt", "post_disruption_test": "Post"})
        + " H" + bm["horizon_months"].astype(str)
    )
    x = np.arange(len(bm))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    w = 0.25
    axes[0].bar(x - w, bm["brier_score"], width=w, color="#087D80", label="Model")
    axes[0].bar(x, bm["historical_frequency_brier"], width=w, color="#A97018", label="Historical frequency")
    axes[0].bar(x + w, bm["persistence_brier"], width=w, color="#CC5D2B", label="Last-change persistence")
    axes[0].set_ylabel("Multiclass Brier score")
    axes[0].set_title("Probability accuracy (lower is better)")
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    bars = axes[1].bar(x, bm["max_calibration_gap"] * 100,
                       color=["#337653" if v <= 0.10 else "#A43B32" for v in bm["max_calibration_gap"]])
    axes[1].axhline(10, color="#CC5D2B", linestyle="--", label="10% gate")
    axes[1].set_ylabel("Maximum calibration gap (%)")
    axes[1].set_title("Worst reliability-bin calibration gap")
    axes[1].set_xticks(x, bm["label"], rotation=35, ha="right")
    axes[1].legend(frameon=False)
    for bar, h in zip(bars, bm["max_calibration_gap"] * 100):
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=7)
    fig.suptitle("Model B probability quality and calibration", fontsize=16, fontweight="bold")
    fig.tight_layout()
    p = TMP / "model_b_performance.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["b_performance"] = p

    # Model B H3 validation confusion matrix.
    bp = data["b_predictions"]
    subset = bp[(bp["split"] == "validation") & (bp["horizon_months"] == 3)]
    classes = ["collapse", "fall", "stable", "rise"]
    cm = pd.crosstab(subset["actual_class"], subset["predicted_class"]).reindex(
        index=classes, columns=classes, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.7))
    im = ax.imshow(cm.values, cmap="GnBu")
    ax.set_xticks(range(4), classes)
    ax.set_yticks(range(4), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Model B H3 validation confusion matrix")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm.iloc[i, j]), ha="center", va="center",
                    color="white" if cm.iloc[i, j] > cm.values.max() * 0.55 else "#12353B",
                    fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = TMP / "model_b_confusion.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["b_confusion"] = p

    # Apparent training versus validation accuracy for Model B.
    td = data["training_diag"].copy()
    bv = data["b_metrics"][data["b_metrics"]["split"] == "validation"].copy()
    horizons = [3, 6, 12]
    train_acc = [float(td.loc[td["horizon_months"] == h, "accuracy"].iloc[0]) * 100 for h in horizons]
    val_acc = [float(bv.loc[bv["horizon_months"] == h, "accuracy"].iloc[0]) * 100 for h in horizons]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.bar(x - 0.18, train_acc, width=0.36, color="#55A9A4", label="Apparent training")
    ax.bar(x + 0.18, val_acc, width=0.36, color="#CC5D2B", label="Blocked validation")
    ax.set_xticks(x, ["H3", "H6", "H12"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model B in-sample accuracy is optimistic")
    ax.legend(frameon=False)
    for xx, vals in [(x - 0.18, train_acc), (x + 0.18, val_acc)]:
        for pos, value in zip(xx, vals):
            ax.text(pos, value + 2, f"{value:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    p = TMP / "model_b_train_validation.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["b_train_validation"] = p
    return paths


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverKicker", parent=styles["Normal"], fontName="Helvetica-Bold",
                              fontSize=9, leading=12, textColor=ORANGE, spaceAfter=10,
                              tracking=1.5, uppercase=True))
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Times-Bold",
                              fontSize=34, leading=37, textColor=INK, alignment=TA_LEFT, spaceAfter=16))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=12, leading=18, textColor=MUTED, spaceAfter=18))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Times-Bold",
                              fontSize=23, leading=27, textColor=INK, spaceBefore=4, spaceAfter=12))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Times-Bold",
                              fontSize=16, leading=20, textColor=INK, spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle(name="H3x", parent=styles["Heading3"], fontName="Helvetica-Bold",
                              fontSize=10.5, leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=9.2, leading=13.8, textColor=INK, spaceAfter=7))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=7.7, leading=10.4, textColor=MUTED, spaceAfter=4))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=9.2, leading=14, textColor=INK, leftIndent=9, rightIndent=9,
                              borderPadding=9, borderColor=ORANGE, borderWidth=0,
                              borderLeftWidth=3, backColor=ORANGE_LIGHT, spaceBefore=7, spaceAfter=10))
    styles.add(ParagraphStyle(name="GoodCallout", parent=styles["Callout"], borderColor=GREEN,
                              backColor=GREEN_LIGHT))
    styles.add(ParagraphStyle(name="WarnCallout", parent=styles["Callout"], borderColor=GOLD,
                              backColor=GOLD_LIGHT))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique",
                              fontSize=7.5, leading=10, textColor=MUTED, alignment=TA_CENTER,
                              spaceBefore=3, spaceAfter=8))
    styles.add(ParagraphStyle(name="TableHead", parent=styles["Normal"], fontName="Helvetica-Bold",
                              fontSize=7, leading=8.5, textColor=WHITE))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=7.1, leading=9.3, textColor=INK))
    styles.add(ParagraphStyle(name="TableCellSmall", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=6.4, leading=8.1, textColor=INK))
    return styles


def p(text: str, styles, style: str = "Bodyx") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str, styles) -> Paragraph:
    return Paragraph(f"&#8226;&nbsp;&nbsp;{text}", styles["Bodyx"])


def report_image(path: Path, max_width: float = 7.05 * inch, max_height: float = 4.9 * inch) -> Image:
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)


def table(rows, widths, styles, small=False, header=True, alignments=None):
    cell_style = styles["TableCellSmall" if small else "TableCell"]
    converted = []
    for ridx, row in enumerate(rows):
        converted.append([
            Paragraph(str(value), styles["TableHead"] if header and ridx == 0 else cell_style)
            for value in row
        ])
    t = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAPER]),
    ]
    if alignments:
        for col, alignment in alignments.items():
            commands.append(("ALIGN", (col, 1), (col, -1), alignment))
    t.setStyle(TableStyle(commands))
    return t


def page_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont("Helvetica-Bold", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, height - 10.5 * mm, "TITANIUM MODELS A AND B")
    canvas.setFont("Helvetica", 7.2)
    canvas.drawRightString(width - 18 * mm, height - 10.5 * mm, "End-to-end performance report")
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.drawString(18 * mm, 9.5 * mm, "Research models - not production approved")
    canvas.drawRightString(width - 18 * mm, 9.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def add_section_title(story, title, subtitle, styles):
    story.append(p(title, styles, "H1x"))
    if subtitle:
        story.append(p(subtitle, styles, "Bodyx"))


def build_report(data: dict, charts: dict[str, Path]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Titanium Models A and B End-to-End Report",
        author="Supply Chain Intelligence model review",
        subject="Data, training, evaluation and model-readiness record",
    )
    story = []

    # Cover.
    story += [
        Spacer(1, 20 * mm),
        p("SUPPLY CHAIN INTELLIGENCE", styles, "CoverKicker"),
        p("Titanium Models A and B", styles, "CoverTitle"),
        p("End-to-end data, training and performance report", styles, "CoverSub"),
        p(
            "A complete record of the data lineage, preprocessing, engineered features, "
            "training design, model selection, validation, disruption testing, post-disruption "
            "monitoring, results and remaining limitations for the titanium import-price and "
            "supplier-shift models.",
            styles,
            "CoverSub",
        ),
        Spacer(1, 8 * mm),
    ]
    cover_cards = [
        ["MODEL A", "MODEL B"],
        ["Import-price forecast", "Country supply-shift classification"],
        ["Direct H3, H6 and H12 P05/P50/P95", "Collapse, fall, stable and rise probabilities"],
        ["6/6 validation configurations pass", "Beats baselines; calibration remains limited"],
    ]
    cover = table(cover_cards, [82 * mm, 82 * mm], styles, header=True)
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("TOPPADDING", (0, 1), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 9),
    ]))
    story += [cover, Spacer(1, 14 * mm), p("Prepared 6 October 2026", styles, "Smallx"),
              p("Evidence current through June 2026", styles, "Smallx"), PageBreak()]

    # Executive summary.
    add_section_title(story, "Executive summary",
                      "The two models share a data foundation but solve different prediction tasks.", styles)
    story.append(report_image(charts["workflow"], max_height=3.0 * inch))
    story.append(p(
        "<b>Model A</b> forecasts titanium import unit value in GBP/kg for HS 810820 and HS 810890. "
        "It produces direct P05, P50 and P95 forecasts at 3, 6 and 12 months. All six selected "
        "HS/horizon configurations passed blocked validation. All four primary H3/H6 combinations "
        "also passed the 2022-2023 disruption test. Post-disruption evidence is mixed: three of four "
        "primary combinations pass, while HS 810890 H3 narrowly loses to the no-change baseline.",
        styles, "GoodCallout"))
    story.append(p(
        "<b>Model B</b> predicts whether each supplier's UK import-weight share will collapse, fall, "
        "remain stable or rise. H3 Random Forest and H6 Extra Trees beat both probability baselines "
        "and pass calibration during the disruption test. Validation and post-disruption probabilities "
        "still fail the strict maximum calibration-gap rule. H12 is reporting-only.",
        styles, "WarnCallout"))
    summary_rows = [
        ["Question", "Model A", "Model B"],
        ["Prediction", "Future GBP/kg price distribution", "Future supplier-share movement class"],
        ["Unit", "GBP per kg", "Probability for four classes"],
        ["Horizon", "Direct H3, H6, H12", "Direct H3, H6, H12"],
        ["Primary demo status", "Suitable research demonstration", "Directional research demonstration"],
        ["Production status", "Not approved", "Not approved"],
    ]
    story += [table(summary_rows, [38 * mm, 64 * mm, 64 * mm], styles), PageBreak()]

    # Scope and sources.
    add_section_title(story, "1. Data collection and provenance",
                      "The implementation uses three permitted source families and explicitly excludes unrelated or incomplete sources.", styles)
    source_rows = [
        ["Source", "Local input", "Coverage", "Use"],
        ["UN Comtrade", "17 annual TradeData CSV exports", "Jan 2010-Jun 2026",
         "UK monthly imports, USD value, net weight, partner rows and supplier shares"],
        ["Bank of England", "Bank of England Database.csv", "Jan 2010-Jun 2026",
         "Monthly USD per GBP; converts USD trade value and unit value into GBP"],
        ["ONS ERUC/PPI", "Inflation Data/ppi.csv; ERUC series", "Jan 2010-Jun 2026",
         "Import-price inflation proxy for other non-ferrous metals, 2015=100"],
    ]
    story.append(table(source_rows, [28 * mm, 42 * mm, 30 * mm, 66 * mm], styles, small=True))
    story.append(Spacer(1, 4 * mm))
    story += [
        p("<b>UN Comtrade extraction:</b> 28 September 2026. Reporter is the United Kingdom, trade flow is imports, frequency is monthly, and the retained as-reported HS codes are 810820 and 810890.", styles),
        p("<b>HS 810820:</b> titanium unwrought and powders. <b>HS 810890:</b> titanium bars, sheet, forgings and other articles. HS 810830 waste and scrap is excluded.", styles),
        p("<b>Excluded sources:</b> Metals-API titanium benchmark data, USGS titanium sponge data, and World Bank aluminium/nickel prices are not model inputs. Their shorter, stale or cross-metal histories did not meet the final source policy.", styles, "Callout"),
        p("The ONS ERUC series is titled 'PPI INDEX IMPORT - C2445 Other non-ferrous metal 2015=100'. It is a broad non-ferrous inflation proxy, not a titanium-only market benchmark.", styles),
        p("Availability assumptions prevent look-ahead: UN Comtrade month-end plus 42 days, Bank of England month-end plus 7 days, and ERUC month-end plus 31 days.", styles),
        report_image(charts["splits"], max_height=2.7 * inch),
        p("Figure 1. Expanding-window model development and evaluation periods. Validation overlaps the earlier model-selection history but uses only prior data within each blocked fold.", styles, "Caption"),
        PageBreak(),
    ]

    # Preprocessing.
    add_section_title(story, "2. Data preparation and quality controls",
                      "The pipeline preserves source anomalies as flags and prevents future labels from entering historical training windows.", styles)
    prep_rows = [
        ["Step", "Operation", "Control"],
        ["1", "Load all annual Comtrade CSVs with Latin-1 encoding", "Retain UK monthly imports only"],
        ["2", "Keep HS 810820 and 810890; preserve World and partner rows", "One World row per HS/month"],
        ["3", "Use World primaryValue and positive World net weight", "Reconcile World value to partner sum"],
        ["4", "Reconstruct missing/zero World weights from partner-country weights", "37 rows flagged; values are not silently replaced"],
        ["5", "Compute USD/kg, join monthly FX, then compute GBP/kg", "GBP value = USD value / USD-per-GBP"],
        ["6", "Join ERUC and create 1/3/6/12-month changes", "Availability-date checks before use"],
        ["7", "Create price, quantity, volatility, supplier and seasonal features", "All lags use current or earlier months"],
        ["8", "Create direct future H3/H6/H12 labels", "Target availability date controls training eligibility"],
        ["9", "Apply data-quality and target-reliability weights", "Problematic targets are downweighted, not hidden"],
    ]
    story.append(table(prep_rows, [12 * mm, 92 * mm, 62 * mm], styles, small=True))
    story.append(p(
        "The monthly GBP unit value is calculated as <b>trade value in GBP / net weight in kg</b>. "
        "It is an import unit value, not a quoted exchange price. Mix changes across grades, forms and suppliers can therefore move the value even when the underlying commodity market is stable.",
        styles, "Callout"))

    a_data = data["a_data"]
    b_data = data["b_data"]
    profile_rows = [
        ["Profile", "Model A matrix", "Model B panel"],
        ["Shape", f"{len(a_data):,} rows x {len(a_data.columns)} columns", f"{len(b_data):,} rows x {len(b_data.columns)} columns"],
        ["Grain", "One month per HS code", "One month per supplier group"],
        ["Entities", "2 HS products", "6 supplier groups"],
        ["Time", "Jan 2010-Jun 2026", "Jan 2010-Jun 2026"],
        ["Feature-complete", f"{int(a_data.model_a_selected_feature_complete.sum())}/{len(a_data)}",
         f"{int(b_data.feature_complete.sum())}/{len(b_data)}"],
        ["Training-eligible H3/H6/H12",
         f"{int(a_data.model_a_training_eligible_h3.sum())} / {int(a_data.model_a_training_eligible_h6.sum())} / {int(a_data.model_a_training_eligible_h12.sum())}",
         f"{int(b_data.model_b_training_eligible_h3.sum())} / {int(b_data.model_b_training_eligible_h6.sum())} / {int(b_data.model_b_training_eligible_h12.sum())}"],
    ]
    story.append(table(profile_rows, [48 * mm, 59 * mm, 59 * mm], styles))
    story.append(p(
        "Missing cells are expected at the edges of the series. The first 12 months lack some lagged features, while the final 3, 6 or 12 months lack future target labels. These rows are excluded horizon-by-horizon rather than imputed as outcomes. FX and ERUC joined successfully for every HS/month row.",
        styles, "GoodCallout"))
    quality_rows = [
        ["Quality condition", "Count", "Treatment"],
        ["Reconstructed World weight", "37 HS-month rows", "Partner weights summed; explicit flag retained"],
        ["UN-estimated World weight", "91 HS-month rows", "Flag retained as model-quality information"],
        ["Robust unit-value outlier", "6 HS-month rows", "Preserved and excluded/weighted where required"],
        ["Partner anomaly", "13 HS-month rows", "Review flag retained"],
        ["Target reliability", "310 reliable; 79 caution; 7 review", "Quality-weighted fitting"],
    ]
    story += [table(quality_rows, [55 * mm, 37 * mm, 74 * mm], styles), PageBreak()]

    # Feature engineering.
    add_section_title(story, "3. Feature engineering",
                      "Both models use only information assumed available at the simulated forecast date.", styles)
    feature_rows = [
        ["Feature family", "Model A price forecast", "Model B supply-shift forecast"],
        ["Price history", "GBP price changes 1/2/3/6/12m; 6/12m volatility",
         "Combined titanium unit-value changes 1/3/6/12m"],
        ["Import volume", "Quantity changes 1/3/12m; 6/12m volatility; log weight; low-volume ratio/flag",
         "Total import changes 1/3/6/12m; country log quantity and changes"],
        ["Supplier structure", "Six monthly supplier shares; HHI; top share; 3m concentration changes",
         "Country share and changes 1/3/6/12m; HHI; top share; country identity"],
        ["External context", "FX changes/volatility; ERUC changes/volatility",
         "FX changes and ERUC changes 1/3/6/12m"],
        ["Calendar", "Month sine/cosine; January 2021 EU collection-change flag",
         "Month sine/cosine; January 2021 EU collection-change flag"],
        ["Quality", "Price shock, low volume and UN-estimated weight flags",
         "Reconstructed/estimated weight flags and partner count"],
    ]
    story.append(table(feature_rows, [34 * mm, 66 * mm, 66 * mm], styles, small=True))
    story.append(p(
        "Log changes express proportional movement: log(value[t] / value[t-k]). Volatility is the rolling standard deviation of historical changes. Month sine and cosine encode the annual cycle without treating December and January as far apart.", styles))
    story.append(p(
        "Future target columns, target months, availability dates, identifiers and audit flags are retained for traceability but are never passed to the estimator as predictive inputs.", styles, "Callout"))
    story.append(report_image(MODEL_A / "visualizations" / "selected_model_feature_importance.png", max_height=4.7 * inch))
    story.append(p("Figure 2. Model A selected-model feature influence. Importance describes model reliance, not causation.", styles, "Caption"))
    story.append(PageBreak())

    # Model A.
    add_section_title(story, "4. Model A: titanium import-price forecast",
                      "Model A predicts a full future price distribution separately for each HS product and horizon.", styles)
    story += [
        p("<b>Target:</b> future logarithmic change in monthly import unit value in GBP/kg. For horizon h: log(price[t+h]) - log(price[t]). The predicted future price is price[t] multiplied by exp(predicted log change).", styles),
        p("<b>Outputs:</b> P05, P10, P25, P50, P75, P90 and P95 price quantiles. The dashboard focuses on P05, P50 and P95. H3, H6 and H12 are direct targets; they are not recursive H1 forecasts.", styles),
        p("<b>Separate models:</b> HS 810820 and HS 810890 are not averaged because their unit values and market composition differ materially.", styles, "Callout"),
    ]
    candidate_rows = [
        ["Candidate", "Role"],
        ["Linear quantile", "Fits conditional price-change quantiles directly"],
        ["Gradient boosting quantile", "Nonlinear tree boosting for each quantile"],
        ["Random forest distribution", "Uses the distribution of tree predictions"],
        ["Elastic Net residual", "Regularized linear correction around a base forecast"],
        ["Huber residual", "Robust linear residual model, less sensitive to outliers"],
        ["Adaptive no-change", "Safe history-based forecast with adaptive interval"],
        ["No-change baseline", "Reference model: future price equals current price"],
    ]
    story.append(table(candidate_rows, [49 * mm, 117 * mm], styles))
    story.append(p(
        "Candidate selection uses blocked validation ending January 2022. A candidate is eligible when calibrated P05-P95 coverage is at least 90% and Mean Interval Score is lower than the independently calibrated no-change baseline. The minimum eligible MIS wins; raw pinball loss is only a tie-breaker.",
        styles, "GoodCallout"))

    av = data["a_metrics"][data["a_metrics"]["split"] == "validation"].sort_values(["hs_code", "horizon_months"])
    selected_rows = [["HS", "H", "Selected model", "Train rows", "Coverage", "MIS ratio"]]
    for _, row in av.iterrows():
        hkey = f"h{int(row.horizon_months)}"
        train_rows = data["a_manifest"]["hs_models"][str(int(row.hs_code))]["horizons"][hkey]["training_rows"]
        selected_rows.append([
            str(int(row.hs_code)), f"H{int(row.horizon_months)}", row.selected_model,
            str(train_rows), pct(row.canonical_coverage), num(row.mis_ratio_to_baseline),
        ])
    story.append(table(selected_rows, [20 * mm, 14 * mm, 63 * mm, 23 * mm, 23 * mm, 23 * mm], styles))
    story.append(report_image(charts["a_validation"], max_height=3.3 * inch))
    story.append(p("Figure 3. Every selected Model A configuration clears both validation interval gates.", styles, "Caption"))
    story.append(PageBreak())

    add_section_title(story, "5. Model A evaluation results",
                      "Point accuracy and interval quality are evaluated together; interval gates drive acceptance.", styles)
    metric_defs = [
        ["Metric", "Meaning"],
        ["MAE / RMSE", "Average and squared-error-sensitive P50 price error in GBP/kg"],
        ["sMAPE", "Scale-normalized percentage difference between actual and P50"],
        ["Pinball loss", "Quantile forecast error; lower is better"],
        ["90% coverage", "Share of actual outcomes inside P05-P95; gate is at least 90%"],
        ["MIS", "Interval width plus a severe penalty for misses; lower is better"],
        ["MIS ratio", "Selected-model MIS divided by no-change baseline MIS; pass is below 1"],
    ]
    story.append(table(metric_defs, [37 * mm, 129 * mm], styles))

    val_rows = [["HS", "H", "Model", "MAE", "sMAPE", "Coverage", "MIS ratio", "Result"]]
    for _, row in av.iterrows():
        val_rows.append([
            str(int(row.hs_code)), f"H{int(row.horizon_months)}", row.selected_model,
            money(row.mae_gbp_per_kg), pct(row.smape), pct(row.canonical_coverage),
            num(row.mis_ratio_to_baseline), "Pass" if row.passes_acceptance else "Review",
        ])
    story.append(p("Validation: forecast origins January 2017-January 2022", styles, "H2x"))
    story.append(table(val_rows, [15 * mm, 12 * mm, 42 * mm, 27 * mm, 18 * mm, 20 * mm, 18 * mm, 14 * mm], styles, small=True))
    story.append(p(
        "HS 810820 validation sMAPE ranges from 13.7% to 16.3%. HS 810890 is more difficult, with 23.8% to 32.4% sMAPE and much larger GBP/kg errors because that product series is higher-priced and more volatile.",
        styles))
    story.append(report_image(charts["a_oos"], max_height=3.35 * inch))
    story.append(p("Figure 4. Primary H3/H6 disruption and post-disruption interval tests.", styles, "Caption"))

    ae = data["a_metrics"][(data["a_metrics"]["split"] != "validation") & (data["a_metrics"]["horizon_months"].isin([3, 6]))]
    eval_rows = [["Period", "HS", "H", "MAE", "sMAPE", "Coverage", "MIS ratio", "Result"]]
    for _, row in ae.sort_values(["split", "hs_code", "horizon_months"]).iterrows():
        eval_rows.append([
            "Disruption" if row.split == "disruption_test" else "Post",
            str(int(row.hs_code)), f"H{int(row.horizon_months)}", money(row.mae_gbp_per_kg),
            pct(row.smape), pct(row.canonical_coverage), num(row.mis_ratio_to_baseline),
            "Pass" if row.passes_acceptance else "Review",
        ])
    story.append(table(eval_rows, [24 * mm, 15 * mm, 12 * mm, 26 * mm, 18 * mm, 20 * mm, 20 * mm, 18 * mm], styles, small=True))
    story.append(p(
        "The disruption test covers February 2022-December 2023 and all four primary H3/H6 combinations pass. The post-disruption evaluation covers January 2024-December 2025: both HS 810820 horizons and HS 810890 H6 pass; HS 810890 H3 has 100% coverage but an MIS ratio of 1.037, so the no-change baseline is slightly sharper.",
        styles, "GoodCallout"))
    story.append(p(
        "H12 is reported but not counted in the four primary combinations. Post-disruption H12 has only 18 of 24 outcomes because the source ends in June 2026. Its 75% observed coverage and infinite MIS are therefore incomplete evidence, not a valid acceptance result.",
        styles, "WarnCallout"))

    # Model B.
    add_section_title(story, "6. Model B: titanium supply shift by country",
                      "Model B predicts how each supplier's share of UK titanium import weight will change.", styles)
    story += [
        p("The six supplier groups are USA, Japan, France, Russian Federation, China and Other countries. Monthly country weight is reconstructed from each HS row's supplier share multiplied by net weight, then aggregated across HS 810820 and 810890.", styles),
        p("For horizon h, the numeric target is share[t+h] - share[t]. Fixed class boundaries are derived only from pooled H3 changes ending June 2021.", styles),
    ]
    class_rows = [
        ["Class", "Future share change"],
        ["Collapse", "Below -14.4608 percentage points"],
        ["Fall", "-14.4608 to below -2.2945 percentage points"],
        ["Stable", "-2.2945 through +2.0887 percentage points"],
        ["Rise", "Above +2.0887 percentage points"],
    ]
    story.append(table(class_rows, [43 * mm, 123 * mm], styles))
    story.append(p(
        "The specification originally requested class-boundary history from 2005. Available monthly data begins in 2010, so the frozen thresholds use January 2010-June 2021. This is a documented specification deviation.",
        styles, "WarnCallout"))
    bmodel_rows = [["Horizon", "Selected model", "Parameters", "Temperature", "Final train rows", "Use"]]
    for h in [3, 6, 12]:
        entry = data["b_manifest"]["horizons"][f"h{h}"]
        params = ", ".join(f"{k}={v}" for k, v in entry["params"].items())
        bmodel_rows.append([
            f"H{h}", entry["candidate"], params, str(entry["temperature"]),
            str(entry["training_rows"]), "Reporting only" if entry["reporting_only"] else "Primary",
        ])
    story.append(table(bmodel_rows, [15 * mm, 30 * mm, 58 * mm, 22 * mm, 22 * mm, 20 * mm], styles, small=True))
    story.append(p(
        "Candidate families are multinomial logistic regression, Random Forest and Extra Trees. Numeric values receive median imputation. Logistic regression also receives standardization. Country identity is one-hot encoded. Probability temperature is selected on blocked validation to reduce Brier score.",
        styles))
    story.append(report_image(MODEL_B / "visualizations" / "selected_model_feature_importance.png", max_height=3.0 * inch))
    story.append(p("Figure 5. Model B feature importance. Current country share and its recent changes dominate all horizons.", styles, "Caption"))

    # Model B results.
    add_section_title(story, "7. Model B evaluation results",
                      "Probability accuracy beats simple baselines, but reliability varies by period and horizon.", styles)
    bdefs = [
        ["Metric", "Meaning"],
        ["Accuracy", "Share of rows whose most probable class equals the actual class"],
        ["Balanced accuracy", "Average class recall; reduces dominance by common classes"],
        ["Macro F1", "Equal-weight average classification F1 across four classes"],
        ["Brier score", "Squared error of the four predicted probabilities; lower is better"],
        ["Log loss", "Penalty for assigning low probability to the true class; lower is better"],
        ["ECE", "Weighted average difference between confidence and observed accuracy"],
        ["Maximum calibration gap", "Worst populated reliability-bin gap; required at or below 10%"],
    ]
    story.append(table(bdefs, [45 * mm, 121 * mm], styles))
    story.append(report_image(charts["b_performance"], max_height=4.45 * inch))
    story.append(p("Figure 6. Model B beats both Brier baselines in every shown split, but only disruption H3/H6 meet the strict maximum-gap calibration rule.", styles, "Caption"))

    bm = data["b_metrics"].sort_values(["split", "horizon_months"])
    b_rows = [["Period", "H", "Model", "Accuracy", "Bal. acc.", "Macro F1", "Brier", "Max gap", "Result"]]
    for _, row in bm.iterrows():
        period = {"validation": "Validation", "disruption_test": "Disruption", "post_disruption_test": "Post"}[row.split]
        result = "Pass" if row.passes_acceptance else ("Report" if int(row.horizon_months) == 12 else "Recalibrate")
        b_rows.append([
            period, f"H{int(row.horizon_months)}", row.selected_model, pct(row.accuracy),
            pct(row.balanced_accuracy), pct(row.macro_f1), num(row.brier_score),
            pct(row.max_calibration_gap), result,
        ])
    story.append(table(b_rows, [20 * mm, 11 * mm, 27 * mm, 17 * mm, 17 * mm, 17 * mm, 16 * mm, 18 * mm, 23 * mm], styles, small=True))
    story.append(p(
        "Validation accuracy is approximately 42%-46%, while post-disruption accuracy is approximately 54%-58%. The model carries useful directional information because its Brier score is lower than both baselines. However, a forecast can beat a baseline and still be poorly calibrated: predicted confidence does not consistently match observed frequency within every probability bin.",
        styles, "Callout"))

    story.append(KeepTogether([
        p("Classification behavior and overfitting diagnostic", styles, "H2x"),
        report_image(charts["b_confusion"], max_height=3.45 * inch),
        p("Figure 7. H3 blocked-validation confusion matrix. The model distinguishes common fall, stable and rise cases better than the relatively rare collapse class.", styles, "Caption"),
    ]))
    story.append(report_image(charts["b_train_validation"], max_height=3.3 * inch))
    story.append(p("Figure 8. Apparent training accuracy is around 80%, materially above blocked-validation accuracy. This is an overfitting diagnostic; in-sample results do not determine demo readiness.", styles, "Caption"))
    story.append(p(
        "<b>Disruption test, July 2021-June 2023:</b> H3 accuracy 50.0%, Brier 0.603 and maximum gap 7.7%; H6 accuracy 47.2%, Brier 0.624 and maximum gap 8.3%. Both primary horizons pass. H12 has a 19.8% maximum gap and remains reporting-only.",
        styles, "GoodCallout"))
    story.append(p(
        "<b>Post-disruption monitoring:</b> H3/H6/H12 accuracy improves to 56.1%/54.4%/58.3% and Brier scores remain below both baselines. Maximum gaps rise to 16.8%/47.5%/72.3%, so exact probabilities require recalibration and prospective monitoring.",
        styles, "WarnCallout"))

    # Evaluation design.
    add_section_title(story, "8. Training, validation and test design",
                      "Both pipelines use blocked or rolling time splits. Random shuffling is not used.", styles)
    split_rows = [
        ["Stage", "Model A", "Model B"],
        ["Source history", "Jan 2010-Jun 2026", "Jan 2010-Jun 2026"],
        ["Development pool", "Origins through Jan 2022", "Origins through Jun 2021"],
        ["Blocked validation", "Jan 2017-Jan 2022; 60 selected forecasts per HS/horizon",
         "2018, 2019 and Jan 2020-Jun 2021 blocks"],
        ["Disruption test", "Feb 2022-Dec 2023; 23 origins per HS/horizon",
         "Jul 2021-Jun 2023; 24 months x 6 suppliers"],
        ["Post-disruption", "Jan 2024-Dec 2025; 24 expected origins",
         "Jul 2023-Jun 2026; realized labels shrink with horizon"],
        ["Latest bundle origin", "Jun 2026", "Jun 2026"],
        ["Training mode", "Expanding/rolling, target availability enforced", "Expanding/rolling, target availability enforced"],
    ]
    story.append(table(split_rows, [37 * mm, 65 * mm, 64 * mm], styles, small=True))
    story += [
        p("For each simulated forecast origin, training includes only rows dated before the origin and only labels whose assumed publication date is already available. This blocks direct target leakage.", styles, "GoodCallout"),
        p("Model A uses separate pre-test residual histories to calibrate the selected model and the no-change baseline. This is important because sharing one calibration history would make MIS comparisons unfair.", styles),
        p("Model B freezes the selected classifier family and temperature before the disruption test. Historical-frequency and last-observed-change baselines are scored on the same rows.", styles),
    ]

    readiness_rows = [
        ["Evidence question", "Model A", "Model B"],
        ["Does validation beat its baseline?", "Yes: 6/6 configurations", "Yes on Brier: 3/3 horizons"],
        ["Are primary disruption gates met?", "Yes: 4/4 H3/H6 combinations", "Yes: H3 and H6"],
        ["Is post-disruption behavior stable?", "Mostly; 3/4 primary combinations pass", "Brier remains useful; calibration fails"],
        ["Can it be demonstrated?", "Yes, with research-model label", "Yes, as directional probabilities"],
        ["Production approved?", "No", "No"],
    ]
    story += [table(readiness_rows, [54 * mm, 56 * mm, 56 * mm], styles), PageBreak()]

    # Interpretation and limitations.
    add_section_title(story, "9. Interpretation, limitations and next work",
                      "The models are useful evidence packages, but neither should be presented as production-approved.", styles)
    story += [
        p("<b>Model A interpretation:</b> the probabilistic intervals are the strongest result. Coverage is consistently high and the selected intervals are generally sharper than no-change. Median-price error is notably larger for HS 810890, so price-level forecasts for this code should be presented with the interval, not as a single precise number.", styles),
        p("<b>Model B interpretation:</b> the model detects meaningful supplier-share direction better than simple probability baselines. The main weakness is calibration. A 60% predicted event should occur around 60% of the time; the post-disruption reliability bins do not consistently satisfy this.", styles),
    ]
    limitation_rows = [
        ["Limitation", "Effect", "Required action"],
        ["ERUC is not titanium-specific", "Broad inflation may miss titanium market shocks", "Seek a long titanium-specific benchmark when available"],
        ["History starts in 2010", "Model B class boundaries do not meet requested 2005 start", "Keep deviation visible; extend only with authoritative data"],
        ["Supplier quantities reconstructed", "Model B inherits the aggregated shares in trade core", "Rebuild from raw partner rows for production"],
        ["Post-disruption data previously explored", "Not a pristine final holdout", "Collect a genuinely prospective holdout"],
        ["Model B calibration drift", "Exact probabilities may be over/under-confident", "Recalibrate prospectively and monitor by horizon"],
        ["H12 target incompleteness", "Fewer independent outcomes and incomplete Model A post period", "Treat H12 as reporting-only until more labels arrive"],
        ["Model A source dependency missing", "Selected V6 training script references an absent improved_v2 source file", "Restore/vendor the dependency before CI or clean-room retraining"],
    ]
    story.append(table(limitation_rows, [45 * mm, 58 * mm, 63 * mm], styles, small=True))
    story.append(p(
        "The saved Model A bundles, metrics, predictions and dashboards are present. However, the selected V6 training script imports a sibling improved_v2 Python source that is not currently inside the selected Model A folder tree. This does not invalidate the saved evidence, but it is a reproducibility and deployment-packaging blocker.",
        styles, "WarnCallout"))
    story += [
        p("Recommended next sequence", styles, "H2x"),
        bullet("Restore the missing Model A inherited training source and run the package from a clean environment.", styles),
        bullet("Freeze the current artifacts and record checksums for data, model bundles and result files.", styles),
        bullet("Collect new prospective months without retuning; score them as a genuinely unseen holdout.", styles),
        bullet("Recalibrate Model B probabilities using only new validation history and recheck maximum calibration gap.", styles),
        bullet("Keep H12 reporting-only until the number of independent outcomes is sufficient.", styles),
        bullet("Define production monitoring thresholds for drift, missing sources, interval coverage and calibration.", styles),
        PageBreak(),
    ]

    # Artifact map.
    add_section_title(story, "10. Reproducibility and artifact map",
                      "The following files hold the training data, models and auditable predictions described in this report.", styles)
    artifacts = [
        ["Artifact", "Location within selected package"],
        ["Model A training data", "production_v6_titanium_core_corrected/data/...training.csv"],
        ["Model A saved bundles", "production_v6_titanium_core_corrected/models/*.joblib"],
        ["Model A evaluation", "production_v6_titanium_core_corrected/results/validation, stress_2022_2023, post_2024_2025"],
        ["Model B training data", "Training Titanium Model B/data/titanium_model_b_country_month_training_2010_2026.csv"],
        ["Model B saved bundle", "Training Titanium Model B/models/titanium_model_b_country_supply_shift_selected.joblib"],
        ["Model B evaluation", "Training Titanium Model B/results/*.csv and *.json"],
        ["Combined evidence CSVs", "Titanium Model A and B Performance Review/results"],
        ["Combined dashboard", "Titanium Model A and B Performance Review/visualizations/...dashboard.html"],
    ]
    story.append(table(artifacts, [48 * mm, 118 * mm], styles, small=True))
    story.append(Spacer(1, 5 * mm))
    story.append(p("Metric interpretation rules", styles, "H2x"))
    story += [
        bullet("Do not compare Model A MIS directly with Model B Brier score. They measure different tasks.", styles),
        bullet("Training diagnostics are secondary. Validation and time-ordered test results determine readiness.", styles),
        bullet("A pass means the implemented acceptance rule was met for that period. It does not mean the model is risk-free or production approved.", styles),
        bullet("A failed calibration gate does not mean classification has no value; it means the stated probability should not be treated as fully reliable.", styles),
    ]
    story.append(Spacer(1, 7 * mm))
    story.append(p("Final assessment", styles, "H2x"))
    story.append(p(
        "Model A is the stronger demonstration package because its interval gates pass consistently across validation and the primary disruption period. Model B is a useful directional prototype that improves on simple probability baselines, but its validation and post-disruption probabilities need recalibration. The shared preprocessing is auditable and time-aware. Production approval remains pending prospective evidence, operational packaging and monitoring controls.",
        styles, "GoodCallout"))

    doc.build(story, onFirstPage=page_header_footer, onLaterPages=page_header_footer)


def verify_report() -> None:
    if not OUTPUT.exists() or OUTPUT.stat().st_size < 100_000:
        raise RuntimeError("Report PDF was not created or is unexpectedly small.")
    from pypdf import PdfReader

    reader = PdfReader(str(OUTPUT))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    required = [
        "Data collection and provenance",
        "Model A: titanium import-price forecast",
        "Model B: titanium supply shift by country",
        "Training, validation and test design",
        "Final assessment",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"Report verification failed; missing sections: {missing}")
    print(f"Created: {OUTPUT}")
    print(f"Pages: {len(reader.pages)}")
    print(f"Bytes: {OUTPUT.stat().st_size}")
    print("Content verification: PASS")


def main() -> None:
    data = load_data()
    charts = create_charts(data)
    build_report(data, charts)
    verify_report()


if __name__ == "__main__":
    main()
