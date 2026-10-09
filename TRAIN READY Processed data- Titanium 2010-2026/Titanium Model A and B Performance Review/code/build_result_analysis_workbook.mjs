import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const reviewRoot = String.raw`C:\Anitha\SupplyChainEfficiency\NEW SCI PREDICTION TITANIUM\TRAIN READY Processed data- Titanium 2010-2026\Titanium Model A and B Performance Review`;
const inputDir = path.join(reviewRoot, "RESULT ANALYSIS");
const outputPath = path.join(inputDir, "Titanium_Model_A_B_Result_Analysis.xlsx");
const previewDir = path.join(inputDir, "workbook_previews");
const font = "Arial";
const palette = {
  navy: "#12353B",
  teal: "#087D80",
  orange: "#CC5D2B",
  sand: "#F3EBDD",
  paleTeal: "#E6F1EF",
  paleOrange: "#F8E5D8",
  ink: "#17363C",
  muted: "#617579",
  grid: "#D8D0C2",
  green: "#337653",
  red: "#A43B32",
  amber: "#A97018",
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows.filter((r) => r.some((value) => value !== ""));
}

function isNumericHeader(header) {
  const textHeaders = new Set([
    "hs_code", "horizon_months", "forecast_origin", "target_month", "split",
    "split_label", "hs_product", "model", "selected_model", "country",
    "actual_class", "predicted_class", "ground_truth_status", "acceptance_status",
  ]);
  if (textHeaders.has(header)) return false;
  return /price|error|deviation|width|coverage|rate|accuracy|confidence|probability|margin|rows|rmse|mae|smape|mape|brier|loss|calibration|f1|gap|change|share|score|correct_numeric|probability_sum|squared/i.test(header);
}

function convertCell(header, value) {
  if (value === "") return null;
  if (value === "True" || value === "true") return true;
  if (value === "False" || value === "false") return false;
  if (isNumericHeader(header) && Number.isFinite(Number(value))) return Number(value);
  return value;
}

async function loadCsv(filename) {
  const text = await fs.readFile(path.join(inputDir, filename), "utf8");
  const rows = parseCsv(text);
  const headers = rows[0];
  const values = rows.slice(1).map((row) => headers.map((header, index) => convertCell(header, row[index] ?? "")));
  return { headers, values, objects: values.map((row) => Object.fromEntries(headers.map((h, i) => [h, row[i]]))) };
}

function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function weightedMean(rows, field) {
  const usable = rows.filter((row) => Number.isFinite(Number(row[field])) && Number.isFinite(Number(row.rows)));
  const denominator = usable.reduce((sum, row) => sum + Number(row.rows), 0);
  return denominator ? usable.reduce((sum, row) => sum + Number(row[field]) * Number(row.rows), 0) / denominator : null;
}

function title(sheet, text, subtitle = "") {
  sheet.showGridLines = false;
  sheet.getRange("A2:H2").merge();
  sheet.getRange("A2").values = [[text]];
  sheet.getRange("A2").format.font = { name: font, size: 16, bold: true, color: palette.navy };
  sheet.getRange("A3:H3").merge();
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange("A3").format.font = { name: font, size: 10, italic: true, color: palette.muted };
  sheet.getRange("A4:H4").format.borders = { bottom: { style: "thin", color: palette.orange } };
}

function styleHeader(range) {
  range.format.fill = palette.navy;
  range.format.font = { name: font, size: 10, bold: true, color: "#FFFFFF" };
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.wrapText = true;
  range.format.rowHeight = 30;
}

function styleBody(range) {
  range.format.font = { name: font, size: 9, color: palette.ink };
  range.format.verticalAlignment = "center";
}

function formatByHeaders(sheet, headers, startRow, rowCount) {
  headers.forEach((header, index) => {
    const col = columnName(index);
    const range = sheet.getRange(`${col}${startRow}:${col}${startRow + rowCount - 1}`);
    if (/price_gbp_per_kg|mae_gbp_per_kg|rmse_gbp_per_kg|interval_width_gbp_per_kg|absolute_error_gbp_per_kg/i.test(header)) {
      range.format.numberFormat = '"GBP "0.0000';
    } else if (/rate$|accuracy$|coverage$|direction_accuracy|within_.*_rate|macro_f1|passes_acceptance|correct_numeric/i.test(header)) {
      range.format.numberFormat = "0.0%";
    } else if (/probability|confidence|margin|brier|log_loss|calibration|gap/i.test(header)) {
      range.format.numberFormat = "0.000";
    } else if (/deviation_pct|change_from_origin_pct|width_pct|smape|mape/i.test(header)) {
      range.format.numberFormat = "0.0";
    } else if (/rows$|horizon_months/i.test(header)) {
      range.format.numberFormat = "0";
    }
  });
}

function addDataSheet(workbook, name, dataset, tableName, tabColor) {
  const sheet = workbook.worksheets.add(name);
  sheet.tabColor = tabColor;
  sheet.showGridLines = false;
  const endCol = columnName(dataset.headers.length - 1);
  sheet.getRange(`A1:${endCol}${dataset.values.length + 1}`).values = [dataset.headers, ...dataset.values];
  styleHeader(sheet.getRange(`A1:${endCol}1`));
  styleBody(sheet.getRange(`A2:${endCol}${dataset.values.length + 1}`));
  formatByHeaders(sheet, dataset.headers, 2, dataset.values.length);
  const table = sheet.tables.add(`A1:${endCol}${dataset.values.length + 1}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  sheet.freezePanes.freezeRows(1);
  sheet.getUsedRange().format.autofitColumns();
  dataset.headers.forEach((header, index) => {
    const col = columnName(index);
    let width = 13;
    if (/product|model|status/i.test(header)) width = 25;
    if (/date|origin|month/i.test(header)) width = 14;
    if (/probability|deviation|coverage|confidence|interval/i.test(header)) width = 17;
    sheet.getRange(`${col}:${col}`).format.columnWidth = Math.min(width, 28);
  });
  return sheet;
}

const [aSummary, aDetail, bSummary, bDetail, aFuture, bFuture] = await Promise.all([
  loadCsv("model_a_performance_analysis.csv"),
  loadCsv("model_a_ground_truth_vs_predictions_analysis.csv"),
  loadCsv("model_b_performance_analysis.csv"),
  loadCsv("model_b_ground_truth_vs_predictions_analysis.csv"),
  loadCsv("model_a_future_forecasts_2026_2027.csv"),
  loadCsv("model_b_future_forecasts_2026_2027.csv"),
]);

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
summary.tabColor = palette.navy;
title(summary, "Titanium Models A and B performance analysis", "Historical ground truth comparisons plus future forecasts from a June 2026 origin");

const splitOrder = ["validation", "disruption_test", "post_disruption_test"];
const splitNames = { validation: "Validation", disruption_test: "Disruption test", post_disruption_test: "Post-disruption test" };
const aOverview = splitOrder.map((split) => {
  const rows = aSummary.objects.filter((row) => row.split === split);
  return [
    splitNames[split],
    rows.reduce((sum, row) => sum + Number(row.rows), 0),
    weightedMean(rows, "mean_absolute_deviation_pct") / 100,
    weightedMean(rows, "within_10_pct_rate"),
    weightedMean(rows, "direction_accuracy"),
    weightedMean(rows, "p05_p95_coverage"),
  ];
});
const bOverview = splitOrder.map((split) => {
  const rows = bSummary.objects.filter((row) => row.split === split);
  return [
    splitNames[split],
    rows.reduce((sum, row) => sum + Number(row.rows), 0),
    weightedMean(rows, "accuracy"),
    weightedMean(rows, "mean_model_confidence"),
    weightedMean(rows, "balanced_accuracy"),
    weightedMean(rows, "macro_f1"),
    weightedMean(rows, "brier_score"),
    weightedMean(rows, "expected_calibration_error"),
  ];
});

summary.getRange("A6:F6").merge();
summary.getRange("A6").values = [["Model A: import price forecast"]];
summary.getRange("A6").format = { fill: palette.paleTeal, font: { name: font, bold: true, color: palette.navy } };
summary.getRange("A7:F10").values = [["Evaluation period", "Rows", "Mean absolute deviation", "Within 10%", "Direction accuracy", "P05-P95 coverage"], ...aOverview];
styleHeader(summary.getRange("A7:F7"));
styleBody(summary.getRange("A8:F10"));
summary.getRange("C8:F10").format.numberFormat = "0.0%";

summary.getRange("A13:H13").merge();
summary.getRange("A13").values = [["Model B: supplier-share movement classification"]];
summary.getRange("A13").format = { fill: palette.paleOrange, font: { name: font, bold: true, color: palette.navy } };
summary.getRange("A14:H17").values = [["Evaluation period", "Rows", "Accuracy", "Mean confidence", "Balanced accuracy", "Macro F1", "Brier score", "Calibration error"], ...bOverview];
styleHeader(summary.getRange("A14:H14"));
styleBody(summary.getRange("A15:H17"));
summary.getRange("C15:F17").format.numberFormat = "0.0%";
summary.getRange("G15:H17").format.numberFormat = "0.000";

summary.getRange("A20:H24").values = [
  ["Interpretation", "Meaning", null, null, null, null, null, null],
  ["Model A deviation", "Absolute percentage difference between P50 and the observed GBP/kg price. Lower is better.", null, null, null, null, null, null],
  ["Model A coverage", "Share of actual prices inside the P05-P95 interval. A 90% interval is expected to cover about 90% over many forecasts.", null, null, null, null, null, null],
  ["Model B confidence", "Probability assigned to the predicted class. Because calibration is imperfect, this is not a guaranteed real-world chance of correctness.", null, null, null, null, null, null],
  ["Future forecasts", "Targets in Sep 2026, Dec 2026 and Jun 2027 have no ground truth in the source data ending Jun 2026.", null, null, null, null, null, null],
];
summary.getRange("A20:H20").merge();
summary.getRange("A20").format = { fill: palette.navy, font: { name: font, bold: true, color: "#FFFFFF" } };
for (let row = 21; row <= 24; row += 1) summary.getRange(`B${row}:H${row}`).merge();
summary.getRange("A21:H24").format.font = { name: font, size: 10, color: palette.ink };
summary.getRange("A21:A24").format.font = { name: font, size: 10, bold: true, color: palette.navy };
summary.getRange("B21:H24").format.wrapText = true;
summary.getRange("A21:H24").format.rowHeight = 31;

summary.getRange("T7:X10").values = [
  ["Evaluation period", "Mean absolute deviation", "Within 10%", "Direction accuracy", "P05-P95 coverage"],
  ...aOverview.map((row) => [row[0], row[2], row[3], row[4], row[5]]),
];
summary.getRange("U8:X10").format.numberFormat = "0.0%";
const chartA = summary.charts.add("column", summary.getRange("T7:X10"));
chartA.title = "Model A performance by evaluation period";
chartA.hasLegend = true;
chartA.yAxis = { numberFormatCode: "0%", numberFormatSourceLinked: false };
chartA.setPosition("J6", "R18");
summary.getRange("T14:X17").values = [
  ["Evaluation period", "Accuracy", "Mean confidence", "Balanced accuracy", "Macro F1"],
  ...bOverview.map((row) => [row[0], row[2], row[3], row[4], row[5]]),
];
summary.getRange("U15:X17").format.numberFormat = "0.0%";
const chartB = summary.charts.add("column", summary.getRange("T14:X17"));
chartB.title = "Model B classification performance";
chartB.hasLegend = true;
chartB.yAxis = { numberFormatCode: "0%", numberFormatSourceLinked: false };
chartB.setPosition("J20", "R33");

summary.getRange("A:R").format.font = { name: font, size: 10, color: palette.ink };
summary.getRange("A:A").format.columnWidth = 24;
summary.getRange("B:B").format.columnWidth = 18;
summary.getRange("C:H").format.columnWidth = 17;
summary.getRange("I:I").format.columnWidth = 3;
summary.getRange("T:X").format.columnWidth = 0.1;
summary.getRange("T7:X17").format.font = { name: font, size: 1, color: "#FFFFFF" };
summary.freezePanes.freezeRows(4);

addDataSheet(workbook, "Model A Metrics", aSummary, "ModelAMetrics", palette.teal);
addDataSheet(workbook, "Model A Ground Truth", aDetail, "ModelAGroundTruth", palette.teal);
addDataSheet(workbook, "Model B Metrics", bSummary, "ModelBMetrics", palette.orange);
addDataSheet(workbook, "Model B Ground Truth", bDetail, "ModelBGroundTruth", palette.orange);

addDataSheet(workbook, "Model A Future", aFuture, "ModelAFuture", palette.amber);
addDataSheet(workbook, "Model B Future", bFuture, "ModelBFuture", palette.amber);

const guide = workbook.worksheets.add("Guide");
guide.tabColor = palette.muted;
title(guide, "Data periods and metric guide", "Use this sheet to distinguish historical evaluation from genuine future forecasts");
guide.getRange("A6:D11").values = [
  ["Model", "Period", "Role", "Exact implementation"],
  ["Model A", "Development / expanding training", "Model fitting", "Development through Jan 2022; blocked origins Jan 2017-Jan 2022 are evaluated out of fold."],
  ["Model A", "Disruption test", "Stress test", "Forecast origins Feb 2022-Dec 2023."],
  ["Model A", "Post-disruption test", "Recent holdout", "Forecast origins Jan 2024-Dec 2025; H12 is limited by the Jun 2026 source cutoff."],
  ["Model B", "Development / expanding training", "Model fitting", "Starts with 2010-2017 and expands forward."],
  ["Model B", "Validation / disruption / post", "Historical evaluation", "Validation Jan 2018-Jun 2021; disruption Jul 2021-Jun 2023; post Jul 2023-latest completed target."],
];
styleHeader(guide.getRange("A6:D6"));
styleBody(guide.getRange("A7:D11"));
guide.getRange("A14:B23").values = [
  ["Metric", "Plain-language meaning"],
  ["P50", "Model A central price forecast in GBP/kg."],
  ["P05-P95", "Model A uncertainty interval containing the central 90% of the forecast distribution."],
  ["Mean absolute deviation", "Average absolute percentage difference between P50 and actual price."],
  ["Within 10%", "Share of P50 forecasts no more than 10% away from actual price."],
  ["Accuracy", "Share of Model B predicted classes matching the actual class."],
  ["Balanced accuracy", "Average recall across the four classes, reducing dominance by common classes."],
  ["Macro F1", "Average class-level precision/recall balance, giving each class equal weight."],
  ["Brier score", "Squared error of the full probability vector. Lower is better."],
  ["Calibration error", "Difference between stated confidence and observed correctness. Lower is better."],
];
styleHeader(guide.getRange("A14:B14"));
styleBody(guide.getRange("A15:B23"));
guide.getRange("A:A").format.columnWidth = 27;
guide.getRange("B:B").format.columnWidth = 66;
guide.getRange("C:C").format.columnWidth = 20;
guide.getRange("D:D").format.columnWidth = 70;
guide.getRange("A6:D23").format.wrapText = true;
guide.getRange("A7:D11").format.rowHeight = 42;
guide.getRange("A15:B23").format.rowHeight = 32;

workbook.recalculate();
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["Summary", "Model A Metrics", "Model B Metrics", "Model A Future", "Model B Future", "Guide"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName.replace(/ /g, "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const summaryCheck = await workbook.inspect({ kind: "table", range: "Summary!A1:R33", include: "values,formulas", tableMaxRows: 35, tableMaxCols: 18 });
console.log(summaryCheck.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
console.log(errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Workbook saved: ${outputPath}`);
