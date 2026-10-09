from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(__file__).with_name("titanium_uk_imports_data_dashboard.html")
HS_LABELS = {
    "810820": "Unwrought titanium and powders",
    "810890": "Other titanium articles",
}


def read_export(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="latin1", index_col=False, low_memory=False)
    frame["source_file"] = path.name
    return frame


def choose_authoritative_files(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    for path in paths:
        frame = read_export(path)
        periods = pd.to_numeric(frame["period"], errors="coerce").dropna().astype(int)
        if periods.empty:
            continue
        for year in sorted(set(periods // 100)):
            year_rows = frame.loc[(pd.to_numeric(frame["period"], errors="coerce") // 100) == year]
            codes = year_rows["cmdCode"].astype(str).str.replace(r"\.0$", "", regex=True)
            for code in sorted(set(codes) & set(HS_LABELS)):
                code_rows = year_rows.loc[codes.eq(code)]
                candidates.append(
                    {
                        "year": int(year),
                        "hs_code": code,
                        "path": path,
                        "months": int(code_rows["period"].nunique()),
                        "rows": int(len(code_rows)),
                    }
                )

    selected: list[dict] = []
    audit: list[dict] = []
    groups = sorted({(item["year"], item["hs_code"]) for item in candidates})
    for year, code in groups:
        options = [item for item in candidates if item["year"] == year and item["hs_code"] == code]
        winner = max(options, key=lambda item: (item["months"], item["rows"], item["path"].stat().st_mtime))
        selected.append(winner)
        for item in options:
            audit.append(
                {
                    "year": year,
                    "hs_code": code,
                    "file": item["path"].name,
                    "months": item["months"],
                    "rows": item["rows"],
                    "status": "selected" if item["path"] == winner["path"] else "excluded: less complete",
                }
            )
    return selected, audit


def prepare_data(selections: list[dict]):
    frames = []
    for selection in selections:
        frame = read_export(selection["path"])
        period = pd.to_numeric(frame["period"], errors="coerce")
        code = frame["cmdCode"].astype(str).str.replace(r"\.0$", "", regex=True)
        frames.append(frame.loc[(period // 100).eq(selection["year"]) & code.eq(selection["hs_code"])])
    raw = pd.concat(frames, ignore_index=True)
    raw["period"] = pd.to_numeric(raw["period"], errors="coerce").astype("Int64")
    raw["cmdCode"] = raw["cmdCode"].astype(str).str.replace(r"\.0$", "", regex=True)
    raw = raw.loc[raw["cmdCode"].isin(HS_LABELS) & raw["period"].notna()].copy()
    for column in ["primaryValue", "netWgt", "qty"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["date"] = pd.to_datetime(raw["period"].astype(str), format="%Y%m")

    key = ["period", "cmdCode", "partnerCode", "partner2Code", "flowCode", "customsCode", "motCode"]
    duplicate_count = int(raw.duplicated(key, keep=False).sum())
    raw = raw.sort_values(["period", "cmdCode", "partnerCode", "source_file"]).drop_duplicates(key, keep="last")

    world = raw.loc[raw["partnerCode"].eq(0)].copy()
    partners = raw.loc[~raw["partnerCode"].eq(0)].copy()

    partner_month = (
        partners.groupby(["period", "date", "cmdCode"], as_index=False)
        .agg(partner_value=("primaryValue", "sum"), partner_weight=("netWgt", "sum"))
    )
    monthly = world[["period", "date", "cmdCode", "primaryValue", "netWgt", "isNetWgtEstimated"]].merge(
        partner_month, on=["period", "date", "cmdCode"], how="outer"
    )
    monthly["value_usd"] = monthly["primaryValue"].fillna(monthly["partner_value"])
    needs_weight = monthly["netWgt"].isna() | monthly["netWgt"].le(0)
    monthly["weight_kg"] = monthly["netWgt"]
    monthly.loc[needs_weight, "weight_kg"] = monthly.loc[needs_weight, "partner_weight"]
    monthly["weight_source"] = np.where(needs_weight, "partner sum", "world row")
    monthly["unit_value_usd_per_kg"] = monthly["value_usd"] / monthly["weight_kg"].replace(0, np.nan)
    monthly["value_gap_pct"] = (
        (monthly["partner_value"] - monthly["primaryValue"]) / monthly["primaryValue"].replace(0, np.nan) * 100
    )
    monthly = monthly.sort_values(["cmdCode", "date"])
    monthly["rolling_12m_value_usd"] = monthly.groupby("cmdCode")["value_usd"].transform(
        lambda series: series.rolling(12, min_periods=6).sum()
    )

    supplier = (
        partners.groupby(["period", "date", "cmdCode", "partnerCode", "partnerDesc"], as_index=False)
        .agg(value_usd=("primaryValue", "sum"), weight_kg=("netWgt", "sum"))
    )
    supplier["month_total"] = supplier.groupby(["period", "cmdCode"])["value_usd"].transform("sum")
    supplier["share"] = supplier["value_usd"] / supplier["month_total"].replace(0, np.nan)
    hhi = (
        supplier.assign(square_share=lambda frame: frame["share"] ** 2)
        .groupby(["period", "date", "cmdCode"], as_index=False)["square_share"]
        .sum()
        .rename(columns={"square_share": "hhi"})
    )
    monthly = monthly.merge(hhi, on=["period", "date", "cmdCode"], how="left")

    return raw, monthly, supplier, duplicate_count


def records(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> list[dict]:
    output = frame[columns].copy()
    for column in output.select_dtypes(include=["float", "float64", "float32"]).columns:
        output[column] = output[column].round(digits)
    output = output.replace({np.nan: None, pd.NA: None})
    return output.to_dict(orient="records")


def build_payload(raw, monthly, supplier, duplicate_count, file_audit):
    first_year = int(monthly["date"].dt.year.min())
    last_year = int(monthly["date"].dt.year.max())
    last_month = monthly["date"].max()
    expected_periods = pd.period_range(f"{first_year}-01", last_month.to_period("M"), freq="M")

    coverage = []
    missing = []
    for code in HS_LABELS:
        for year in range(first_year, last_year + 1):
            available = set(monthly.loc[monthly["date"].dt.year.eq(year) & monthly["cmdCode"].eq(code), "date"].dt.month)
            max_month = int(last_month.month) if year == last_year else 12
            coverage.append({"year": year, "hs_code": code, "months": [month in available for month in range(1, 13)], "expected": max_month})
    for code in HS_LABELS:
        observed = set(monthly.loc[monthly["cmdCode"].eq(code), "date"].dt.to_period("M"))
        missing.extend(
            {"hs_code": code, "period": str(period)} for period in expected_periods if period not in observed
        )

    annual = (
        monthly.assign(year=monthly["date"].dt.year)
        .groupby(["year", "cmdCode"], as_index=False)
        .agg(
            months=("period", "nunique"),
            value_usd=("value_usd", "sum"),
            weight_kg=("weight_kg", "sum"),
            reconstructed_months=("weight_source", lambda values: int((values == "partner sum").sum())),
        )
    )
    annual["weighted_unit_value"] = annual["value_usd"] / annual["weight_kg"].replace(0, np.nan)

    complete_cutoff = pd.Timestamp(year=last_year - 1, month=12, day=1) if last_month.month < 12 else last_month
    supplier_start = complete_cutoff - pd.DateOffset(months=35)
    top_suppliers = (
        supplier.loc[supplier["date"].between(supplier_start, complete_cutoff)]
        .groupby(["cmdCode", "partnerDesc"], as_index=False)["value_usd"]
        .sum()
    )
    top_suppliers["share"] = top_suppliers["value_usd"] / top_suppliers.groupby("cmdCode")["value_usd"].transform("sum")
    top_suppliers = top_suppliers.sort_values(["cmdCode", "value_usd"], ascending=[True, False]).groupby("cmdCode").head(8)

    partner_checks = supplier.copy()
    partner_checks["unit_value"] = partner_checks["value_usd"] / partner_checks["weight_kg"].replace(0, np.nan)
    partner_checks["partner_median_weight"] = partner_checks.groupby(["cmdCode", "partnerDesc"])["weight_kg"].transform("median")
    partner_checks["weight_multiple"] = partner_checks["weight_kg"] / partner_checks["partner_median_weight"].replace(0, np.nan)
    anomalies = partner_checks.loc[
        (partner_checks["weight_kg"] > 0)
        & ((partner_checks["unit_value"] < 1) | (partner_checks["unit_value"] > 1000))
        & (partner_checks["weight_multiple"] > 10)
    ].sort_values("weight_multiple", ascending=False).head(12)

    series = monthly.copy()
    series["date_label"] = series["date"].dt.strftime("%Y-%m")
    series["value_musd"] = series["value_usd"] / 1_000_000
    series["weight_tonnes"] = series["weight_kg"] / 1_000
    series["rolling_12m_value_musd"] = series["rolling_12m_value_usd"] / 1_000_000

    latest_complete_year = last_year - 1 if last_month.month < 12 else last_year
    latest_annual = annual.loc[annual["year"].eq(latest_complete_year)]
    total_latest_value = float(latest_annual["value_usd"].sum())
    reconstructed = int((monthly["weight_source"] == "partner sum").sum())
    max_value_gap = float(monthly["value_gap_pct"].abs().max(skipna=True))

    return {
        "meta": {
            "first_period": monthly["date"].min().strftime("%B %Y"),
            "last_period": last_month.strftime("%B %Y"),
            "year_count": last_year - first_year + 1,
            "selected_files": len({item["file"] for item in file_audit if item["status"] == "selected"}),
            "raw_rows": int(len(raw)),
            "monthly_rows": int(len(monthly)),
            "latest_complete_year": latest_complete_year,
            "latest_complete_value_musd": round(total_latest_value / 1_000_000, 2),
            "reconstructed_weights": reconstructed,
            "duplicate_rows_removed": duplicate_count,
            "missing_code_months": len(missing),
            "max_value_gap_pct": round(max_value_gap, 6),
        },
        "labels": HS_LABELS,
        "series": records(
            series,
            ["date_label", "cmdCode", "value_musd", "weight_tonnes", "unit_value_usd_per_kg", "rolling_12m_value_musd", "hhi", "weight_source"],
        ),
        "annual": records(annual, ["year", "cmdCode", "months", "value_usd", "weight_kg", "weighted_unit_value", "reconstructed_months"]),
        "coverage": coverage,
        "missing": missing,
        "top_suppliers": records(top_suppliers, ["cmdCode", "partnerDesc", "value_usd", "share"]),
        "anomalies": records(
            anomalies.assign(date_label=anomalies["date"].dt.strftime("%Y-%m")),
            ["date_label", "cmdCode", "partnerDesc", "value_usd", "weight_kg", "unit_value", "weight_multiple"],
        ),
        "file_audit": file_audit,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UK Titanium Imports Data Review</title>
<style>
:root{--ink:#17333a;--muted:#64777b;--paper:#fbfaf5;--sand:#eee7d8;--line:#d7cebc;--teal:#0f7c82;--orange:#cf5b28;--red:#a63d2f;--green:#2b7650;--navy:#123b52}
*{box-sizing:border-box} body{margin:0;background:#e9efec;color:var(--ink);font-family:Georgia,"Times New Roman",serif}
.hero{background:linear-gradient(120deg,#0e3946 0%,#145967 58%,#d36a34 58%,#b94824 100%);color:white;padding:54px max(28px,calc((100vw - 1440px)/2));position:relative;overflow:hidden}
.hero:after{content:"";position:absolute;width:410px;height:410px;border:1px solid rgba(255,255,255,.18);border-radius:50%;right:7%;top:-210px}
.eyebrow{font:700 12px/1.3 Consolas,monospace;letter-spacing:.22em;text-transform:uppercase;color:#f6c8aa}.hero h1{font-size:clamp(38px,5vw,72px);line-height:.98;max-width:900px;margin:14px 0}.hero p{font:17px/1.55 Consolas,monospace;max-width:900px;color:#deeaeb}
main{max-width:1440px;margin:auto;padding:30px}.grid{display:grid;gap:18px}.cards{grid-template-columns:repeat(6,minmax(0,1fr));margin-top:-58px;position:relative}.card,.panel{background:var(--paper);border:1px solid var(--line);box-shadow:0 13px 34px rgba(23,51,58,.08)}
.card{padding:20px;min-height:128px}.card .k{font:700 11px Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--orange)}.card .v{font-size:31px;font-weight:700;margin:11px 0 5px}.card .s{font:12px/1.45 Consolas,monospace;color:var(--muted)}
.section{margin-top:34px}.section-head{display:flex;justify-content:space-between;align-items:end;gap:24px;margin-bottom:13px}.section h2{font-size:34px;margin:0}.section-head p{font:13px/1.5 Consolas,monospace;color:var(--muted);max-width:680px;margin:0}.two{grid-template-columns:1fr 1fr}.panel{padding:22px;min-width:0}.panel h3{font-size:22px;margin:0 0 5px}.sub{font:12px/1.45 Consolas,monospace;color:var(--muted);margin-bottom:14px}.chart{width:100%;height:330px}.legend{display:flex;gap:18px;flex-wrap:wrap;font:12px Consolas,monospace;margin:8px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}
table{width:100%;border-collapse:collapse;font:12px/1.4 Consolas,monospace}th{text-align:left;text-transform:uppercase;letter-spacing:.08em;font-size:10px;color:var(--muted);border-bottom:2px solid var(--ink);padding:9px 8px}td{padding:8px;border-bottom:1px solid var(--line);vertical-align:top}.scroll{overflow:auto;max-height:430px}.num{text-align:right}.status{font-weight:700}.ok{color:var(--green)}.warn{color:var(--red)}
.coverage{display:grid;grid-template-columns:65px repeat(12,1fr);gap:4px;align-items:center;font:11px Consolas,monospace}.coverage .month{text-align:center;color:var(--muted)}.cell{height:18px;border-radius:2px;background:#d9ded9}.cell.y{background:var(--teal)}.cell.partial{background:#eab08b}.note{padding:15px 17px;border-left:4px solid var(--orange);background:#f4eadf;font:13px/1.55 Consolas,monospace}.ready{display:grid;grid-template-columns:1fr 1fr;gap:15px}.check{padding:18px;border:1px solid var(--line);background:#fff}.check h4{margin:0 0 8px;font-size:18px}.check p{font:13px/1.55 Consolas,monospace;color:var(--muted);margin:0}.footer{font:11px/1.5 Consolas,monospace;color:var(--muted);padding:28px 0 60px}
@media(max-width:1050px){.cards{grid-template-columns:repeat(3,1fr)}.two{grid-template-columns:1fr}}@media(max-width:650px){main{padding:18px}.cards{grid-template-columns:1fr 1fr;margin-top:-35px}.hero{padding:40px 22px 75px}.ready{grid-template-columns:1fr}.section-head{display:block}.section-head p{margin-top:8px}.chart{height:290px}}
</style>
</head>
<body>
<header class="hero"><div class="eyebrow">Model A input review / UN Comtrade</div><h1>UK titanium monthly imports</h1><p id="subtitle"></p></header>
<main>
<section class="grid cards" id="cards"></section>

<section class="section"><div class="section-head"><div><div class="eyebrow">Coverage</div><h2>What data is available?</h2></div><p>Coverage is shown separately by HS code. 2026 is intentionally partial because the source currently ends in June.</p></div><div class="panel"><div class="grid two" id="coverage"></div><div class="note" id="coverage-note" style="margin-top:18px"></div></div></section>

<section class="section"><div class="section-head"><div><div class="eyebrow">Scale</div><h2>Value and import volume</h2></div><p>National totals use UN Comtrade World-partner rows. Missing or zero World net weights are reconstructed from the corresponding individual partner rows and remain flagged.</p></div><div class="grid two"><div class="panel"><h3>Monthly import value</h3><div class="sub">USD millions. Raw UN Comtrade currency.</div><div class="legend"></div><svg class="chart" id="value-chart"></svg></div><div class="panel"><h3>Monthly net weight</h3><div class="sub">Metric tonnes. Log scale is used because shipments vary greatly.</div><div class="legend"></div><svg class="chart" id="weight-chart"></svg></div></div></section>

<section class="section"><div class="section-head"><div><div class="eyebrow">Price proxy</div><h2>Declared value per kilogram</h2></div><p>This is trade value divided by net weight, not a market benchmark price. Product mix, processing stage, contract timing and weight anomalies can move it sharply.</p></div><div class="grid two"><div class="panel"><h3>Monthly unit value</h3><div class="sub">USD/kg. Use as a model target only after quality controls.</div><div class="legend"></div><svg class="chart" id="price-chart"></svg></div><div class="panel"><h3>Rolling 12-month import value</h3><div class="sub">USD millions, reducing monthly shipment noise.</div><div class="legend"></div><svg class="chart" id="rolling-chart"></svg></div></div></section>

<section class="section"><div class="section-head"><div><div class="eyebrow">Suppliers</div><h2>Country dependence</h2></div><p>Supplier shares use individual partner-country rows. The concentration index is HHI: values closer to 1 indicate greater dependence on fewer suppliers.</p></div><div class="grid two"><div class="panel"><h3>Monthly supplier concentration</h3><div class="sub">HHI based on import value shares.</div><div class="legend"></div><svg class="chart" id="hhi-chart"></svg></div><div class="panel"><h3>Top suppliers</h3><div class="sub">Share of import value over the latest three complete years.</div><div id="supplier-bars"></div></div></div></section>

<section class="section"><div class="section-head"><div><div class="eyebrow">Annual view</div><h2>Yearly totals by HS code</h2></div><p>2026 is year to date and should not be compared directly with complete years.</p></div><div class="panel scroll"><table id="annual-table"></table></div></section>

<section class="section"><div class="section-head"><div><div class="eyebrow">Quality review</div><h2>Checks and exceptions</h2></div><p>Exceptions are retained for investigation. The dashboard does not silently delete or overwrite source records.</p></div><div class="grid two"><div class="panel"><h3>Source-file selection</h3><div class="sub">The most complete export is selected independently for each year and HS code.</div><div class="scroll"><table id="file-table"></table></div></div><div class="panel"><h3>Potential partner-level anomalies</h3><div class="sub">Large weight multiples combined with implausible unit values.</div><div class="scroll"><table id="anomaly-table"></table></div></div></div></section>

<section class="section"><div class="section-head"><div><div class="eyebrow">Readiness</div><h2>Is the dataset ready for Model A?</h2></div></div><div class="ready"><div class="check"><h4 id="coverage-readiness-title"></h4><p id="coverage-readiness-copy"></p></div><div class="check"><h4 class="warn">Weight anomalies: unresolved</h4><p>Review flagged partner records, especially Italy / HS 810890 / July 2011, before deriving price targets. Train sensitivity variants with the record retained, corrected only from authoritative evidence, or excluded with an audit flag.</p></div><div class="check"><h4 class="warn">GBP conversion: still required</h4><p>UN Comtrade trade value is USD. Join a time-correct monthly USD-to-GBP exchange-rate series before calculating the final GBP/kg target.</p></div><div class="check"><h4 class="warn">External predictors: still required</h4><p>Add World Bank monthly nickel and aluminium prices, USGS annual titanium sponge price with its publication lag, and explicit source-availability dates. These are required by the Model A specification.</p></div></div></section>
<div class="footer">Generated locally from the CSV files in this folder. Raw source files were not modified. Calculations are descriptive and are not forecasts.</div>
</main>
<script>
const D=__PAYLOAD__;
const COLORS={"810820":"#0f7c82","810890":"#cf5b28"};
const fmt=(n,d=1)=>n==null?"N/A":Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
document.getElementById('subtitle').textContent=`${D.meta.first_period} to ${D.meta.last_period}. Two HS6 product series, supplier detail and source-quality controls.`;
const cards=[['Period',`${D.meta.first_period}<br><span style="font-size:18px">to ${D.meta.last_period}</span>`,'2010 is the earliest downloaded observation'],['Selected files',D.meta.selected_files,'One authoritative export per year'],['Monthly observations',D.meta.monthly_rows,'HS-code/month combinations present'],['Missing code-months',D.meta.missing_code_months,'Absence is not automatically a zero'],[`${D.meta.latest_complete_year} import value`,`$${fmt(D.meta.latest_complete_value_musd,1)}m`,'Both HS codes combined'],['Rebuilt weights',D.meta.reconstructed_weights,'World weight replaced by partner sum']];
document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><div class="k">${x[0]}</div><div class="v">${x[1]}</div><div class="s">${x[2]}</div></div>`).join('');
const readinessTitle=document.getElementById('coverage-readiness-title');
const readinessCopy=document.getElementById('coverage-readiness-copy');
if(D.meta.missing_code_months===0){readinessTitle.className='ok';readinessTitle.textContent='Trade coverage: complete';readinessCopy.textContent=`Both HS-code series contain every expected month from ${D.meta.first_period} through ${D.meta.last_period}. The final year remains year to date.`}else{readinessTitle.className='warn';readinessTitle.textContent='Trade coverage: incomplete';readinessCopy.textContent=`${D.meta.missing_code_months} expected HS-code/month observations are missing and should be resolved before final training.`}

const cov=document.getElementById('coverage');
cov.innerHTML=Object.entries(D.labels).map(([code,label])=>{
 const rows=D.coverage.filter(r=>r.hs_code===code);
 let cells='<div></div>'+['J','F','M','A','M','J','J','A','S','O','N','D'].map(m=>`<div class="month">${m}</div>`).join('');
 rows.forEach(r=>{cells+=`<b>${r.year}</b>`+r.months.map((yes,i)=>`<div class="cell ${yes?'y':(i<r.expected?'':'partial')}" title="${code} / ${r.year}-${String(i+1).padStart(2,'0')}: ${yes?'available':i<r.expected?'missing':'not yet expected'}"></div>`).join('')});
 return `<div><h3>HS ${code}</h3><div class="sub">${label}</div><div class="coverage">${cells}</div></div>`;
}).join('');
document.getElementById('coverage-note').innerHTML=D.meta.missing_code_months?`The calendar is covered, but <b>${D.meta.missing_code_months} HS-code/month observations are absent</b>. Treat these as missing until source verification shows they represent zero trade.`:'Both HS-code series are present in every expected month.';

document.querySelectorAll('.legend').forEach(el=>el.innerHTML=Object.entries(D.labels).map(([c,l])=>`<span><i class="dot" style="background:${COLORS[c]}"></i>${c}: ${l}</span>`).join(''));
function lineChart(id,field,{log=false,suffix='',decimals=1}={}){
 const svg=document.getElementById(id),W=780,H=330,m={l:66,r:18,t:18,b:42},iw=W-m.l-m.r,ih=H-m.t-m.b;
 const rows=D.series.filter(r=>r[field]!=null&&(!log||r[field]>0)); const dates=[...new Set(rows.map(r=>r.date_label))].sort();
 let vals=rows.map(r=>log?Math.log10(r[field]):r[field]); let lo=Math.min(...vals),hi=Math.max(...vals); if(lo===hi){lo-=1;hi+=1} const pad=(hi-lo)*.07;lo-=pad;hi+=pad;
 const x=d=>m.l+dates.indexOf(d)/(Math.max(1,dates.length-1))*iw, y=v=>m.t+(hi-(log?Math.log10(v):v))/(hi-lo)*ih;
 let out=`<g font-family="Consolas,monospace" font-size="11" fill="#64777b">`;
 for(let i=0;i<5;i++){const yy=m.t+i*ih/4,raw=hi-i*(hi-lo)/4,val=log?10**raw:raw;out+=`<line x1="${m.l}" x2="${W-m.r}" y1="${yy}" y2="${yy}" stroke="#d7cebc"/><text x="${m.l-10}" y="${yy+4}" text-anchor="end">${fmt(val,decimals)}${suffix}</text>`}
 [0,.25,.5,.75,1].forEach(p=>{const idx=Math.round(p*(dates.length-1)),xx=m.l+p*iw;out+=`<text x="${xx}" y="${H-12}" text-anchor="middle">${dates[idx]}</text>`});out+='</g>';
 Object.keys(D.labels).forEach(code=>{const pts=rows.filter(r=>r.cmdCode===code).map(r=>`${x(r.date_label)},${y(r[field])}`).join(' ');out+=`<polyline fill="none" stroke="${COLORS[code]}" stroke-width="2.2" points="${pts}"/>`});svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=out;
}
lineChart('value-chart','value_musd',{suffix:'m',decimals:1});lineChart('weight-chart','weight_tonnes',{log:true,suffix:'t',decimals:0});lineChart('price-chart','unit_value_usd_per_kg',{suffix:'',decimals:0});lineChart('rolling-chart','rolling_12m_value_musd',{suffix:'m',decimals:0});lineChart('hhi-chart','hhi',{decimals:2});

function bars(){const root=document.getElementById('supplier-bars');root.innerHTML=Object.keys(D.labels).map(code=>{const rows=D.top_suppliers.filter(r=>r.cmdCode===code);return `<div style="margin:10px 0 22px"><b>${code}</b><div class="sub">${D.labels[code]}</div>${rows.map(r=>`<div style="display:grid;grid-template-columns:145px 1fr 52px;gap:8px;align-items:center;font:11px Consolas,monospace;margin:7px 0"><span>${r.partnerDesc}</span><span style="height:13px;background:#e4ded2;display:block"><i style="display:block;height:100%;width:${Math.max(1,r.share*100)}%;background:${COLORS[code]}"></i></span><span class="num">${fmt(r.share*100,1)}%</span></div>`).join('')}</div>`}).join('')}bars();

function table(id,heads,rows){document.getElementById(id).innerHTML=`<thead><tr>${heads.map(h=>`<th>${h[0]}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${heads.map(h=>`<td class="${h[2]||''}">${h[1](r)}</td>`).join('')}</tr>`).join('')}</tbody>`}
table('annual-table',[['Year',r=>r.year],['HS',r=>r.cmdCode],['Months',r=>r.months,'num'],['Value (USD)',r=>'$'+fmt(r.value_usd,0),'num'],['Weight (kg)',r=>fmt(r.weight_kg,0),'num'],['Weighted USD/kg',r=>fmt(r.weighted_unit_value,2),'num'],['Rebuilt weights',r=>r.reconstructed_months,'num']],D.annual);
table('file-table',[['Year',r=>r.year],['HS',r=>r.hs_code],['File',r=>r.file],['Months',r=>r.months,'num'],['Rows',r=>r.rows,'num'],['Decision',r=>`<span class="status ${r.status==='selected'?'ok':'warn'}">${r.status}</span>`]],D.file_audit);
table('anomaly-table',[['Month',r=>r.date_label],['HS',r=>r.cmdCode],['Partner',r=>r.partnerDesc],['Value USD',r=>'$'+fmt(r.value_usd,0),'num'],['Weight kg',r=>fmt(r.weight_kg,0),'num'],['USD/kg',r=>fmt(r.unit_value,3),'num'],['Weight / median',r=>fmt(r.weight_multiple,1)+'x','num']],D.anomalies);
</script>
</body></html>'''


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    selected, file_audit = choose_authoritative_files(csv_files)
    raw, monthly, supplier, duplicate_count = prepare_data(selected)
    payload = build_payload(raw, monthly, supplier, duplicate_count, file_audit)
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard: {OUTPUT_PATH}")
    print(json.dumps(payload["meta"], indent=2))


if __name__ == "__main__":
    main()
