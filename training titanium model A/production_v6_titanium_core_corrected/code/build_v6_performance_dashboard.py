from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
VISUALIZATIONS = ROOT / "visualizations"
OUTPUT = VISUALIZATIONS / "titanium_model_a_v6_performance_dashboard.html"

PRODUCTS = {
    810820: "Unwrought titanium and powders",
    810890: "Bars, sheet, forgings and other articles",
}


def image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def selected_metrics(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data[
        (data["model"] != "no_change_baseline")
        & data["horizon_months"].isin([3, 6])
    ].copy()


def status_badge(value: bool) -> str:
    css_class = "pass" if value else "fail"
    label = "PASS" if value else "FAIL"
    return f'<span class="badge {css_class}">{label}</span>'


def metric_rows(stress: pd.DataFrame, post: pd.DataFrame) -> str:
    merged = stress.merge(
        post,
        on=["hs_code", "horizon_months"],
        suffixes=("_stress", "_post"),
    ).sort_values(["hs_code", "horizon_months"])
    rows = []
    for row in merged.itertuples(index=False):
        model = str(row.model_stress).replace("_", " ").title()
        rows.append(
            "<tr>"
            f"<td><strong>HS {int(row.hs_code)}</strong><small>{PRODUCTS[int(row.hs_code)]}</small></td>"
            f"<td>H{int(row.horizon_months)}</td>"
            f"<td>{model}</td>"
            f"<td>{row.p05_p95_coverage_stress:.1%}</td>"
            f"<td>{row.mis_ratio_to_baseline_stress:.3f}</td>"
            f"<td>{status_badge(bool(row.passes_acceptance_stress))}</td>"
            f"<td>{row.p05_p95_coverage_post:.1%}</td>"
            f"<td>{row.mis_ratio_to_baseline_post:.3f}</td>"
            f"<td>{status_badge(bool(row.passes_acceptance_post))}</td>"
            "</tr>"
        )
    return "".join(rows)


def performance_cards(data: pd.DataFrame, period: str) -> str:
    cards = []
    for row in data.sort_values(["hs_code", "horizon_months"]).itertuples(index=False):
        passed = bool(row.passes_acceptance)
        state = "pass-card" if passed else "fail-card"
        model = str(row.model).replace("_", " ").title()
        cards.append(
            f'<article class="metric-card {state}">'
            f'<div class="eyebrow">HS {int(row.hs_code)} / H{int(row.horizon_months)}</div>'
            f"<h3>{model}</h3>"
            f'<div class="metric-line"><span>Coverage</span><strong>{row.p05_p95_coverage:.1%}</strong></div>'
            f'<div class="bar"><i style="width:{min(row.p05_p95_coverage * 100, 100):.1f}%"></i></div>'
            f'<div class="metric-line"><span>MIS / baseline</span><strong>{row.mis_ratio_to_baseline:.3f}</strong></div>'
            f'<p>{"Beats the baseline interval score." if row.mis_beats_baseline else "Does not beat the baseline interval score."}</p>'
            f"{status_badge(passed)}"
            "</article>"
        )
    return f'<div class="period-label">{period}</div><div class="metric-grid">{"".join(cards)}</div>'


def forecast_rows(forecasts: pd.DataFrame) -> str:
    rows = []
    for row in forecasts.sort_values(["hs_code", "horizon_months"]).itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>HS {int(row.hs_code)}</td>"
            f"<td>H{int(row.horizon_months)}</td>"
            f"<td>{row.target_month}</td>"
            f'<td>{str(row.selected_model).replace("_", " ").title()}</td>'
            f"<td>GBP {row.p50_price_gbp_per_kg:,.2f}/kg</td>"
            f"<td>GBP {row.p05_price_gbp_per_kg:,.2f} to {row.p95_price_gbp_per_kg:,.2f}/kg</td>"
            "</tr>"
        )
    return "".join(rows)


def main() -> None:
    stress = selected_metrics(RESULTS / "stress_2022_2023" / "evaluation_metrics.csv")
    post = selected_metrics(RESULTS / "post_2024_2025" / "evaluation_metrics.csv")
    forecasts = pd.read_csv(RESULTS / "latest" / "latest_reference_forecasts.csv")

    stress_passes = int(stress["passes_acceptance"].sum())
    post_passes = int(post["passes_acceptance"].sum())
    failed_post = post[~post["passes_acceptance"]].iloc[0]

    chart_paths = [
        VISUALIZATIONS / "stress_2022_2023_coverage_and_mis.png",
        VISUALIZATIONS / "post_2024_2025_coverage_and_mis.png",
        VISUALIZATIONS / "post_2024_2025_actual_vs_forecast.png",
        VISUALIZATIONS / "selected_model_feature_importance.png",
        VISUALIZATIONS / "target_reliability_over_time.png",
    ]
    charts = [image_uri(path) for path in chart_paths]

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Titanium Model A V6 performance dashboard</title>
<style>
:root{{--ink:#153b43;--teal:#087d80;--orange:#cf5c2c;--sand:#f3ecdf;--paper:#fffdf8;--line:#d9cebd;--muted:#60747a;--good:#19704b;--bad:#a43c2d}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#f3ecdf 0%,#e4efec 58%,#edf0e9 100%);color:var(--ink);font-family:"Trebuchet MS",sans-serif}}
header{{padding:52px max(5vw,28px) 88px;background:radial-gradient(circle at 88% 15%,#1a6970 0,#153b43 34%,#102e35 78%);color:#fff;position:relative;overflow:hidden}}
header:after{{content:"";position:absolute;width:330px;height:330px;border:1px solid #ffffff2e;border-radius:50%;right:-70px;top:-125px;box-shadow:0 0 0 55px #ffffff0b,0 0 0 110px #ffffff08}}
.kicker,.eyebrow,.period-label{{text-transform:uppercase;letter-spacing:.14em;font-size:.76rem;font-weight:700}}h1,h2{{font-family:Georgia,serif;font-weight:500}}h1{{font-size:clamp(2.6rem,5vw,5.2rem);max-width:1050px;margin:.2em 0 .22em;line-height:1.02}}header p{{max-width:930px;color:#d4e3e1;font-size:1.08rem;line-height:1.65}}
main{{max-width:1540px;margin:auto;padding:0 28px 60px}}.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:-54px;position:relative;z-index:2}}.summary article,section{{background:var(--paper);border:1px solid var(--line);box-shadow:0 14px 35px #153b4315}}.summary article{{padding:22px}}.summary span{{display:block;color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.1em}}.summary strong{{display:block;font:2.25rem Georgia,serif;color:var(--teal);margin-top:8px}}
section{{padding:30px;margin-top:24px}}h2{{font-size:2rem;margin:0 0 16px}}.lede{{font-size:1.05rem;line-height:1.65;max-width:1100px}}.callout{{border-left:5px solid var(--orange);background:#fff1e6;padding:16px 20px;line-height:1.55;margin:20px 0}}
.period-label{{margin:24px 0 10px;color:var(--orange)}}.metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.metric-card{{border:1px solid var(--line);padding:18px;background:#fff}}.metric-card.pass-card{{border-top:5px solid var(--good)}}.metric-card.fail-card{{border-top:5px solid var(--bad)}}.metric-card h3{{min-height:42px;font-size:1rem;margin:8px 0 16px}}.metric-line{{display:flex;justify-content:space-between;margin:9px 0}}.metric-line span{{color:var(--muted)}}.bar{{height:7px;background:#e7e1d7;border-radius:8px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--teal)}}.metric-card p{{min-height:42px;color:var(--muted);font-size:.88rem;line-height:1.4}}
.badge{{display:inline-block;padding:6px 10px;border-radius:3px;font-weight:700;font-size:.74rem;letter-spacing:.08em}}.badge.pass{{background:#d9eee2;color:#165e40}}.badge.fail{{background:#f5ddd5;color:#8b2f22}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;min-width:980px}}th{{background:var(--ink);color:#fff;text-align:left;padding:11px;font-size:.78rem;letter-spacing:.04em}}td{{padding:11px;border-bottom:1px solid var(--line);font-size:.9rem}}td small{{display:block;color:var(--muted);margin-top:4px}}
.visual-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0;background:#fff;border:1px solid var(--line);padding:10px}}figure.wide{{grid-column:1/-1}}figure img{{width:100%;display:block}}figcaption{{padding:9px;color:var(--muted);font-size:.85rem}}
.decision-grid{{display:grid;grid-template-columns:1.2fr .8fr;gap:20px}}.checklist{{margin:0;padding-left:20px;line-height:1.8}}footer{{max-width:1540px;margin:auto;padding:0 28px 50px;color:var(--muted);font-size:.85rem}}
@media(max-width:1050px){{.summary,.metric-grid{{grid-template-columns:1fr 1fr}}.visual-grid,.decision-grid{{grid-template-columns:1fr}}figure.wide{{grid-column:auto}}}}
@media(max-width:650px){{header{{padding:38px 22px 48px}}main{{padding:0 14px 40px}}.summary{{grid-template-columns:1fr;margin-top:16px}}.metric-grid{{grid-template-columns:1fr}}section{{padding:20px}}}}
@media print{{body{{background:#fff}}header{{padding:28px;background:var(--ink)}}.summary{{margin-top:10px}}section,.summary article{{box-shadow:none;break-inside:avoid}}}}
</style>
</head>
<body>
<header><div class="kicker">SCI Model A / corrected research evaluation</div><h1>Titanium import-price forecast performance</h1><p>V6 evaluates direct H3, H6 and H12 GBP/kg forecasts for UK titanium imports. It excludes USGS and World Bank cross-metal inputs and uses independent calibration histories for the selected models and the fixed no-change baseline.</p></header>
<main>
<div class="summary">
<article><span>Disruption passes</span><strong>{stress_passes}/4</strong></article>
<article><span>Post-period passes</span><strong>{post_passes}/4</strong></article>
<article><span>Required coverage</span><strong>90%</strong></article>
<article><span>Data through</span><strong>Jun 2026</strong></article>
</div>
<section><h2>Current assessment</h2><p class="lede">V6 passes all required H3/H6 combinations during the 2022-2023 disruption stress test. Three of four combinations also pass the 2024-2025 post-disruption review. This is a strong research result, but the model remains a shadow-deployment candidate until a genuinely prospective holdout is available.</p><div class="callout"><strong>Remaining issue:</strong> HS {int(failed_post.hs_code)} H{int(failed_post.horizon_months)} has {failed_post.p05_p95_coverage:.1%} post-period coverage, but its MIS ratio is {failed_post.mis_ratio_to_baseline:.3f}. Its uncertainty interval is therefore {abs(failed_post.mis_ratio_to_baseline - 1) * 100:.1f}% worse than the no-change baseline even though the central forecast error is lower.</div></section>
<section><h2>Acceptance overview</h2>{performance_cards(stress,"2022-2023 disruption stress test")}{performance_cards(post,"2024-2025 post-disruption evaluation")}</section>
<section><h2>Detailed comparison</h2><div class="table-wrap"><table><thead><tr><th>Product</th><th>Horizon</th><th>Selected model</th><th>Stress coverage</th><th>Stress MIS ratio</th><th>Stress result</th><th>Post coverage</th><th>Post MIS ratio</th><th>Post result</th></tr></thead><tbody>{metric_rows(stress,post)}</tbody></table></div></section>
<section><h2>Latest reference forecasts</h2><p class="lede">These forecasts originate from June 2026. H12 remains report-only and is not part of the formal H3/H6 acceptance count.</p><div class="table-wrap"><table><thead><tr><th>Product</th><th>Horizon</th><th>Target month</th><th>Model</th><th>P50 forecast</th><th>P05-P95 range</th></tr></thead><tbody>{forecast_rows(forecasts)}</tbody></table></div></section>
<section><h2>Evaluation evidence</h2><div class="visual-grid"><figure><img src="{charts[0]}" alt="Disruption coverage and MIS"><figcaption>2022-2023 disruption coverage and MIS ratios.</figcaption></figure><figure><img src="{charts[1]}" alt="Post-period coverage and MIS"><figcaption>2024-2025 post-disruption coverage and MIS ratios.</figcaption></figure><figure class="wide"><img src="{charts[2]}" alt="Actual versus forecast"><figcaption>Post-disruption actual prices, median forecasts and P05-P95 intervals.</figcaption></figure><figure class="wide"><img src="{charts[3]}" alt="Feature importance"><figcaption>Feature importance for each selected model. The adaptive no-change model has no learned feature weights.</figcaption></figure><figure class="wide"><img src="{charts[4]}" alt="Target reliability"><figcaption>Historical target-reliability diagnostics by HS code.</figcaption></figure></div></section>
<section><div class="decision-grid"><div><h2>What the tests show</h2><ul class="checklist"><li>The model produced every required H3/H6 forecast.</li><li>All selected H3/H6 intervals met the 90% coverage requirement.</li><li>All four disruption combinations beat the independently calibrated baseline MIS.</li><li>HS 810890 H3 is the only post-period MIS failure.</li><li>HS 810890 remains harder because the code combines heterogeneous fabricated titanium products.</li></ul></div><div><h2>Decision</h2><p class="lede"><strong>Recommended status:</strong> research candidate for shadow monitoring.</p><p>Do not tune further against the already examined 2022-2025 periods. Lock V6 and evaluate newly observed months as a prospective holdout before production approval.</p></div></div></section>
</main>
<footer>Source: Titanium Model A V6 corrected evaluation outputs. MIS ratios below 1.0 outperform the fixed no-change baseline. Importance values are predictive associations, not causal effects.</footer>
</body></html>"""

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Saved dashboard: {OUTPUT}")


if __name__ == "__main__":
    main()
