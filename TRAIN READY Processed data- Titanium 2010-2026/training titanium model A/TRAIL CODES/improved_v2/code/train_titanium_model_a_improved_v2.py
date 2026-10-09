from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve()
RUN_DIR = HERE.parents[1]
PARENT_RUN_DIR = RUN_DIR.parent
BASE_SCRIPT = PARENT_RUN_DIR / "code" / "train_titanium_model_a_eruc.py"
SOURCE_DATA_PATH = RUN_DIR / "prepared" / "titanium_model_a_improved_training_2010_2026.csv"

spec = importlib.util.spec_from_file_location("titanium_model_a_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load base pipeline: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


FEATURES = [
    "price_log_change_gbp_1m",
    "price_log_change_gbp_2m",
    "price_log_change_gbp_3m",
    "price_log_change_gbp_6m",
    "price_log_change_gbp_12m",
    "price_volatility_6m",
    "price_volatility_12m",
    "quantity_log_change_1m",
    "quantity_log_change_3m",
    "quantity_log_change_12m",
    "quantity_volatility_6m",
    "quantity_volatility_12m",
    "log_net_weight_kg",
    "volume_ratio_to_prior_12m_median",
    "low_volume_history_flag",
    "supplier_share_usa",
    "supplier_share_japan",
    "supplier_share_france",
    "supplier_share_russia",
    "supplier_share_china",
    "supplier_share_other",
    "supplier_hhi_weight",
    "top_supplier_share_weight",
    "log_partner_count",
    "fx_log_change_1m",
    "fx_log_change_3m",
    "fx_log_change_6m",
    "fx_log_change_12m",
    "fx_volatility_6m",
    "eruc_log_change_1m",
    "eruc_log_change_3m",
    "eruc_log_change_6m",
    "eruc_log_change_12m",
    "eruc_volatility_6m",
    "month_sin",
    "month_cos",
    "price_shock_history_flag",
    "weight_estimated_by_un_flag",
    "eu_collection_change_flag",
]

CANDIDATES = (
    base.Candidate("linear_quantile", "linear", "Weighted regularised linear quantile regression"),
    base.Candidate("gradient_boosting_quantile", "gradient_boosting", "Weighted non-linear quantile boosting"),
    base.Candidate("random_forest_distribution", "random_forest", "Weighted random-forest tree distribution"),
    base.Candidate("elastic_net_residual", "elastic_net", "Weighted Elastic Net median plus residual distribution"),
    base.Candidate("huber_residual", "huber", "Weighted robust Huber median plus residual distribution"),
    base.Candidate("adaptive_no_change", "adaptive_baseline", "Validation-selectable no-change fallback with volatility scaling"),
)

ORIGINAL_PREDICT = base.predict_candidate


def sample_weights(train: pd.DataFrame, horizon: int) -> np.ndarray:
    weights = train["training_sample_weight"].to_numpy(dtype=float).copy()
    target_quality = f"target_quality_flag_h{horizon}"
    if target_quality in train:
        weights *= np.where(train[target_quality].fillna(0).astype(float).gt(0), 0.70, 1.0)
    return np.clip(weights, 0.15, 2.0)


def volatility_reference(train: pd.DataFrame) -> float:
    value = float(train["price_volatility_6m"].median())
    return value if np.isfinite(value) and value > 0 else 0.05


def fit_candidate(candidate: base.Candidate, train: pd.DataFrame, horizon: int) -> dict[str, object]:
    x = train[FEATURES]
    y = train[base.target_column(horizon)].to_numpy(dtype=float)
    weights = sample_weights(train, horizon)

    if candidate.family in {"linear", "gradient_boosting"}:
        models = {}
        for quantile in base.QUANTILES:
            model = base.make_quantile_pipeline(candidate, quantile)
            model.fit(x, y, model__sample_weight=weights)
            models[quantile] = model
        return {
            "family": candidate.family,
            "quantile_models": models,
            "volatility_reference": volatility_reference(train),
        }

    if candidate.family == "random_forest":
        imputer = SimpleImputer(strategy="median")
        x_imputed = imputer.fit_transform(x)
        model = RandomForestRegressor(
            n_estimators=420,
            max_depth=7,
            min_samples_leaf=6,
            max_features=0.70,
            bootstrap=True,
            random_state=base.SEED,
            n_jobs=-1,
        )
        model.fit(x_imputed, y, sample_weight=weights)
        return {
            "family": candidate.family,
            "imputer": imputer,
            "model": model,
            "volatility_reference": volatility_reference(train),
        }

    if candidate.family in {"elastic_net", "huber"}:
        estimator = (
            ElasticNet(alpha=0.025, l1_ratio=0.35, max_iter=30000, random_state=base.SEED)
            if candidate.family == "elastic_net"
            else HuberRegressor(epsilon=1.50, alpha=0.01, max_iter=3000)
        )
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        model.fit(x, y, model__sample_weight=weights)
        residuals = y - model.predict(x)
        residual_quantiles = {
            quantile: float(np.quantile(residuals, quantile, method="linear"))
            for quantile in base.QUANTILES
        }
        return {
            "family": candidate.family,
            "model": model,
            "residual_quantiles": residual_quantiles,
            "volatility_reference": volatility_reference(train),
        }

    if candidate.family == "adaptive_baseline":
        return {
            "family": candidate.family,
            "quantiles": base.baseline_quantiles(train[base.target_column(horizon)]),
            "volatility_reference": volatility_reference(train),
        }
    raise ValueError(candidate.family)


def adaptive_scale(fitted: dict[str, object], row: pd.DataFrame) -> float:
    current = float(row["price_volatility_6m"].iloc[0])
    reference = float(fitted.get("volatility_reference", 0.05))
    if not np.isfinite(current) or current <= 0 or reference <= 0:
        return 1.0
    return float(np.clip(np.sqrt(current / reference), 0.80, 1.80))


def scale_distribution(values: dict[float, float], scale: float) -> dict[float, float]:
    median = float(values[0.50])
    scaled = {
        quantile: median + (float(value) - median) * scale
        for quantile, value in values.items()
    }
    return base.enforce_non_crossing(scaled)


def predict_candidate(fitted: dict[str, object], row: pd.DataFrame) -> dict[float, float]:
    family = str(fitted["family"])
    if family in {"linear", "gradient_boosting"}:
        models = fitted["quantile_models"]
        raw = {
            quantile: float(models[quantile].predict(row[FEATURES])[0])
            for quantile in base.QUANTILES
        }
    elif family == "random_forest":
        x_imputed = fitted["imputer"].transform(row[FEATURES])
        tree_predictions = np.asarray(
            [tree.predict(x_imputed)[0] for tree in fitted["model"].estimators_],
            dtype=float,
        )
        raw = {
            quantile: float(np.quantile(tree_predictions, quantile, method="linear"))
            for quantile in base.QUANTILES
        }
    elif family in {"elastic_net", "huber"}:
        center = float(fitted["model"].predict(row[FEATURES])[0])
        raw = {
            quantile: center + float(fitted["residual_quantiles"][quantile])
            for quantile in base.QUANTILES
        }
    elif family == "adaptive_baseline":
        raw = dict(fitted["quantiles"])
    else:
        raise ValueError(family)
    return scale_distribution(base.enforce_non_crossing(raw), adaptive_scale(fitted, row))


def feature_importance(candidate: base.Candidate, fitted: dict[str, object]) -> dict[str, float]:
    if candidate.family == "linear":
        raw = np.abs(np.asarray(fitted["quantile_models"][0.50].named_steps["model"].coef_))
    elif candidate.family == "gradient_boosting":
        raw = np.asarray(fitted["quantile_models"][0.50].named_steps["model"].feature_importances_)
    elif candidate.family == "random_forest":
        raw = np.asarray(fitted["model"].feature_importances_)
    elif candidate.family in {"elastic_net", "huber"}:
        raw = np.abs(np.asarray(fitted["model"].named_steps["model"].coef_))
    elif candidate.family == "adaptive_baseline":
        raw = np.zeros(len(FEATURES), dtype=float)
    else:
        raise ValueError(candidate.family)
    total = float(raw.sum())
    values = np.zeros_like(raw) if total <= 0 else raw / total * 100.0
    return {feature: float(value) for feature, value in zip(FEATURES, values)}


def configure_base_module() -> None:
    base.RUN_DIR = RUN_DIR
    base.SOURCE_DIR = RUN_DIR
    base.SOURCE_DATA_PATH = SOURCE_DATA_PATH
    base.DATA_DIR = RUN_DIR / "data"
    base.MODELS_DIR = RUN_DIR / "models"
    base.RESULTS_DIR = RUN_DIR / "results"
    base.VIS_DIR = RUN_DIR / "visualizations"
    base.DOCS_DIR = RUN_DIR / "documentation"
    base.DATA_COPY_PATH = base.DATA_DIR / SOURCE_DATA_PATH.name
    base.FEATURES = FEATURES
    base.CANDIDATES = CANDIDATES
    base.fit_candidate = fit_candidate
    base.predict_candidate = predict_candidate
    base.feature_importance = feature_importance
    base.MODEL_COLORS = {
        "linear_quantile": "#0d7c80",
        "gradient_boosting_quantile": "#d05c2b",
        "random_forest_distribution": "#d39b28",
        "elastic_net_residual": "#355c9a",
        "huber_residual": "#8a4f7d",
        "adaptive_no_change": "#54824b",
        base.BASELINE_NAME: "#66757d",
    }


def select_with_validation_gates(comparison: pd.DataFrame) -> dict[tuple[int, int], base.Candidate]:
    lookup = {candidate.name: candidate for candidate in CANDIDATES}
    selected = {}
    comparison["validation_mis_ratio_to_baseline"] = np.nan
    comparison["validation_gate_eligible"] = False
    for hs_code in base.HS_CODES:
        for horizon in base.HORIZONS:
            mask = (comparison["hs_code"] == hs_code) & (comparison["horizon_months"] == horizon)
            group = comparison.loc[mask].copy()
            baseline_mis = float(
                group.loc[group["model"] == base.BASELINE_NAME, "calibrated_mean_interval_score"].iloc[0]
            )
            candidate_mask = mask & comparison["model"].isin(lookup)
            comparison.loc[candidate_mask, "validation_mis_ratio_to_baseline"] = (
                comparison.loc[candidate_mask, "calibrated_mean_interval_score"] / baseline_mis
            )
            comparison.loc[candidate_mask, "validation_gate_eligible"] = (
                (comparison.loc[candidate_mask, "calibrated_p05_p95_coverage"] >= 0.90)
                & (comparison.loc[candidate_mask, "validation_mis_ratio_to_baseline"] < 1.0)
            )
            choices = comparison.loc[candidate_mask]
            gated = choices[choices["validation_gate_eligible"]]
            pool = gated if not gated.empty else choices
            winner = str(pool.sort_values("raw_mean_pinball_loss").iloc[0]["model"])
            selected[(hs_code, horizon)] = lookup[winner]
    comparison["selected_by_pretest_validation"] = comparison.apply(
        lambda row: row["model"]
        == selected.get((int(row["hs_code"]), int(row["horizon_months"])), base.Candidate("", "", "")).name,
        axis=1,
    )
    comparison.sort_values(["hs_code", "horizon_months", "raw_mean_pinball_loss"]).to_csv(
        base.RESULTS_DIR / "pretest_validation_model_comparison.csv", index=False
    )
    return selected


def write_improvement_report(summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    old_summary_path = PARENT_RUN_DIR / "results" / "selected_model_summary.csv"
    old = pd.read_csv(old_summary_path)
    comparison_rows = []
    for row in summary.itertuples(index=False):
        previous = old[(old["hs_code"] == row.hs_code) & (old["horizon_months"] == row.horizon_months)].iloc[0]
        comparison_rows.append(
            {
                "hs_code": row.hs_code,
                "horizon_months": row.horizon_months,
                "previous_model": previous["validation_selected_model"],
                "improved_model": row.validation_selected_model,
                "previous_test_coverage": previous["test_p05_p95_coverage"],
                "improved_test_coverage": row.test_p05_p95_coverage,
                "previous_test_mis_ratio": previous["test_mis_ratio_to_baseline"],
                "improved_test_mis_ratio": row.test_mis_ratio_to_baseline,
                "previous_pass": previous["passes_spec_h3_h6"],
                "improved_pass": row.passes_spec_h3_h6,
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        base.RESULTS_DIR / "previous_vs_improved_selected_models.csv", index=False
    )

    report = f"""# Titanium Model A Improved V2

## Implemented improvements

- Added FX movements and FX volatility.
- Added price and quantity volatility.
- Added import-volume level, relative-volume and low-volume indicators.
- Added supplier concentration, top-supplier share and partner-count features.
- Added seasonality and observed data-quality indicators.
- Down-weighted low-volume, estimated-weight, supplier-anomaly and historical price-shock observations during fitting.
- Added weighted Elastic Net and robust Huber candidates.
- Added volatility-adaptive predictive distributions.
- Added a validation-selectable no-change fallback.
- Selection first requires pre-test validation coverage >=90% and MIS below baseline where possible, then minimises pinball loss.

## Evaluation boundary

The February 2022 to December 2023 disruption test remains frozen and is not used for selection. H3 and H6 are acceptance horizons; H12 remains report-only.

H3/H6 combinations passing every frozen-test rule: **{int(summary[summary['horizon_months'].isin([3, 6])]['passes_spec_h3_h6'].sum())}/4**.

## Remaining specification gaps

World Bank aluminium/nickel changes and the annual USGS titanium sponge series are still absent. They were not fabricated. ERUC remains an experimental broad non-ferrous import-price proxy, so this is not literal full-spec compliance.
"""
    (RUN_DIR / "README.md").write_text(report, encoding="utf-8")


def patch_dashboard() -> None:
    path = base.VIS_DIR / "titanium_model_a_eruc_dashboard.html"
    improved_path = base.VIS_DIR / "titanium_model_a_improved_v2_dashboard.html"
    html = path.read_text(encoding="utf-8")
    html = html.replace("Titanium Model A - ERUC experiment", "Titanium Model A - improved V2")
    html = html.replace("3 + baseline", "6 + baseline")
    html = html.replace("Three model families", "Six validation candidates")
    html = html.replace(
        "Linear quantile regression, gradient-boosted quantile regression and a random-forest tree distribution are compared against a no-change baseline.",
        "Weighted quantile, tree, Elastic Net, Huber and adaptive no-change candidates are compared against the fixed no-change benchmark.",
    )
    html = html.replace(
        "Pre-test residuals provide a split-conformal expansion for the P05-P95 range.",
        "Pre-test residuals provide split-conformal calibration, while forecast-origin volatility adaptively scales the distribution.",
    )
    html = html.replace(
        "this experiment uses ONS ERUC",
        "this improved experiment uses ONS ERUC",
    )
    improved_path.write_text(html, encoding="utf-8")
    path.unlink()


def main() -> None:
    configure_base_module()
    base.ensure_directories()
    shutil.copy2(SOURCE_DATA_PATH, base.DATA_COPY_PATH)
    base.setup_plot_style()
    data = base.load_data()

    print(f"Loaded {len(data)} improved rows")
    print("Running pre-test validation for six candidates...")
    validation_predictions, comparison, calibration, _ = base.run_pretest_validation(data)
    selected = select_with_validation_gates(comparison)
    for key, candidate in sorted(selected.items()):
        print(f"Selected HS {key[0]} H{key[1]}: {candidate.name}")

    print("Running frozen 2022-02 to 2023-12 disruption test...")
    test_predictions, test_metrics = base.run_disruption_test(data, calibration, selected)
    summary = base.save_selected_summary(comparison, test_metrics, selected)
    forecasts, importance, manifest = base.fit_reference_models(data, calibration, selected)

    manifest["model_name"] = "Titanium Model A improved V2"
    manifest["improvements"] = [
        "quality-weighted fitting",
        "FX and volatility features",
        "supplier concentration and volume-regime features",
        "Elastic Net and Huber candidates",
        "volatility-adaptive distributions",
        "validation-gated model selection",
    ]
    (base.RESULTS_DIR / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plot_paths = [
        base.plot_validation_comparison(comparison),
        base.plot_test_coverage(test_metrics),
        base.plot_test_mis(test_metrics),
        base.plot_selected_actual_vs_forecast(test_predictions),
        base.plot_rolling_coverage(test_predictions),
        base.plot_feature_importance(importance),
        base.plot_latest_forecasts(forecasts),
    ]
    base.build_dashboard(summary, forecasts, plot_paths)
    patch_dashboard()
    base.write_methodology()
    write_improvement_report(summary, comparison)

    pass_count = int(summary[summary["horizon_months"].isin([3, 6])]["passes_spec_h3_h6"].sum())
    print("=" * 78)
    print(f"Improved pipeline complete: {pass_count}/4 H3/H6 combinations pass")
    print(f"Results: {base.RESULTS_DIR}")
    print(f"Dashboard: {base.VIS_DIR / 'titanium_model_a_improved_v2_dashboard.html'}")


if __name__ == "__main__":
    main()
