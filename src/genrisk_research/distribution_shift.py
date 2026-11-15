"""Descriptive source-level distribution shift analysis for UCI Heart Disease."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from .uci_heart_disease import (
    CATEGORICAL_FEATURES,
    DEVELOPMENT_SOURCES,
    EXTERNAL_SOURCE,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    duplicate_audit,
    load_uci_heart_sources,
    repository_root,
)

SOURCES = (*DEVELOPMENT_SOURCES, EXTERNAL_SOURCE)
OUTPUT_SUBDIRECTORY = "distribution_shift"
ANALYSIS_SENTINEL_VALUES: dict[str, tuple[float, ...]] = {
    # The UCI documentation identifies -9 as a missing-value code. The
    # processed sources also contain impossible zero values for measured
    # cholesterol and resting blood pressure; those are treated as missing
    # for this descriptive analysis only. The loader/model harmonisation is
    # intentionally unchanged.
    "age": (-9.0,),
    "trestbps": (0.0, -9.0),
    "chol": (0.0, -9.0),
    "thalach": (-9.0,),
}


def _float_or_none(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def _standardized_mean_difference(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    if len(values_a) == 0 or len(values_b) == 0:
        return None
    mean_difference = float(np.mean(values_a) - np.mean(values_b))
    if len(values_a) + len(values_b) <= 2:
        return None
    variance_a = float(np.var(values_a, ddof=1)) if len(values_a) > 1 else 0.0
    variance_b = float(np.var(values_b, ddof=1)) if len(values_b) > 1 else 0.0
    pooled_variance = ((len(values_a) - 1) * variance_a + (len(values_b) - 1) * variance_b) / (len(values_a) + len(values_b) - 2)
    pooled_sd = float(np.sqrt(pooled_variance))
    if pooled_sd == 0:
        return 0.0 if mean_difference == 0 else None
    return mean_difference / pooled_sd


def prepare_distribution_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis copy with documented impossible sentinel values missing."""

    prepared = data.copy(deep=True)
    for feature, sentinels in ANALYSIS_SENTINEL_VALUES.items():
        prepared.loc[prepared[feature].isin(sentinels), feature] = np.nan
    return prepared


def sentinel_audit(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Report raw sentinel counts without modifying the supplied frame."""

    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        source_data = data[data["source"].eq(source)]
        for feature, sentinels in ANALYSIS_SENTINEL_VALUES.items():
            for sentinel in sentinels:
                count = int(source_data[feature].eq(sentinel).sum())
                if count:
                    rows.append({"source": source, "feature": feature, "sentinel_value": sentinel, "count": count, "treatment": "treated_as_missing_for_distribution_analysis_only"})
    return rows


def _js_divergence(proportions_a: dict[str, float], proportions_b: dict[str, float]) -> float | None:
    categories = sorted(set(proportions_a) | set(proportions_b))
    if not categories:
        return None
    a = np.asarray([proportions_a.get(category, 0.0) for category in categories], dtype=float)
    b = np.asarray([proportions_b.get(category, 0.0) for category in categories], dtype=float)
    midpoint = (a + b) / 2.0

    def entropy(values: np.ndarray) -> float:
        positive = values > 0
        return float(-np.sum(values[positive] * np.log2(values[positive])))

    return float(entropy(midpoint) - (entropy(a) + entropy(b)) / 2.0)


def feature_summaries(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise each selected feature separately for each source."""

    data = prepare_distribution_data(data)
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        source_data = data[data["source"].eq(source)]
        for feature in FEATURES:
            series = source_data[feature]
            observed = series.dropna()
            row: dict[str, Any] = {
                "source": source,
                "feature": feature,
                "feature_type": "numeric" if feature in NUMERIC_FEATURES else "categorical",
                "sample_count": int(len(series)),
                "observed_count": int(len(observed)),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": float(series.isna().mean() * 100.0) if len(series) else None,
                "mean": None,
                "standard_deviation": None,
                "median": None,
                "q1": None,
                "q3": None,
                "interquartile_range": None,
                "minimum": None,
                "maximum": None,
                "category_frequencies_json": None,
                "category_proportions_json": None,
            }
            if feature in NUMERIC_FEATURES and len(observed):
                row.update({
                    "mean": float(observed.mean()),
                    "standard_deviation": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
                    "median": float(observed.median()),
                    "q1": float(observed.quantile(0.25)),
                    "q3": float(observed.quantile(0.75)),
                    "interquartile_range": float(observed.quantile(0.75) - observed.quantile(0.25)),
                    "minimum": float(observed.min()),
                    "maximum": float(observed.max()),
                })
            elif feature in CATEGORICAL_FEATURES and len(observed):
                counts = observed.value_counts(dropna=False).sort_index()
                proportions = observed.value_counts(normalize=True, dropna=False).sort_index()
                row["category_frequencies_json"] = json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True)
                row["category_proportions_json"] = json.dumps({str(k): float(v) for k, v in proportions.items()}, sort_keys=True)
            rows.append(row)
    return pd.DataFrame(rows)


def _observed_category_proportions(data: pd.DataFrame, source: str, feature: str) -> dict[str, float]:
    observed = data.loc[data["source"].eq(source), feature].dropna()
    if observed.empty:
        return {}
    proportions = observed.value_counts(normalize=True, dropna=False)
    return {str(category): float(value) for category, value in proportions.items()}


def pairwise_feature_distances(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate effect sizes and distances for every source pair and feature."""

    data = prepare_distribution_data(data)
    rows: list[dict[str, Any]] = []
    for source_a, source_b in combinations(SOURCES, 2):
        comparison_type = "development_development" if source_a in DEVELOPMENT_SOURCES and source_b in DEVELOPMENT_SOURCES else "development_external"
        for feature in FEATURES:
            series_a = data.loc[data["source"].eq(source_a), feature]
            series_b = data.loc[data["source"].eq(source_b), feature]
            row: dict[str, Any] = {
                "source_a": source_a,
                "source_b": source_b,
                "comparison_type": comparison_type,
                "feature": feature,
                "feature_type": "numeric" if feature in NUMERIC_FEATURES else "categorical",
                "observed_count_a": int(series_a.notna().sum()),
                "observed_count_b": int(series_b.notna().sum()),
                "standardized_mean_difference": None,
                "kolmogorov_smirnov_statistic": None,
                "wasserstein_distance": None,
                "scale_normalized_wasserstein_distance": None,
                "total_variation_distance": None,
                "jensen_shannon_divergence": None,
            }
            if feature in NUMERIC_FEATURES:
                values_a = series_a.dropna().to_numpy(dtype=float)
                values_b = series_b.dropna().to_numpy(dtype=float)
                if len(values_a) and len(values_b):
                    row["standardized_mean_difference"] = _standardized_mean_difference(values_a, values_b)
                    row["kolmogorov_smirnov_statistic"] = float(ks_2samp(values_a, values_b, method="auto").statistic)
                    distance = float(wasserstein_distance(values_a, values_b))
                    row["wasserstein_distance"] = distance
                    pooled_variance = ((len(values_a) - 1) * np.var(values_a, ddof=1) + (len(values_b) - 1) * np.var(values_b, ddof=1)) / (len(values_a) + len(values_b) - 2) if len(values_a) + len(values_b) > 2 else 0.0
                    pooled_sd = float(np.sqrt(pooled_variance))
                    row["scale_normalized_wasserstein_distance"] = distance / pooled_sd if pooled_sd > 0 else (0.0 if distance == 0 else None)
            else:
                proportions_a = _observed_category_proportions(data, source_a, feature)
                proportions_b = _observed_category_proportions(data, source_b, feature)
                categories = sorted(set(proportions_a) | set(proportions_b))
                if categories:
                    vector_a = np.asarray([proportions_a.get(category, 0.0) for category in categories])
                    vector_b = np.asarray([proportions_b.get(category, 0.0) for category in categories])
                    row["total_variation_distance"] = float(0.5 * np.abs(vector_a - vector_b).sum())
                    row["jensen_shannon_divergence"] = _js_divergence(proportions_a, proportions_b)
            rows.append(row)
    return pd.DataFrame(rows)


def missingness_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    data = prepare_distribution_data(data)
    rows: list[dict[str, Any]] = []
    for source_a, source_b in combinations(SOURCES, 2):
        comparison_type = "development_development" if source_a in DEVELOPMENT_SOURCES and source_b in DEVELOPMENT_SOURCES else "development_external"
        for feature in FEATURES:
            series_a = data.loc[data["source"].eq(source_a), feature]
            series_b = data.loc[data["source"].eq(source_b), feature]
            percentage_a = float(series_a.isna().mean() * 100.0) if len(series_a) else None
            percentage_b = float(series_b.isna().mean() * 100.0) if len(series_b) else None
            rows.append({
                "source_a": source_a,
                "source_b": source_b,
                "comparison_type": comparison_type,
                "feature": feature,
                "missing_count_a": int(series_a.isna().sum()),
                "missing_count_b": int(series_b.isna().sum()),
                "missing_percentage_a": percentage_a,
                "missing_percentage_b": percentage_b,
                "absolute_missing_percentage_difference": abs(percentage_a - percentage_b) if percentage_a is not None and percentage_b is not None else None,
            })
    return pd.DataFrame(rows)


def target_prevalence_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prevalence: dict[str, float] = {}
    for source in SOURCES:
        source_target = data.loc[data["source"].eq(source), TARGET].dropna()
        positive = int((source_target == 1).sum())
        negative = int((source_target == 0).sum())
        value = float(positive / len(source_target)) if len(source_target) else None
        prevalence[source] = value if value is not None else np.nan
        rows.append({"row_type": "source", "source": source, "source_a": source, "source_b": None, "comparison_type": None, "observation_count": int(len(source_target)), "positive_count": positive, "negative_count": negative, "positive_prevalence": value, "absolute_prevalence_difference": None})
    for source_a, source_b in combinations(SOURCES, 2):
        rows.append({"row_type": "pairwise", "source": None, "source_a": source_a, "source_b": source_b, "comparison_type": "development_development" if source_a in DEVELOPMENT_SOURCES and source_b in DEVELOPMENT_SOURCES else "development_external", "observation_count": None, "positive_count": None, "negative_count": None, "positive_prevalence": None, "absolute_prevalence_difference": abs(prevalence[source_a] - prevalence[source_b])})
    return pd.DataFrame(rows)


def _write_comparison_svg(path: Path, distances: pd.DataFrame, missingness: pd.DataFrame, prevalence: pd.DataFrame) -> None:
    width, height = 1040, 640
    body = [f'<rect width="{width}" height="{height}" fill="white"/>', '<text x="25" y="28" font-family="sans-serif" font-size="18">Source distribution-shift diagnostics</text>']
    panels = [("Numeric |SMD|", "standardized_mean_difference"), ("Numeric KS", "kolmogorov_smirnov_statistic"), ("Categorical JS", "jensen_shannon_divergence"), ("Missingness pp", "absolute_missing_percentage_difference")]
    for panel_index, (title, column) in enumerate(panels):
        x0 = 35 + (panel_index % 2) * 500
        y0 = 75 + (panel_index // 2) * 270
        body.append(f'<text x="{x0}" y="{y0}" font-family="sans-serif" font-size="14">{title}</text>')
        if column == "absolute_missing_percentage_difference":
            values = missingness.groupby("feature")[column].max().dropna().sort_values(ascending=False).head(7)
        else:
            values = distances.groupby("feature")[column].apply(lambda series: float(series.abs().max()) if series.notna().any() else np.nan).dropna().sort_values(ascending=False).head(7)
        max_value = max(float(values.max()), 1e-12) if len(values) else 1.0
        for index, (feature, value) in enumerate(values.items()):
            bar_width = float(value) / max_value * 390
            y = y0 + 22 + index * 25
            body.append(f'<text x="{x0}" y="{y + 13}" font-family="sans-serif" font-size="10">{feature}</text>')
            body.append(f'<rect x="{x0 + 75}" y="{y}" width="{bar_width}" height="16" fill="#1f77b4"/>')
            body.append(f'<text x="{x0 + 80 + bar_width}" y="{y + 13}" font-family="sans-serif" font-size="10">{float(value):.3f}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(body) + "</svg>\n", encoding="utf-8")


def _write_report(output: Path, summary: dict[str, Any]) -> None:
    largest_numeric = summary["largest_numeric_differences"]
    largest_categorical = summary["largest_categorical_differences"]
    largest_missingness = summary["largest_missingness_differences"]
    prevalence = summary["target_prevalence_by_source"]
    lines = [
        "# Source-level distribution-shift analysis",
        "",
        "This analysis describes observable differences across the four UCI Heart Disease sources. It separates observed feature distributions, missingness, and target prevalence. It does not establish causes of model deterioration or concept shift.",
        "",
        "## Definitions and method",
        "",
        "For numeric features, the analysis reports source summaries, standardised mean difference SMD = (mean_A - mean_B) / pooled_SD, where pooled_SD = sqrt(((n_A - 1)s_A^2 + (n_B - 1)s_B^2) / (n_A + n_B - 2)), two-sample Kolmogorov–Smirnov statistic, Wasserstein distance, and Wasserstein distance divided by pooled_SD when that scale is non-zero. For categorical features, it reports category frequencies/proportions, total variation distance, and Jensen–Shannon divergence using base-2 logarithms. Missing values are excluded only from observed-value distribution metrics and are measured separately.",
        "",
        "If pooled_SD is zero, SMD is 0 when the two means are equal and otherwise undefined; the normalised Wasserstein distance follows the same zero-scale rule. If there are insufficient observations to estimate sample variances, the standardised metrics are undefined.",
        "",
        "The raw loader converts documented `?` tokens to missing values. This analysis additionally treats impossible zero cholesterol and zero resting-blood-pressure values, and documented -9 codes, as missing for descriptive summaries only. The raw data, loader, model inputs, and previous experiment outputs are unchanged. Zero remains a valid category code for selected categorical variables.",
        "",
        "Covariate shift refers to changes in P(X); prevalence or label shift refers to changes in P(Y); concept shift refers to changes in P(Y|X). This analysis directly measures aspects of P(X), missingness, and P(Y). It cannot establish P(Y|X) changes.",
        "",
        "## Largest observed differences",
        "",
        "### Numeric features",
        "",
        "| Comparison | Feature | Absolute SMD | KS statistic | Wasserstein | Normalised Wasserstein |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in largest_numeric[:8]:
        lines.append(f"| {row['source_a']} vs {row['source_b']} | {row['feature']} | {abs(row['standardized_mean_difference']):.4f} | {row['kolmogorov_smirnov_statistic']:.4f} | {row['wasserstein_distance']:.4f} | {row['scale_normalized_wasserstein_distance']:.4f} |")
    lines.extend(["", "### Categorical features", "", "| Comparison | Feature | TV distance | JS divergence |", "|---|---|---:|---:|"])
    for row in largest_categorical[:8]:
        lines.append(f"| {row['source_a']} vs {row['source_b']} | {row['feature']} | {row['total_variation_distance']:.4f} | {row['jensen_shannon_divergence']:.4f} |")
    lines.extend(["", "### Missingness", "", "| Comparison | Feature | Absolute difference (percentage points) |", "|---|---|---:|"])
    for row in largest_missingness[:8]:
        lines.append(f"| {row['source_a']} vs {row['source_b']} | {row['feature']} | {row['absolute_missing_percentage_difference']:.2f} |")
    lines.extend(["", "### Target prevalence", "", "| Source | Observations | Negative | Positive | Positive prevalence |", "|---|---:|---:|---:|---:|"])
    for row in prevalence:
        lines.append(f"| {row['source']} | {row['observation_count']} | {row['negative_count']} | {row['positive_count']} | {row['positive_prevalence']:.4f} |")
    lines.extend([
        "", "## Interpretation", "",
        "The measurements establish source-associated covariate and missingness differences, and the prevalence table establishes outcome-composition differences. They are relevant candidate explanations for why performance may vary across sources, but they do not identify which difference caused deterioration. External-source comparisons are descriptive and were not used for model fitting or selection.",
        "", "## Limitations", "",
        "The data is historical and the sources differ in population, measurement, missingness, and label processes. Standardised distances can be undefined for zero-variance features; scale-normalised Wasserstein distance is undefined when the pooled standard deviation is zero. Effect sizes and distances are descriptive and no p-values are used as evidence. These analyses do not establish concept shift, causality, or clinical validity.",
        "",
        "The extreme raw cholesterol contrast between Cleveland and Switzerland is not a valid observed-cholesterol contrast: all 123 Switzerland cholesterol values are zero, while the Cleveland values are positive. The zero values are therefore treated as missing here; the analysis reports no cholesterol distribution distance for that pair rather than interpreting the sentinel as measured cholesterol.",
    ])
    (output / "distribution_shift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _performance_shift_context(
    output: Path,
    summaries: pd.DataFrame,
    distances: pd.DataFrame,
    missingness: pd.DataFrame,
    prevalence: pd.DataFrame,
) -> None:
    """Join descriptive shift summaries to already-saved model evaluations."""

    diagnostics = repository_root() / "results" / "uci_heart_disease" / "diagnostics"
    source_performance = json.loads((diagnostics / "source_performance.json").read_text(encoding="utf-8"))
    reliability = json.loads((diagnostics / "reliability_analysis.json").read_text(encoding="utf-8"))
    prevalence_by_source = {
        row["source"]: row for row in prevalence[prevalence["row_type"].eq("source")].to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        source_pairs = distances[(distances["source_a"].eq(source) | distances["source_b"].eq(source))]
        source_pairs = source_pairs[
            source_pairs.apply(
                lambda row: (row["source_b"] if row["source_a"] == source else row["source_a"]) in DEVELOPMENT_SOURCES,
                axis=1,
            )
        ]
        numeric_pairs = source_pairs[source_pairs["feature_type"].eq("numeric")]
        categorical_pairs = source_pairs[source_pairs["feature_type"].eq("categorical")]
        missing_pairs = missingness[
            missingness["source_a"].eq(source) | missingness["source_b"].eq(source)
        ]
        missing_pairs = missing_pairs[
            missing_pairs.apply(
                lambda row: (row["source_b"] if row["source_a"] == source else row["source_a"]) in DEVELOPMENT_SOURCES,
                axis=1,
            )
        ]
        source_summary = summaries[summaries["source"].eq(source)]
        shifts = {
            "maximum_absolute_numeric_smd": float(numeric_pairs["standardized_mean_difference"].abs().max()) if numeric_pairs["standardized_mean_difference"].notna().any() else None,
            "maximum_categorical_js_divergence": float(categorical_pairs["jensen_shannon_divergence"].max()) if categorical_pairs["jensen_shannon_divergence"].notna().any() else None,
            "maximum_missingness_difference_percentage_points": float(missing_pairs["absolute_missing_percentage_difference"].max()) if missing_pairs["absolute_missing_percentage_difference"].notna().any() else None,
        }
        shift_reference = "development_sources" if source == EXTERNAL_SOURCE else "other_development_sources"
        for model in ("logistic_regression", "random_forest"):
            if source in DEVELOPMENT_SOURCES:
                performance = source_performance["experiment_a_existing_predictions"][source][model]
                evaluation_scope = "existing repeated random-split predictions; rows are not independent"
                reliability_scope = "source-specific Brier only; source-specific calibration curve unavailable"
                calibration = reliability["evaluations"]["A_random_group_stratified"][model]
                brier = performance["metrics"]["brier_score"]
                prevalence_brier = prevalence_by_source[source]["positive_prevalence"] * (1.0 - prevalence_by_source[source]["positive_prevalence"])
            else:
                performance = source_performance["external_va_existing_predictions"][model]
                evaluation_scope = "existing external Long Beach VA predictions"
                reliability_scope = "external VA calibration curve and prevalence baselines"
                calibration = reliability["evaluations"]["B_external_va"][model]
                brier = performance["metrics"]["brier_score"]
                prevalence_brier = calibration["baselines"]["prevalence_only_brier"]
            rows.append({
                "source": source,
                "model": model,
                "evaluation_scope": evaluation_scope,
                "observation_count": prevalence_by_source[source]["observation_count"],
                "positive_count": prevalence_by_source[source]["positive_count"],
                "negative_count": prevalence_by_source[source]["negative_count"],
                "positive_prevalence": prevalence_by_source[source]["positive_prevalence"],
                "feature_shift_reference": shift_reference,
                "maximum_absolute_numeric_smd": shifts["maximum_absolute_numeric_smd"],
                "maximum_categorical_js_divergence": shifts["maximum_categorical_js_divergence"],
                "maximum_missingness_difference_percentage_points": shifts["maximum_missingness_difference_percentage_points"],
                "missingness_percentage_by_feature_json": json.dumps({row["feature"]: row["missing_percentage"] for row in source_summary.to_dict(orient="records")}, sort_keys=True),
                "roc_auc": performance["metrics"]["roc_auc"],
                "pr_auc": performance["metrics"].get("pr_auc"),
                "balanced_accuracy": performance["metrics"]["balanced_accuracy"],
                "sensitivity": performance["metrics"]["sensitivity"],
                "specificity": performance["metrics"]["specificity"],
                "brier_score": brier,
                "prevalence_only_brier": prevalence_brier,
                "mean_predicted_probability": calibration.get("probability_summary", {}).get("mean"),
                "calibration_scope": reliability_scope,
            })
    pd.DataFrame(rows).to_csv(output / "source_performance_shift_context.csv", index=False)


def run_distribution_shift_analysis(output_dir: Path | None = None) -> dict[str, Any]:
    """Execute the source-level analysis without changing source data or model outputs."""

    output = output_dir or repository_root() / "results" / "uci_heart_disease" / OUTPUT_SUBDIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    data = load_uci_heart_sources()
    analysis_data = prepare_distribution_data(data)
    summaries = feature_summaries(analysis_data)
    distances = pairwise_feature_distances(analysis_data)
    missingness = missingness_comparisons(analysis_data)
    prevalence_frame = target_prevalence_comparisons(analysis_data)
    summaries.to_csv(output / "source_feature_summaries.csv", index=False)
    distances.to_csv(output / "pairwise_feature_distances.csv", index=False)
    missingness.to_csv(output / "missingness_comparison.csv", index=False)
    prevalence_frame.to_csv(output / "target_prevalence_comparison.csv", index=False)

    numeric = distances[distances["feature_type"].eq("numeric")].dropna(subset=["standardized_mean_difference"])
    categorical = distances[distances["feature_type"].eq("categorical")].dropna(subset=["jensen_shannon_divergence"])
    missing_ranked = missingness.dropna(subset=["absolute_missing_percentage_difference"]).sort_values("absolute_missing_percentage_difference", ascending=False)
    prevalence_source = prevalence_frame[prevalence_frame["row_type"].eq("source")]
    summary = {
        "sources": list(SOURCES),
        "development_sources": list(DEVELOPMENT_SOURCES),
        "external_source": EXTERNAL_SOURCE,
        "raw_duplicate_audit": duplicate_audit(data),
        "raw_sentinel_audit": sentinel_audit(data),
        "analysis_missing_value_rules": {feature: list(values) for feature, values in ANALYSIS_SENTINEL_VALUES.items()},
        "features": {feature: {"type": "numeric" if feature in NUMERIC_FEATURES else "categorical"} for feature in FEATURES},
        "definitions": {
            "standardized_mean_difference": "difference in means divided by pooled standard deviation; undefined when the pooled standard deviation is zero and the means differ",
            "kolmogorov_smirnov_statistic": "maximum absolute difference between empirical cumulative distributions",
            "wasserstein_distance": "minimum transport cost between one-dimensional observed-value distributions",
            "scale_normalized_wasserstein_distance": "Wasserstein distance divided by pooled standard deviation when positive",
            "total_variation_distance": "one half of the sum of absolute category-proportion differences",
            "jensen_shannon_divergence": "symmetric, smoothed divergence of category proportions using base-2 logarithms",
            "missingness": "percentage of source records with missing feature values; excluded from observed-value summaries",
            "target_prevalence": "positive target count divided by non-missing target observations",
        },
        "largest_numeric_differences": numeric.assign(_absolute=numeric["standardized_mean_difference"].abs()).sort_values("_absolute", ascending=False).drop(columns=["_absolute"]).head(20).to_dict(orient="records"),
        "largest_categorical_differences": categorical.sort_values("jensen_shannon_divergence", ascending=False).head(20).to_dict(orient="records"),
        "largest_missingness_differences": missing_ranked.head(20).to_dict(orient="records"),
        "target_prevalence_by_source": prevalence_source.to_dict(orient="records"),
        "target_prevalence_pairwise": prevalence_frame[prevalence_frame["row_type"].eq("pairwise")].to_dict(orient="records"),
    }
    summary = _json_value(summary)
    (output / "distribution_shift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_comparison_svg(output / "distribution_shift_comparison.svg", distances, missingness, prevalence_frame)
    _write_report(output, summary)
    _performance_shift_context(output, summaries, distances, missingness, prevalence_frame)
    return summary


if __name__ == "__main__":
    run_distribution_shift_analysis()
