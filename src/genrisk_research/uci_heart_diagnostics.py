"""Phase 2 diagnostics for the completed UCI Heart Disease study.

The primary experiment artifacts are read-only inputs. Diagnostics are written
to a separate ``results/uci_heart_disease/diagnostics`` directory and do not
overwrite the original study outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from .uci_heart_disease import (
    CATEGORICAL_FEATURES,
    DEVELOPMENT_SOURCES,
    EXTERNAL_SOURCE,
    FEATURES,
    METRIC_NAMES,
    TARGET,
    classification_metrics,
    development_data,
    duplicate_audit,
    external_data,
    fit_model,
    load_uci_heart_sources,
    repeated_group_stratified_splits,
    repository_root,
)

NUMERIC_FEATURES = ("age", "trestbps", "chol", "thalach")
MODEL_NAMES = ("logistic_regression", "random_forest")
PRIMARY_RESULTS = repository_root() / "results" / "uci_heart_disease"
DIAGNOSTICS_RESULTS = PRIMARY_RESULTS / "diagnostics"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2), encoding="utf-8")


def standardized_mean_difference(left: pd.Series, right: pd.Series) -> float | None:
    left_values = pd.to_numeric(left, errors="coerce").dropna().to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").dropna().to_numpy(dtype=float)
    if len(left_values) == 0 or len(right_values) == 0:
        return None
    pooled_sd = np.sqrt((left_values.var(ddof=1) + right_values.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0 if left_values.mean() == right_values.mean() else None
    return float((left_values.mean() - right_values.mean()) / pooled_sd)


def jensen_shannon_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    categories = sorted(set(left) | set(right))
    p = np.asarray([left.get(category, 0.0) for category in categories], dtype=float)
    q = np.asarray([right.get(category, 0.0) for category in categories], dtype=float)
    midpoint = (p + q) / 2
    p_terms = np.zeros_like(p)
    q_terms = np.zeros_like(q)
    p_mask = p > 0
    q_mask = q > 0
    p_terms[p_mask] = p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask])
    q_terms[q_mask] = q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask])
    return float((p_terms.sum() + q_terms.sum()) / 2)


def _category_proportions(series: pd.Series) -> dict[str, float]:
    values = series.astype(object).where(series.notna(), "__MISSING__").astype(str)
    return {str(key): float(value) for key, value in values.value_counts(normalize=True).sort_index().items()}


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    return {
        "count": int(values.notna().sum()),
        "missing_count": int(values.isna().sum()),
        "mean": float(values.mean()) if values.notna().any() else None,
        "std": float(values.std(ddof=1)) if values.notna().sum() > 1 else None,
        "median": float(values.median()) if values.notna().any() else None,
        "q1": float(values.quantile(0.25)) if values.notna().any() else None,
        "q3": float(values.quantile(0.75)) if values.notna().any() else None,
    }


def distribution_shift_analysis(data: pd.DataFrame) -> dict[str, Any]:
    development = development_data(data)
    external = external_data(data)
    reports: dict[str, Any] = {"numeric": {}, "categorical": {}, "comparisons": {}}
    for feature in NUMERIC_FEATURES:
        reports["numeric"][feature] = {
            source: _numeric_summary(frame[feature])
            for source, frame in data.groupby("source", sort=False)
        }
        reports["comparisons"][feature] = {
            "va_vs_pooled_development_smd": standardized_mean_difference(external[feature], development[feature]),
            "development_sources_vs_pooled_smd": {
                source: standardized_mean_difference(frame[feature], development[feature])
                for source, frame in development.groupby("source", sort=False)
            },
        }
    for feature in CATEGORICAL_FEATURES:
        source_props = {
            source: _category_proportions(frame[feature])
            for source, frame in data.groupby("source", sort=False)
        }
        pooled_props = _category_proportions(development[feature])
        reports["categorical"][feature] = source_props
        reports["comparisons"][feature] = {
            "va_vs_pooled_development_absolute_differences": {
                category: abs(source_props[EXTERNAL_SOURCE].get(category, 0.0) - pooled_props.get(category, 0.0))
                for category in sorted(set(source_props[EXTERNAL_SOURCE]) | set(pooled_props))
            },
            "va_vs_pooled_development_js_divergence": jensen_shannon_divergence(
                source_props[EXTERNAL_SOURCE], pooled_props
            ),
            "development_source_pairwise_js_divergence": {
                f"{left}_vs_{right}": jensen_shannon_divergence(
                    _category_proportions(development[development.source == left][feature]),
                    _category_proportions(development[development.source == right][feature]),
                )
                for index, left in enumerate(DEVELOPMENT_SOURCES)
                for right in DEVELOPMENT_SOURCES[index + 1:]
            },
        }
    reports["interpretation"] = (
        "Distribution differences are descriptive associations between source and feature values. "
        "They do not establish that any feature shift caused external performance deterioration."
    )
    return reports


def missingness_analysis(data: pd.DataFrame) -> dict[str, Any]:
    reports: dict[str, Any] = {"by_source": {}, "patterns": {}, "largest_va_vs_development_differences": {}}
    development = development_data(data)
    external = external_data(data)
    for source, frame in data.groupby("source", sort=False):
        missing = frame[list(FEATURES)].isna()
        reports["by_source"][source] = {
            "row_count": int(len(frame)),
            "feature_missing_counts": {feature: int(missing[feature].sum()) for feature in FEATURES},
            "feature_missing_percentages": {feature: float(missing[feature].mean() * 100) for feature in FEATURES},
            "complete_record_count": int((~missing.any(axis=1)).sum()),
            "complete_record_percentage": float((~missing.any(axis=1)).mean() * 100),
            "mean_missing_features_per_record": float(missing.sum(axis=1).mean()),
        }
        pattern_counts = {}
        for pattern, count in missing.groupby(list(FEATURES), dropna=False).size().items():
            pattern = pattern if isinstance(pattern, tuple) else (pattern,)
            label = "+".join(
                feature for feature, is_missing in zip(FEATURES, pattern) if is_missing
            ) or "complete"
            pattern_counts[label] = int(count)
        reports["patterns"][source] = {"pattern_counts": pattern_counts}
    pooled_missing = development[list(FEATURES)].isna().mean() * 100
    va_missing = external[list(FEATURES)].isna().mean() * 100
    reports["largest_va_vs_development_differences"] = {
        feature: float(abs(va_missing[feature] - pooled_missing[feature]))
        for feature in sorted(FEATURES, key=lambda item: abs(va_missing[item] - pooled_missing[item]), reverse=True)
    }
    reports["interpretation"] = (
        "Missingness differences identify possible measurement or collection-process differences. "
        "They are not evidence of causal effects on model performance."
    )
    return reports


def _prevalence_references(y: pd.Series) -> dict[str, float]:
    prevalence = float(y.mean())
    return {
        "prevalence": prevalence,
        "prevalence_only_brier": prevalence * (1.0 - prevalence),
        "prevalence_pr_auc_baseline": prevalence,
    }


def _calibration_curve(y: pd.Series, probabilities: pd.Series, bins: int = 10) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"y": y.to_numpy(), "p": probabilities.to_numpy()}).sort_values("p")
    if frame.empty:
        return []
    bin_count = min(bins, len(frame))
    frame["bin"] = np.minimum((np.arange(len(frame)) * bin_count) // len(frame), bin_count - 1)
    result = []
    for bin_number, bin_id in enumerate(frame["bin"].drop_duplicates(), start=1):
        values = frame.loc[frame["bin"] == bin_id]
        result.append({
            "bin": bin_number,
            "count": int(len(values)),
            "mean_predicted_probability": float(values["p"].mean()),
            "observed_positive_rate": float(values["y"].mean()),
        })
    return result


def reliability_analysis(predictions: pd.DataFrame) -> dict[str, Any]:
    reports: dict[str, Any] = {"evaluations": {}}
    for experiment, frame in predictions.groupby("experiment", sort=False):
        reports["evaluations"].setdefault(experiment, {})
        for model, model_frame in frame.groupby("model", sort=False):
            y = model_frame["true_target"].astype(int)
            p = model_frame["predicted_probability"].astype(float)
            metrics = classification_metrics(y, p)
            reports["evaluations"][experiment][model] = {
                "record_count": int(len(model_frame)),
                "unique_record_count": int(model_frame["record_id"].nunique()),
                "probability_summary": {
                    "mean": float(p.mean()), "std": float(p.std(ddof=1)),
                    "median": float(p.median()), "q1": float(p.quantile(0.25)), "q3": float(p.quantile(0.75)),
                },
                "metrics": metrics,
                "baselines": _prevalence_references(y),
                "calibration_curve": _calibration_curve(y, p),
            }
    reports["interpretation"] = (
        "The reliability analysis is descriptive. The saved probabilities were not recalibrated and VA was not used for tuning."
    )
    return reports


def source_performance(predictions: pd.DataFrame) -> dict[str, Any]:
    reports: dict[str, Any] = {"experiment_a_existing_predictions": {}}
    frame = predictions[predictions["experiment"] == "A_random_group_stratified"]
    for source, source_frame in frame.groupby("source", sort=False):
        reports["experiment_a_existing_predictions"].setdefault(source, {})
        for model, model_frame in source_frame.groupby("model", sort=False):
            reports["experiment_a_existing_predictions"][source][model] = {
                "prediction_row_count": int(len(model_frame)),
                "unique_record_count": int(model_frame["record_id"].nunique()),
                "repeated_split_rows_are_not_independent": True,
                "metrics": classification_metrics(
                    model_frame["true_target"], model_frame["predicted_probability"]
                ),
            }
    reports["external_va_existing_predictions"] = reliability_analysis(
        predictions[predictions["experiment"] == "B_external_va"]
    )["evaluations"]["B_external_va"]
    return reports


def complete_case_sensitivity(data: pd.DataFrame, seeds: Iterable[int]) -> dict[str, Any]:
    complete = data.dropna(subset=list(FEATURES)).reset_index(drop=True)
    development = development_data(complete)
    external = external_data(complete)
    results: dict[str, Any] = {
        "analysis_population": {
            "all_rows": int(len(data)),
            "complete_case_rows": int(len(complete)),
            "excluded_for_sensitivity_only": int(len(data) - len(complete)),
            "source_counts": complete.groupby("source").size().astype(int).to_dict(),
            "class_counts": complete[TARGET].value_counts().sort_index().astype(int).to_dict(),
        },
        "experiment_a": [],
        "experiment_b": {},
    }
    splits = repeated_group_stratified_splits(development, seeds)
    for model in MODEL_NAMES:
        for seed, train_index, test_index in splits:
            fitted = fit_model(model, development.iloc[train_index], random_state=seed)
            test = development.iloc[test_index]
            probabilities = fitted.predict_proba(test[list(FEATURES)])[:, 1]
            results["experiment_a"].append({
                "model": model, "seed": seed, **classification_metrics(test[TARGET], probabilities)
            })
        fitted = fit_model(model, development, random_state=42)
        probabilities = fitted.predict_proba(external[list(FEATURES)])[:, 1]
        results["experiment_b"][model] = {
            "metrics": classification_metrics(external[TARGET], probabilities),
            "external_record_count": int(len(external)),
            "external_class_counts": external[TARGET].value_counts().sort_index().astype(int).to_dict(),
        }
    summary: dict[str, Any] = {}
    for model in MODEL_NAMES:
        runs = [run for run in results["experiment_a"] if run["model"] == model]
        summary[model] = {}
        for metric in METRIC_NAMES:
            values = [run[metric] for run in runs if run[metric] is not None]
            summary[model][f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[model][f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    results["experiment_a_summary"] = summary
    results["interpretation"] = (
        "Complete-case results are sensitivity analyses only. The primary imputation-based study remains unchanged, "
        "and no approach was selected using VA performance."
    )
    return results


def _svg(path: Path, title: str, body: str, width: int = 900, height: int = 600) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="25" y="35" font-family="sans-serif" font-size="20">{title}</text>{body}</svg>',
        encoding="utf-8",
    )


def _line_plot(path: Path, title: str, curves: dict[str, tuple[np.ndarray, np.ndarray]], x_label: str, y_label: str) -> None:
    left, top, width, height = 80, 70, 760, 440
    body = [f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="black"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="black"/><text x="350" y="550" font-family="sans-serif">{x_label}</text><text x="15" y="300" transform="rotate(-90 15 300)" font-family="sans-serif">{y_label}</text>']
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")
    for index, (label, (xs, ys)) in enumerate(curves.items()):
        points = " ".join(f"{left + float(x)*width:.2f},{top+height-float(y)*height:.2f}" for x, y in zip(xs, ys))
        color = colors[index % len(colors)]
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="650" y="{85+index*20}" font-family="sans-serif" font-size="12" fill="{color}">{label}</text>')
    _svg(path, title, "".join(body))


def _make_plots(output: Path, data: pd.DataFrame, predictions: pd.DataFrame, reliability: dict[str, Any]) -> None:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for experiment, label in (("A_random_group_stratified", "A"), ("B_external_va", "B")):
        for model in MODEL_NAMES:
            frame = predictions[(predictions.experiment == experiment) & (predictions.model == model)]
            if len(frame) and frame.true_target.nunique() == 2:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(frame.true_target, frame.predicted_probability)
                curves[f"{label} {model}"] = (fpr, tpr)
    _line_plot(output / "reliability_roc_curves.svg", "Reliability investigation: ROC curves", curves, "False-positive rate", "True-positive rate")

    calibration_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for experiment, models in reliability["evaluations"].items():
        for model, report in models.items():
            points = report["calibration_curve"]
            calibration_curves[f"{experiment[:1]} {model}"] = (
                np.asarray([point["mean_predicted_probability"] for point in points]),
                np.asarray([point["observed_positive_rate"] for point in points]),
            )
    _line_plot(output / "calibration_curves.svg", "Reliability curves", calibration_curves, "Mean predicted probability", "Observed positive rate")

    missing = missingness_analysis(data)
    sources = list(missing["by_source"])
    body = []
    for feature_index, feature in enumerate(FEATURES):
        x = 80 + feature_index * 105
        body.append(f'<text x="{x}" y="540" transform="rotate(45 {x} 540)" font-family="sans-serif" font-size="11">{feature}</text>')
        for source_index, source in enumerate(sources):
            value = missing["by_source"][source]["feature_missing_percentages"][feature]
            height = float(value) * 4
            body.append(f'<rect x="{x+source_index*7}" y="{480-height}" width="6" height="{height}" fill="{("#1f77b4", "#d62728", "#2ca02c", "#9467bd")[source_index]}"/>')
    _svg(output / "missingness_differences.svg", "Missingness percentage by source", "".join(body))

    numeric = distribution_shift_analysis(data)["numeric"]
    body = []
    for index, feature in enumerate(NUMERIC_FEATURES):
        x = 80 + index * 180
        values = [numeric[feature][source]["mean"] for source in sources]
        maximum = max(value for value in values if value is not None) or 1
        body.append(f'<text x="{x}" y="540" font-family="sans-serif" font-size="11">{feature}</text>')
        for source_index, value in enumerate(values):
            height = float(value or 0) / maximum * 400
            body.append(f'<rect x="{x+source_index*22}" y="{480-height}" width="18" height="{height}" fill="{("#1f77b4", "#d62728", "#2ca02c", "#9467bd")[source_index]}"/>')
    _svg(output / "numeric_feature_shift.svg", "Numeric feature means by source", "".join(body))


def _report(output: Path, distribution: dict[str, Any], missing: dict[str, Any], source: dict[str, Any], reliability: dict[str, Any], complete_case: dict[str, Any]) -> None:
    va = reliability["evaluations"]["B_external_va"]
    lines = [
        "# Phase 2 diagnostic investigation: UCI Heart Disease",
        "",
        "## Summary of the first study",
        "",
        "The primary study found development-source mean ROC-AUC of 0.8646 for Logistic Regression and 0.8589 for Random Forest, versus 0.6668 and 0.6556 on the held-out Long Beach VA source. This phase investigates descriptive explanations without changing the original protocol or results.",
        "",
        "## Feature distribution shift",
        "",
        "The machine-readable distribution comparisons are in `distribution_shift.json`. Numeric summaries include means, standard deviations, medians, interquartile ranges, and standardized mean differences. Categorical summaries include proportions, absolute differences, and Jensen-Shannon divergence.",
        "",
        "The largest standardized differences and categorical divergences should be interpreted as source-associated distribution differences, not causal explanations. Source is not used as a model feature.",
        "",
        "## Missingness differences",
        "",
        "Missingness counts, percentages, complete-record percentages, and missingness patterns are in `missingness_analysis.json`. The analysis identifies whether VA missingness differs from pooled development missingness, but cannot establish that missingness caused performance deterioration.",
        "",
        "## Source-specific performance",
        "",
        "Source-specific Experiment A results use the existing repeated-split predictions. Because records appear in multiple repeated test splits, these descriptive rows are not independent observations. The small Switzerland source should not be treated as conclusive.",
        "",
        "## Reliability analysis",
        "",
        "Saved probability distributions, ROC/PR metrics, calibration bins, Brier scores, prevalence-only Brier references, and prevalence PR baselines are in `reliability_analysis.json`. VA prevalence is 74.5%, so a prevalence-only Brier reference is 0.1900 and the PR baseline is 0.7450. These baselines make the high VA PR-AUC values less informative than they appear without prevalence context.",
        "",
        "| Model | VA ROC-AUC | VA PR-AUC | VA Brier | VA prevalence-only Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    def display(value: Any) -> str:
        return "undefined" if value is None else f"{float(value):.4f}"

    distribution_table = [
        "",
        "| Numeric feature | VA versus pooled-development SMD |",
        "|---|---:|",
    ]
    for feature in NUMERIC_FEATURES:
        smd = distribution["comparisons"][feature]["va_vs_pooled_development_smd"]
        distribution_table.append(f"| {feature} | {display(smd)} |")
    distribution_table.extend([
        "",
        "| Categorical feature | VA versus pooled-development JS divergence |",
        "|---|---:|",
    ])
    for feature in CATEGORICAL_FEATURES:
        js_value = distribution["comparisons"][feature]["va_vs_pooled_development_js_divergence"]
        distribution_table.append(f"| {feature} | {display(js_value)} |")
    lines[lines.index("## Missingness differences"):lines.index("## Missingness differences")] = distribution_table

    missing_table = [
        "",
        "Largest absolute VA versus pooled-development missingness differences:",
        "",
        "| Feature | Difference in missingness percentage points |",
        "|---|---:|",
    ]
    for feature, difference in list(missing["largest_va_vs_development_differences"].items())[:3]:
        missing_table.append(f"| {feature} | {difference:.2f} |")
    lines[lines.index("## Source-specific performance"):lines.index("## Source-specific performance")] = missing_table

    source_table = [
        "",
        "| Source | Model | Unique records | Class counts | ROC-AUC | PR-AUC | Brier |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for source_name, models in source["experiment_a_existing_predictions"].items():
        for model, report in models.items():
            metrics = report["metrics"]
            source_table.append(
                f"| {source_name} | {model} | {report['unique_record_count']} | {metrics['class_counts']} | {display(metrics['roc_auc'])} | {display(metrics['pr_auc'])} | {display(metrics['brier_score'])} |"
            )
    lines[lines.index("## Reliability analysis"):lines.index("## Reliability analysis")] = source_table

    complete_table = [
        "",
        "| Model | Complete-case VA records | Class counts | ROC-AUC | PR-AUC | Balanced accuracy | Brier |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for model, report in complete_case["experiment_b"].items():
        metrics = report["metrics"]
        complete_table.append(
            f"| {model} | {report['external_record_count']} | {metrics['class_counts']} | {display(metrics['roc_auc'])} | {display(metrics['pr_auc'])} | {display(metrics['balanced_accuracy'])} | {display(metrics['brier_score'])} |"
        )
    for model in MODEL_NAMES:
        report = va[model]
        lines.append(f"| {model} | {report['metrics']['roc_auc']:.4f} | {report['metrics']['pr_auc']:.4f} | {report['metrics']['brier_score']:.4f} | {report['baselines']['prevalence_only_brier']:.4f} |")
    lines.extend([
        "",
        "## Complete-case sensitivity analysis",
        "",
        f"The complete-case sensitivity analysis retained {complete_case['analysis_population']['complete_case_rows']} of {complete_case['analysis_population']['all_rows']} records and excluded {complete_case['analysis_population']['excluded_for_sensitivity_only']} records only for this sensitivity analysis. The primary imputation-based study was not changed, and VA was not used to select a preferred approach.",
        "",
        "## What the evidence supports",
        "",
        "The results support that the held-out VA source differs in feature distributions and missingness patterns from development sources, and that external discrimination and threshold performance are lower than the random development-source benchmark. Reliability and prevalence-aware baselines must be considered when interpreting VA PR-AUC and Brier scores.",
        "",
        "## What remains uncertain",
        "",
        "Source, population, measurement, missingness, and label processes are confounded. These diagnostics cannot identify which factor caused deterioration. The small VA sample and small Switzerland source also produce unstable subgroup estimates.",
        "",
        "## Limitations",
        "",
        "The dataset is historical, the target is observed disease status rather than prospective risk, and duplicate records reduce effective sample size. Complete-case results change the analysis population and are sensitivity evidence only. No clinical validity claim is made.",
        "",
        "## Follow-up hypotheses",
        "",
        "1. Source-level feature and missingness shifts jointly explain a material portion of the external reliability deterioration.",
        "2. Calibration degradation under source shift is more pronounced than ranking degradation for at least one classical model.",
        "3. Source-aware, leakage-safe missingness handling may improve cross-source reliability, but this requires a separately pre-registered experiment and must not use VA for selection.",
    ])
    marker_index = lines.index("## What the evidence supports")
    lines[marker_index:marker_index] = complete_table
    (output / "research_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostics(output_dir: Path | None = None) -> dict[str, Any]:
    """Run Phase 2 diagnostics without changing primary study artifacts."""

    output = output_dir or DIAGNOSTICS_RESULTS
    output.mkdir(parents=True, exist_ok=True)
    data = load_uci_heart_sources()
    predictions = pd.read_csv(PRIMARY_RESULTS / "predictions.csv")
    distribution = distribution_shift_analysis(data)
    missingness = missingness_analysis(data)
    source = source_performance(predictions)
    reliability = reliability_analysis(predictions)
    metadata = json.loads((PRIMARY_RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    complete_case = complete_case_sensitivity(data, metadata["seeds"])
    _write_json(output / "distribution_shift.json", distribution)
    _write_json(output / "missingness_analysis.json", missingness)
    _write_json(output / "source_performance.json", source)
    _write_json(output / "reliability_analysis.json", reliability)
    _write_json(output / "complete_case_sensitivity.json", complete_case)
    _make_plots(output, data, predictions, reliability)
    _report(output, distribution, missingness, source, reliability, complete_case)
    return {
        "distribution_shift": distribution,
        "missingness_analysis": missingness,
        "source_performance": source,
        "reliability_analysis": reliability,
        "complete_case_sensitivity": complete_case,
    }


if __name__ == "__main__":
    run_diagnostics()
