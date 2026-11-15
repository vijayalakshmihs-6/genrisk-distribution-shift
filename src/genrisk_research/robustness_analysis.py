"""Uncertainty and robustness analysis for source-held-out predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .source_generalization import SOURCE_HOLDOUT_FOLDS, _extended_metrics
from .uci_heart_disease import FEATURES, MODEL_NAMES, TARGET, load_uci_heart_sources, repository_root

METRICS = ("roc_auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1_score", "brier_score")
DEFAULT_RESAMPLES = 2000
CONFIDENCE_LEVEL = 0.95
OUTPUT_SUBDIRECTORY = "robustness_analysis"


def _metric_values(y_true: Iterable[int], probabilities: Iterable[float]) -> dict[str, float | None]:
    metrics = _extended_metrics(y_true, probabilities)
    return {metric: metrics.get(metric) for metric in METRICS}


def bootstrap_metric_intervals(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    groups: Iterable[str] | None = None,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    """Return percentile intervals from class-stratified, group-cluster bootstrap.

    Duplicate groups are sampled as clusters. Groups are sampled separately
    within each class so every valid replicate retains both classes. Metrics
    that remain undefined are counted as invalid rather than imputed.
    """

    y = np.asarray(list(y_true), dtype=int)
    probabilities_array = np.asarray(list(probabilities), dtype=float)
    if len(y) != len(probabilities_array) or len(y) == 0:
        raise ValueError("y_true and probabilities must be non-empty and equally sized")
    if groups is None:
        group_array = np.asarray([f"row:{index}" for index in range(len(y))], dtype=str)
    else:
        group_array = np.asarray(list(groups), dtype=str)
        if len(group_array) != len(y):
            raise ValueError("groups must have the same length as y_true")
    if not set(y).issubset({0, 1}):
        raise ValueError("binary targets must be coded as 0 and 1")

    point = _metric_values(y, probabilities_array)
    class_group_indices: dict[int, dict[str, np.ndarray]] = {}
    for label in (0, 1):
        class_group_indices[label] = {}
        for group in np.unique(group_array[y == label]):
            class_group_indices[label][str(group)] = np.flatnonzero((y == label) & (group_array == group))
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}
    invalid: dict[str, int] = {metric: 0 for metric in METRICS}
    for _ in range(int(n_resamples)):
        sampled_indices: list[int] = []
        for label in (0, 1):
            available = list(class_group_indices[label])
            if not available:
                sampled_indices = []
                break
            sampled_groups = rng.choice(available, size=len(available), replace=True)
            for group in sampled_groups:
                sampled_indices.extend(class_group_indices[label][str(group)].tolist())
        if not sampled_indices:
            for metric in METRICS:
                invalid[metric] += 1
            continue
        replicate = _metric_values(y[sampled_indices], probabilities_array[sampled_indices])
        for metric in METRICS:
            value = replicate[metric]
            if value is None or not np.isfinite(value):
                invalid[metric] += 1
            else:
                samples[metric].append(float(value))

    result: dict[str, Any] = {
        "method": "class-stratified percentile bootstrap with duplicate groups sampled as clusters",
        "confidence_level": CONFIDENCE_LEVEL,
        "alpha": 1.0 - CONFIDENCE_LEVEL,
        "seed": int(seed),
        "n_resamples": int(n_resamples),
        "class_counts": {str(label): int((y == label).sum()) for label in (0, 1)},
        "duplicate_group_count": int(np.unique(group_array).size),
        "metrics": {},
    }
    for metric in METRICS:
        values = np.asarray(samples[metric], dtype=float)
        result["metrics"][metric] = {
            "point_estimate": point[metric],
            "lower": float(np.quantile(values, 0.025)) if len(values) else None,
            "upper": float(np.quantile(values, 0.975)) if len(values) else None,
            "valid_replicates": int(len(values)),
            "invalid_replicates": int(invalid[metric]),
        }
    return result


def random_split_uncertainty(random_metrics: pd.DataFrame) -> dict[str, Any]:
    """Summarise variability across fixed random seeds, not independent datasets."""

    output: dict[str, Any] = {
        "interpretation": "Empirical variability across fixed random seeds; not independent-dataset uncertainty.",
        "models": {},
    }
    for model in MODEL_NAMES:
        model_frame = random_metrics[random_metrics["model"].eq(model)]
        output["models"][model] = {"seed_count": int(model_frame["random_seed"].nunique()), "metrics": {}}
        for metric in METRICS:
            values = model_frame[metric].dropna().astype(float).to_numpy()
            output["models"][model]["metrics"][metric] = {
                "values_by_seed": [float(value) for value in values],
                "mean": float(np.mean(values)) if len(values) else None,
                "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "median": float(np.median(values)) if len(values) else None,
                "interval_2_5": float(np.quantile(values, 0.025)) if len(values) else None,
                "interval_97_5": float(np.quantile(values, 0.975)) if len(values) else None,
                "valid_seed_count": int(len(values)),
            }
    return output


def source_robustness(source_metrics: pd.DataFrame, random_uncertainty: dict[str, Any]) -> dict[str, Any]:
    """Compare held-out source values with the pooled random benchmark."""

    output: dict[str, Any] = {
        "interpretation": "Source-level descriptive variation; no significance claims are made.",
        "models": {},
    }
    for model in MODEL_NAMES:
        model_frame = source_metrics[source_metrics["model"].eq(model)]
        output["models"][model] = {"metrics": {}}
        for metric in METRICS:
            values = model_frame[["held_out_source", metric]].dropna()
            random_mean = random_uncertainty["models"][model]["metrics"][metric]["mean"]
            source_values = {str(row.held_out_source): float(getattr(row, metric)) for row in values.itertuples()}
            higher_is_better = metric != "brier_score"
            worse_flags = [value < random_mean if higher_is_better else value > random_mean for value in source_values.values()] if random_mean is not None else []
            output["models"][model]["metrics"][metric] = {
                "source_values": source_values,
                "mean": float(values[metric].mean()) if len(values) else None,
                "standard_deviation": float(values[metric].std(ddof=1)) if len(values) > 1 else None,
                "median": float(values[metric].median()) if len(values) else None,
                "range": float(values[metric].max() - values[metric].min()) if len(values) else None,
                "random_split_mean": random_mean,
                "random_minus_source_mean": (random_mean - float(values[metric].mean())) if random_mean is not None and len(values) and higher_is_better else ((float(values[metric].mean()) - random_mean) if random_mean is not None and len(values) else None),
                "direction": "higher_is_better" if higher_is_better else "lower_is_better",
                "all_sources_worse_than_random_mean": bool(worse_flags) and all(worse_flags),
            }
    return output


def _write_svg(path: Path, robustness: dict[str, Any]) -> None:
    width, height = 980, 600
    metrics = ("roc_auc", "balanced_accuracy", "brier_score")
    body = [f'<rect width="{width}" height="{height}" fill="white"/>', '<text x="25" y="28" font-family="sans-serif" font-size="18">Source robustness by metric</text>']
    colors = {"logistic_regression": "#1f77b4", "random_forest": "#d62728"}
    for panel, metric in enumerate(metrics):
        origin_x = 45 + panel * 310
        body.append(f'<text x="{origin_x}" y="70" font-family="sans-serif" font-size="14">{metric}</text>')
        labels = ["random", "Cleveland", "Hungary", "Switzerland"]
        for index, label in enumerate(labels):
            x = origin_x + index * 62
            body.append(f'<text x="{x}" y="545" transform="rotate(-35 {x} 545)" font-family="sans-serif" font-size="10">{label}</text>')
            for model_index, model in enumerate(MODEL_NAMES):
                if label == "random":
                    value = robustness["models"][model]["metrics"][metric]["random_split_mean"]
                else:
                    value = robustness["models"][model]["metrics"][metric]["source_values"].get(label)
                if value is None:
                    continue
                scale = float(value) if metric != "brier_score" else float(value) * 2.0
                bar_height = max(0.0, min(scale, 1.0)) * 400
                body.append(f'<rect x="{x + model_index * 20}" y="{500 - bar_height}" width="15" height="{bar_height}" fill="{colors[model]}"/>')
        body.append(f'<line x1="{origin_x}" y1="500" x2="{origin_x + 250}" y2="500" stroke="#333"/>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(body) + "</svg>\n", encoding="utf-8")


def _write_report(output: Path, bootstrap: list[dict[str, Any]], random: dict[str, Any], robustness: dict[str, Any]) -> None:
    lines = [
        "# Statistical uncertainty and robustness analysis",
        "",
        "This analysis quantifies conditional test-sample uncertainty for the source-held-out predictions and describes variability across the fixed random-split seeds. It does not replace external validation or quantify training/model-selection uncertainty.",
        "",
        "## Bootstrap method",
        "",
        "A 2,000-replicate 95% percentile bootstrap was applied separately within each held-out source and model. Duplicate groups were sampled as clusters, and clusters were sampled separately within each class. Undefined metric replicates were counted as invalid rather than imputed. This preserves class representation but conditions the interval on the observed class composition and does not capture training or source-selection uncertainty.",
        "",
        "## Random-split uncertainty",
        "",
        "The random-split intervals summarize the ten fixed seeds. They describe split-to-split variability, not ten independent datasets and not an independent external-validation interval.",
        "",
        "## Source robustness",
        "",
        "| Model | Metric | Random mean | Source-held-out mean | Source range | All sources worse than random mean |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model in MODEL_NAMES:
        for metric in ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "brier_score"):
            item = robustness["models"][model]["metrics"][metric]
            display = lambda value: "undefined" if value is None else f"{value:.4f}"
            lines.append(f"| {model} | {metric} | {display(item['random_split_mean'])} | {display(item['mean'])} | {display(item['range'])} | {item['all_sources_worse_than_random_mean']} |")
    lines.extend([
        "", "## Interpretation", "",
        "Logistic Regression has a smaller source-held-out ROC-AUC range than Random Forest in this analysis. That is a descriptive consistency result, not evidence that Logistic Regression is universally more robust. Metric-specific patterns differ: discrimination, threshold classification, and calibration should not be conflated.",
        "", "## Limitations", "",
        "The intervals are conditional on fixed fitted-model predictions and the observed held-out samples. They do not include uncertainty from model training, feature selection, hyperparameter selection, source selection, or dataset shift itself. Stratified cluster resampling preserves class representation and duplicate records, but it conditions on the observed prevalence and may not represent future-source prevalence uncertainty. No causal or clinical claim is made.",
    ])
    (output / "robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_robustness_analysis(output_dir: Path | None = None, *, n_resamples: int = DEFAULT_RESAMPLES, seed: int = 2026) -> dict[str, Any]:
    """Run Phase 3 without writing to the primary or source-generalisation outputs."""

    source_dir = repository_root() / "results" / "uci_heart_disease" / "source_generalization"
    output = output_dir or repository_root() / "results" / "uci_heart_disease" / OUTPUT_SUBDIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    source_metrics = pd.read_csv(source_dir / "fold_metrics.csv")
    random_metrics = pd.read_csv(source_dir / "random_split_metrics.csv")
    predictions = pd.read_csv(source_dir / "predictions.csv")
    data = load_uci_heart_sources()[["record_id", "duplicate_group", TARGET]]
    predictions = predictions.merge(
        data,
        left_on=["record_id", "true_target"],
        right_on=["record_id", TARGET],
        how="left",
        validate="many_to_one",
    )
    if predictions["duplicate_group"].isna().any():
        raise ValueError("source-held-out predictions contain unknown record IDs")

    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_json: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(SOURCE_HOLDOUT_FOLDS):
        for model_index, model in enumerate(MODEL_NAMES):
            subset = predictions[predictions["evaluation"].eq(fold) & predictions["model"].eq(model)]
            result = bootstrap_metric_intervals(subset[TARGET], subset["predicted_probability"], subset["duplicate_group"], n_resamples=n_resamples, seed=seed + fold_index * 100 + model_index)
            for metric, values in result["metrics"].items():
                bootstrap_rows.append({"fold": fold, "held_out_source": str(subset["source"].iloc[0]), "model": model, "metric": metric, **values, "n_resamples": result["n_resamples"], "seed": result["seed"], "method": result["method"]})
            bootstrap_json.append({"fold": fold, "held_out_source": str(subset["source"].iloc[0]), "model": model, **result})

    random_uncertainty = random_split_uncertainty(random_metrics)
    robustness = source_robustness(source_metrics, random_uncertainty)
    pd.DataFrame(bootstrap_rows).to_csv(output / "bootstrap_confidence_intervals.csv", index=False)
    (output / "random_split_uncertainty.json").write_text(json.dumps(random_uncertainty, indent=2), encoding="utf-8")
    (output / "source_robustness.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")
    (output / "bootstrap_details.json").write_text(json.dumps(bootstrap_json, indent=2), encoding="utf-8")
    _write_svg(output / "robustness_comparison.svg", robustness)
    _write_report(output, bootstrap_json, random_uncertainty, robustness)
    return {"bootstrap": bootstrap_json, "random_split": random_uncertainty, "source_robustness": robustness}


if __name__ == "__main__":
    run_robustness_analysis()
