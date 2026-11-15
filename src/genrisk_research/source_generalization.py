"""Source-held-out generalisation experiment for the UCI Heart Disease data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score

from .uci_heart_disease import (
    DEFAULT_SEEDS,
    DEVELOPMENT_SOURCES,
    FEATURES,
    MODEL_NAMES,
    TARGET,
    build_model_pipeline,
    classification_metrics,
    development_data,
    fit_model,
    load_uci_heart_sources,
    repeated_group_stratified_splits,
)

SOURCE_HOLDOUT_FOLDS: dict[str, tuple[tuple[str, ...], str]] = {
    "holdout_cleveland": (("Hungary", "Switzerland"), "Cleveland"),
    "holdout_hungary": (("Cleveland", "Switzerland"), "Hungary"),
    "holdout_switzerland": (("Cleveland", "Hungary"), "Switzerland"),
}
OUTPUT_SUBDIRECTORY = "source_generalization"


def source_holdout_partitions(data: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return the three pre-specified source-held-out train/test partitions."""

    development = development_data(data)
    partitions: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold, (training_sources, held_out_source) in SOURCE_HOLDOUT_FOLDS.items():
        train = development[development["source"].isin(training_sources)].reset_index(drop=True)
        test = development[development["source"].eq(held_out_source)].reset_index(drop=True)
        if set(train["source"]) != set(training_sources):
            raise ValueError(f"{fold} is missing a declared training source")
        if set(test["source"]) != {held_out_source}:
            raise ValueError(f"{fold} has an invalid held-out source")
        if set(train["record_id"]) & set(test["record_id"]):
            raise AssertionError(f"{fold} has overlapping training and held-out records")
        if set(train[TARGET]) != {0, 1} or set(test[TARGET]) != {0, 1}:
            raise ValueError(f"{fold} does not contain both target classes")
        partitions[fold] = (train, test)
    return partitions


def _extended_metrics(y_true: Iterable[int], probabilities: Iterable[float]) -> dict[str, Any]:
    """Calculate requested threshold and probability metrics without false certainty."""

    y = np.asarray(list(y_true), dtype=int)
    probabilities_array = np.asarray(list(probabilities), dtype=float)
    predictions = (probabilities_array >= 0.5).astype(int)
    metrics = classification_metrics(y, probabilities_array)
    predicted_positive = int(predictions.sum())
    actual_positive = int(y.sum())
    metrics.update({
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)) if predicted_positive else None,
        "f1_score": float(f1_score(y, predictions, zero_division=0)) if actual_positive and predicted_positive else None,
    })
    return metrics


def evaluate_model(model_name: str, train: pd.DataFrame, test: pd.DataFrame, random_state: int) -> tuple[dict[str, Any], np.ndarray]:
    """Fit one model on train only and evaluate it on test."""

    model = fit_model(model_name, train, random_state=random_state)
    probabilities = model.predict_proba(test[list(FEATURES)])[:, 1]
    return _extended_metrics(test[TARGET], probabilities), probabilities


def _record_result(
    *, fold: str, model: str, train: pd.DataFrame, test: pd.DataFrame,
    metrics: dict[str, Any], random_seed: int | None,
) -> dict[str, Any]:
    return {
        "evaluation": fold,
        "model": model,
        "training_sources": "+".join(sorted(train["source"].unique())),
        "held_out_source": str(test["source"].iloc[0]),
        "random_seed": random_seed,
        "training_record_count": int(len(train)),
        "held_out_record_count": int(len(test)),
        "training_class_counts": {str(label): int((train[TARGET] == label).sum()) for label in (0, 1)},
        "held_out_class_counts": {str(label): int((test[TARGET] == label).sum()) for label in (0, 1)},
        **metrics,
    }


def run_source_generalization(
    output_dir: Path | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Run source-held-out folds and the pooled random-split benchmark."""

    from .uci_heart_disease import repository_root

    output = output_dir or repository_root() / "results" / "uci_heart_disease" / OUTPUT_SUBDIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in seeds)
    data = load_uci_heart_sources()
    partitions = source_holdout_partitions(data)
    source_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for fold, (train, test) in partitions.items():
        for model_name in MODEL_NAMES:
            metrics, probabilities = evaluate_model(model_name, train, test, random_state=42)
            source_rows.append(_record_result(fold=fold, model=model_name, train=train, test=test, metrics=metrics, random_seed=42))
            for record_id, source, target, probability in zip(test["record_id"], test["source"], test[TARGET], probabilities):
                predictions.append({"evaluation": fold, "model": model_name, "record_id": record_id, "source": source, "true_target": int(target), "predicted_probability": float(probability), "predicted_class": int(probability >= 0.5)})

    development = development_data(data)
    random_rows: list[dict[str, Any]] = []
    for seed, train_index, test_index in repeated_group_stratified_splits(development, seeds=seeds):
        train = development.iloc[train_index]
        test = development.iloc[test_index]
        for model_name in MODEL_NAMES:
            metrics, probabilities = evaluate_model(model_name, train, test, random_state=seed)
            random_rows.append(_record_result(fold="random_development_split", model=model_name, train=train, test=test, metrics=metrics, random_seed=seed))
            for record_id, source, target, probability in zip(test["record_id"], test["source"], test[TARGET], probabilities):
                predictions.append({"evaluation": "random_development_split", "model": model_name, "seed": seed, "record_id": record_id, "source": source, "true_target": int(target), "predicted_probability": float(probability), "predicted_class": int(probability >= 0.5)})

    metric_names = ("roc_auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1_score", "brier_score")
    random_summary: dict[str, dict[str, float | None]] = {}
    for model_name in MODEL_NAMES:
        model_rows = [row for row in random_rows if row["model"] == model_name]
        random_summary[model_name] = {}
        for metric in metric_names:
            values = [row[metric] for row in model_rows if row[metric] is not None]
            random_summary[model_name][f"{metric}_mean"] = float(np.mean(values)) if values else None
            random_summary[model_name][f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None

    source_summary = {model: {
        metric: float(np.mean([row[metric] for row in source_rows if row["model"] == model and row[metric] is not None]))
        if any(row["model"] == model and row[metric] is not None for row in source_rows) else None
        for metric in metric_names
    } for model in MODEL_NAMES}
    degradation = {model: {
        metric: (random_summary[model][f"{metric}_mean"] - source_summary[model][metric])
        if random_summary[model][f"{metric}_mean"] is not None and source_summary[model][metric] is not None else None
        for metric in metric_names
    } for model in MODEL_NAMES}

    fold_frame = pd.DataFrame(source_rows)
    random_frame = pd.DataFrame(random_rows)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    random_frame.to_csv(output / "random_split_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    payload = {
        "source_holdout": {"folds": source_rows, "mean_by_model": source_summary},
        "random_split_benchmark": {"seeds": list(seeds), "runs": random_rows, "summary_by_model": random_summary},
        "degradation_random_mean_minus_source_holdout_mean": degradation,
        "features": list(FEATURES),
        "target_mapping": {"0": 0, "1-4": 1},
        "threshold": 0.5,
        "models": list(MODEL_NAMES),
        "preprocessing": "Reused training-only imputation, encoding, and model-specific scaling from the primary study.",
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_comparison_svg(output / "comparison.svg", source_rows, random_summary)
    _write_report(output, payload)
    return payload


def _write_comparison_svg(path: Path, source_rows: list[dict[str, Any]], random_summary: dict[str, dict[str, float | None]]) -> None:
    """Write a dependency-free ROC-AUC comparison plot."""

    width, height = 900, 520
    body = [f'<rect width="{width}" height="{height}" fill="white"/>', '<text x="30" y="30" font-family="sans-serif" font-size="18">ROC-AUC: random benchmark versus source-held-out folds</text>']
    labels = ["random"] + [fold for fold in SOURCE_HOLDOUT_FOLDS]
    colors = {"logistic_regression": "#1f77b4", "random_forest": "#d62728"}
    for index, label in enumerate(labels):
        x = 100 + index * 190
        body.append(f'<text x="{x}" y="470" font-family="sans-serif" font-size="11">{label.replace("holdout_", "")}</text>')
        for model_index, model in enumerate(MODEL_NAMES):
            if label == "random":
                value = random_summary[model]["roc_auc_mean"]
            else:
                values = [row["roc_auc"] for row in source_rows if row["evaluation"] == label and row["model"] == model]
                value = values[0] if values else None
            if value is None:
                continue
            bar_height = float(value) * 360
            body.append(f'<rect x="{x + model_index * 42}" y="{430 - bar_height}" width="30" height="{bar_height}" fill="{colors[model]}"/>')
            body.append(f'<text x="{x + model_index * 42}" y="{420 - bar_height}" font-family="sans-serif" font-size="10">{float(value):.3f}</text>')
    body.extend(['<text x="700" y="70" font-family="sans-serif" font-size="11" fill="#1f77b4">Logistic Regression</text>', '<text x="700" y="90" font-family="sans-serif" font-size="11" fill="#d62728">Random Forest</text>'])
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(body) + "</svg>\n", encoding="utf-8")


def _write_report(output: Path, payload: dict[str, Any]) -> None:
    def display(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.4f}"

    lines = [
        "# Source-held-out generalisation experiment",
        "",
        "This experiment compares pooled development-source random evaluation with leave-one-source-out evaluation. It does not alter the primary study or make causal or clinical claims.",
        "",
        "## Protocol",
        "",
        "Cleveland, Hungary, and Switzerland are used as development sources. Each source is held out once while the other two are used for training. The random benchmark uses the existing ten-seed grouped stratified split procedure. All preprocessing is fit inside a fresh model pipeline using training records only.",
        "",
        "## Results",
        "",
        "| Model | Random ROC-AUC mean | Random ROC-AUC SD | Source-held-out ROC-AUC mean | Difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        summary = payload["random_split_benchmark"]["summary_by_model"][model]
        source = payload["source_holdout"]["mean_by_model"][model]["roc_auc"]
        difference = payload["degradation_random_mean_minus_source_holdout_mean"][model]["roc_auc"]
        lines.append(f"| {model} | {display(summary['roc_auc_mean'])} | {display(summary['roc_auc_std'])} | {display(source)} | {display(difference)} |")
    lines.extend(["", "### Held-out folds", "", "| Fold | Held-out source | Model | Records | ROC-AUC | Balanced accuracy | Brier score |", "|---|---|---|---:|---:|---:|---:|"])
    for row in payload["source_holdout"]["folds"]:
        lines.append(f"| {row['evaluation']} | {row['held_out_source']} | {row['model']} | {row['held_out_record_count']} | {display(row['roc_auc'])} | {display(row['balanced_accuracy'])} | {display(row['brier_score'])} |")
    lines.extend([
        "", "## Interpretation", "",
        "Lower source-held-out performance than the random benchmark is consistent with source-associated generalisation degradation. The source folds differ in population, measurement, missingness, and label processes, so these results do not identify a causal source effect or establish universal model robustness.",
        "", "## Limitations", "",
        "The three held-out sources are small and heterogeneous, and the experiment uses the same historical binary disease-status target and seven-feature representation as the primary study. Random-split and source-held-out results answer different questions; their difference is a descriptive comparison rather than an unbiased causal effect.",
        "", "## Reproducibility", "",
        "Metrics, predictions, seeds, source assignments, and preprocessing conventions are recorded in `metrics.json`, `fold_metrics.csv`, `random_split_metrics.csv`, and `predictions.csv`. No held-out source was used for preprocessing, feature selection, threshold selection, or model selection.",
    ])
    (output / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_source_generalization()
