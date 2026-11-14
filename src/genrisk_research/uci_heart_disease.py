"""The first GenRisk Research empirical study.

This module defines the approved UCI Heart Disease source-shift protocol but
does not execute it on import. Raw files are read only; derived results are
written only when :func:`run_study` is called explicitly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from importlib.metadata import version as package_version

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ("age", "sex", "cp", "trestbps", "chol", "restecg", "thalach")
TARGET = "target"
CATEGORICAL_FEATURES = ("sex", "cp", "restecg")
NUMERIC_FEATURES = ("age", "trestbps", "chol", "thalach")
PROCESSED_COLUMNS = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
)
DEVELOPMENT_SOURCES = ("Cleveland", "Hungary", "Switzerland")
EXTERNAL_SOURCE = "Long Beach VA"
SOURCE_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "Long Beach VA": "processed.va.data",
}
DEFAULT_SEEDS = tuple(range(10))
MODEL_NAMES = ("logistic_regression", "random_forest")
METRIC_NAMES = (
    "roc_auc", "pr_auc", "balanced_accuracy", "sensitivity",
    "specificity", "brier_score", "log_loss",
)
PREPROCESSING_CONFIGURATION = {
    "missing_numeric": "median_fit_on_training_only",
    "missing_categorical": "most_frequent_fit_on_training_only",
    "categorical_encoding": "one_hot_fit_on_training_categories_only",
    "unseen_categories": "ignore",
    "logistic_numeric_transform": "standardize_fit_on_training_only",
    "random_forest_numeric_transform": "none",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_data_root() -> Path:
    return repository_root() / "data" / "raw" / "uci_heart_disease" / "extracted"


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<MISSING>"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), ".15g")
    return str(value).strip()


def _record_key(row: pd.Series) -> str:
    values = [_canonical_value(row[column]) for column in (*FEATURES, TARGET)]
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def load_uci_heart_sources(data_root: Path | None = None) -> pd.DataFrame:
    """Load the four processed UCI source files with explicit source labels."""

    root = data_root or raw_data_root()
    frames: list[pd.DataFrame] = []
    for source, filename in SOURCE_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"processed UCI source file not found: {path}")
        frame = pd.read_csv(
            path,
            header=None,
            names=PROCESSED_COLUMNS,
            na_values=["?"],
            keep_default_na=True,
        )
        frame["source"] = source
        frame["record_id"] = [f"{source}:{index}" for index in frame.index]
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    for column in (*FEATURES, TARGET):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[list(FEATURES) + [TARGET]].isna().all(axis=0).any():
        raise ValueError("a required feature or target column is entirely missing")
    if not data[TARGET].dropna().isin([0, 1, 2, 3, 4]).all():
        raise ValueError("target contains values outside the documented 0-4 coding")
    data[TARGET] = (data[TARGET] > 0).astype(int)
    data["duplicate_group"] = data.apply(_record_key, axis=1)
    return data


def duplicate_audit(data: pd.DataFrame) -> dict[str, Any]:
    """Return duplicate-group facts without removing any records."""

    grouped = data.groupby("duplicate_group", sort=False)
    duplicate_groups = grouped.size().loc[lambda values: values > 1]
    cross_source = 0
    within_source = 0
    for _, group in grouped:
        if len(group) > 1:
            if group["source"].nunique() > 1:
                cross_source += 1
            else:
                within_source += 1
    return {
        "record_count": int(len(data)),
        "unique_record_groups": int(grouped.ngroups),
        "duplicate_group_count": int(len(duplicate_groups)),
        "duplicate_record_count": int(duplicate_groups.sum()),
        "within_source_duplicate_groups": within_source,
        "cross_source_duplicate_groups": cross_source,
        "duplicate_groups": {
            str(key): int(size) for key, size in duplicate_groups.items()
        },
    }


def development_data(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["source"].isin(DEVELOPMENT_SOURCES)].reset_index(drop=True)


def external_data(data: pd.DataFrame) -> pd.DataFrame:
    return data[data["source"].eq(EXTERNAL_SOURCE)].reset_index(drop=True)


def repeated_group_stratified_splits(
    data: pd.DataFrame,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Create ten deterministic, approximately 80/20 grouped stratified splits."""

    y = data[TARGET].to_numpy()
    groups = data["duplicate_group"].to_numpy()
    X = np.zeros((len(data), 1), dtype=np.uint8)
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=int(seed),
        )
        train_index, test_index = next(splitter.split(X, y, groups))
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups & test_groups:
            raise AssertionError("duplicate group crosses a train/test boundary")
        if set(y[train_index]) != {0, 1} or set(y[test_index]) != {0, 1}:
            raise ValueError(
                f"group-aware stratification produced a single-class partition for seed {seed}"
            )
        splits.append((int(seed), train_index, test_index))
    return splits


def build_model_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    """Build an unweighted model with preprocessing inside the fitted pipeline."""

    scale_numeric = model_name == "logistic_regression"
    if model_name == "logistic_regression":
        model: Any = LogisticRegression(max_iter=2000, solver="lbfgs")
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"unsupported model: {model_name}")

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
        ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def fit_model(model_name: str, train: pd.DataFrame, random_state: int = 42) -> Pipeline:
    pipeline = build_model_pipeline(model_name, random_state=random_state)
    pipeline.fit(train[list(FEATURES)], train[TARGET])
    return pipeline


def classification_metrics(
    y_true: Iterable[int], probabilities: Iterable[float]
) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    probabilities_array = np.asarray(list(probabilities), dtype=float)
    predictions = (probabilities_array >= 0.5).astype(int)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    both_classes = set(y) == {0, 1}
    metrics: dict[str, Any] = {
        "record_count": int(len(y)),
        "class_counts": {str(label): int((y == label).sum()) for label in (0, 1)},
        "roc_auc": float(roc_auc_score(y, probabilities_array)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, probabilities_array)) if both_classes else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)) if both_classes else None,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "brier_score": float(brier_score_loss(y, probabilities_array)),
        "log_loss": float(log_loss(y, probabilities_array, labels=[0, 1])),
    }
    return metrics


def stratified_group_bootstrap_ci(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    groups: Iterable[str],
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap duplicate groups, preserving the number of groups per class."""

    y = np.asarray(list(y_true), dtype=int)
    probabilities_array = np.asarray(list(probabilities), dtype=float)
    group_array = np.asarray(list(groups), dtype=str)
    group_indices: dict[str, np.ndarray] = {}
    group_labels: dict[str, int] = {}
    for group in np.unique(group_array):
        indices = np.flatnonzero(group_array == group)
        labels = np.unique(y[indices])
        if len(labels) != 1:
            raise ValueError("a duplicate group contains multiple target classes")
        group_indices[group] = indices
        group_labels[group] = int(labels[0])

    class_groups = {
        label: np.asarray([group for group, value in group_labels.items() if value == label])
        for label in (0, 1)
    }
    if any(len(groups_for_class) == 0 for groups_for_class in class_groups.values()):
        raise ValueError("stratified bootstrap requires both target classes")

    rng = np.random.default_rng(seed)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    valid = 0
    invalid = 0
    for _ in range(n_resamples):
        selected_groups: list[str] = []
        for groups_for_class in class_groups.values():
            selected_groups.extend(
                rng.choice(groups_for_class, size=len(groups_for_class), replace=True).tolist()
            )
        indices = np.concatenate([group_indices[group] for group in selected_groups])
        try:
            metrics = classification_metrics(y[indices], probabilities_array[indices])
        except ValueError:
            invalid += 1
            continue
        if any(metrics[name] is None for name in METRIC_NAMES):
            invalid += 1
            continue
        valid += 1
        for name in METRIC_NAMES:
            value = metrics[name]
            assert value is not None
            samples[name].append(value)

    if valid == 0:
        raise ValueError("no valid bootstrap resamples were produced")
    return {
        "resamples_requested": int(n_resamples),
        "resamples_valid": valid,
        "invalid_resamples": invalid,
        "confidence_level": 0.95,
        "intervals": {
            name: {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }
            for name, values in samples.items()
        },
    }


def _prediction_frame(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    experiment: str,
    model_name: str,
    seed: int | None,
) -> pd.DataFrame:
    return pd.DataFrame({
        "experiment": experiment,
        "model": model_name,
        "seed": seed,
        "record_id": rows["record_id"].to_numpy(),
        "source": rows["source"].to_numpy(),
        "duplicate_group": rows["duplicate_group"].to_numpy(),
        "true_target": rows[TARGET].to_numpy(),
        "predicted_probability": probabilities,
        "predicted_class": (probabilities >= 0.5).astype(int),
    })


def dataset_quality_report(data: pd.DataFrame) -> dict[str, Any]:
    """Build the non-destructive source quality report used by the study."""

    source_reports: dict[str, Any] = {}
    for source, frame in data.groupby("source", sort=False):
        source_reports[source] = {
            "row_count": int(len(frame)),
            "missing_counts": {
                feature: int(frame[feature].isna().sum()) for feature in FEATURES
            },
            "missing_percentages": {
                feature: float(frame[feature].isna().mean() * 100) for feature in FEATURES
            },
            "target_distribution": {
                str(int(value)): int(count)
                for value, count in frame[TARGET].value_counts().sort_index().items()
            },
            "duplicate_groups": int(frame["duplicate_group"].nunique()),
        }
    return {
        "features": list(FEATURES),
        "target": TARGET,
        "source_reports": source_reports,
        "duplicate_audit": duplicate_audit(data),
        "excluded_records": 0,
        "exclusion_reason": "No records excluded; missing values are imputed in fitted pipelines.",
    }


def _checksum_manifest() -> dict[str, str]:
    manifest = raw_data_root().parent / "checksums.sha256"
    values: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative_path = line.split(maxsplit=1)
            values[relative_path] = digest
    return values


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scikit-learn")
    return {package: package_version(package) for package in packages}


def _svg_document(title: str, body: str, width: int = 900, height: int = 600) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="30" y="35" font-family="sans-serif" font-size="20">{title}</text>'
        f'{body}</svg>'
    )


def _plot_lines_svg(
    path: Path,
    title: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    x_label: str,
    y_label: str,
    diagonal: bool = False,
) -> None:
    left, top, width, height = 80, 70, 760, 440
    elements = [
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="black"/>',
        f'<text x="{left + width / 2 - 25}" y="{top + height + 45}" font-family="sans-serif">{x_label}</text>',
        f'<text x="18" y="{top + height / 2}" transform="rotate(-90 18 {top + height / 2})" font-family="sans-serif">{y_label}</text>',
    ]
    if diagonal:
        elements.append(
            f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top}" stroke="#999" stroke-dasharray="5,5"/>'
        )
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    for color, (label, (x_values, y_values)) in zip(colors, curves.items()):
        points = []
        for x_value, y_value in zip(x_values, y_values):
            px = left + float(x_value) * width
            py = top + height - float(y_value) * height
            points.append(f"{px:.2f},{py:.2f}")
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = 85 + 22 * list(curves).index(label)
        elements.append(f'<line x1="680" y1="{legend_y}" x2="705" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="710" y="{legend_y + 5}" font-family="sans-serif" font-size="13">{label}</text>')
    path.write_text(_svg_document(title, "".join(elements)), encoding="utf-8")


def _plot_comparison_svg(path: Path, summary: dict[str, Any], external: dict[str, Any]) -> None:
    metrics = ("roc_auc", "pr_auc", "balanced_accuracy", "brier_score")
    left, top, bar_width, gap = 80, 80, 32, 42
    elements = []
    for index, metric in enumerate(metrics):
        x = left + index * 190
        elements.append(f'<text x="{x}" y="{top + 410}" font-family="sans-serif" font-size="12">{metric}</text>')
        values = []
        labels = []
        for model in MODEL_NAMES:
            values.append((summary[model].get(f"{metric}_mean"), f"A {model}"))
            values.append((external[model]["metrics"].get(metric), f"B {model}"))
        for position, (value, label) in enumerate(values):
            if value is None:
                continue
            y = top + 380 - float(value) * 350
            color = "#1f77b4" if label.startswith("A") else "#d62728"
            bx = x + position * (bar_width + 4)
            elements.append(f'<rect x="{bx}" y="{y}" width="{bar_width}" height="{top + 380 - y}" fill="{color}"/>')
    elements.append('<text x="80" y="540" font-family="sans-serif">Blue: Experiment A development-source mean; Red: Experiment B VA</text>')
    path.write_text(_svg_document("Random versus external evaluation", "".join(elements)), encoding="utf-8")


def _plot_missingness_svg(path: Path, quality: dict[str, Any]) -> None:
    left, top, width, height = 90, 70, 700, 420
    sources = list(quality["source_reports"])
    features = list(FEATURES)
    elements = []
    for index, feature in enumerate(features):
        x = left + index * (width / len(features))
        elements.append(f'<text x="{x}" y="{top + height + 35}" transform="rotate(45 {x} {top + height + 35})" font-family="sans-serif" font-size="11">{feature}</text>')
        for source_index, source in enumerate(sources):
            value = quality["source_reports"][source]["missing_percentages"][feature]
            bar_x = x + source_index * 5
            bar_height = float(value) / 100 * height
            color = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")[source_index]
            elements.append(f'<rect x="{bar_x}" y="{top + height - bar_height}" width="4" height="{bar_height}" fill="{color}"/>')
    for source_index, source in enumerate(sources):
        y = 35 + source_index * 18
        color = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")[source_index]
        elements.append(f'<rect x="700" y="{y - 10}" width="10" height="10" fill="{color}"/><text x="715" y="{y}" font-family="sans-serif" font-size="12">{source}</text>')
    path.write_text(_svg_document("Selected-feature missingness by source", "".join(elements)), encoding="utf-8")


def _write_research_report(output: Path, metrics: dict[str, Any], metadata: dict[str, Any], quality: dict[str, Any]) -> None:
    summary = metrics["experiment_a_summary"]
    external = metrics["experiment_b"]
    def fmt(value: Any) -> str:
        return "undefined" if value is None else f"{float(value):.4f}"
    lines = [
        "# GenRisk Research: UCI Heart Disease Source-Shift Study",
        "",
        "## Abstract",
        "",
        "This study evaluates how a hospital/source shift affects classical ML generalisation and reliability on a structured healthcare benchmark. Models are trained on Cleveland, Hungary, and Switzerland sources and evaluated both with repeated grouped stratified development splits and on the held-out Long Beach VA source.",
        "",
        "## Research question and motivation",
        "",
        "How does cross-source distribution shift affect the generalisation and reliability of classical ML models? The random development-source evaluation estimates an IID-style benchmark, while the VA evaluation measures external source generalisation.",
        "",
        "## Dataset and source description",
        "",
        f"The study used {metadata['source_row_counts']} records from the verified UCI Heart Disease archive. Features were {', '.join(FEATURES)}. The target was mapped as 0 versus 1–4. No records were excluded.",
        "",
        "## Data quality and missingness",
        "",
        f"Missingness and class distributions are recorded in `dataset_quality_report.json`. The duplicate audit found {metadata['duplicate_audit']['duplicate_group_count']} duplicate groups affecting {metadata['duplicate_audit']['duplicate_record_count']} records. Duplicate groups were kept intact for splitting and clustered bootstrap.",
        "",
        "## Experimental methodology",
        "",
        "Experiment A used five-fold StratifiedGroupKFold with ten fixed seeds, selecting one approximately 20% held-out fold per seed. Experiment B trained on all development records and evaluated once on Long Beach VA. Numeric values used training-only median imputation; categorical values used training-only mode imputation and one-hot encoding. Logistic Regression used training-only numeric standardisation; Random Forest did not.",
        "",
        "## Leakage prevention",
        "",
        "VA records were excluded from fitting, preprocessing, threshold selection, calibration, and model selection. Source identity was not a feature. Duplicate groups could not cross Experiment A partitions. Bootstrap resampling sampled duplicate groups as clusters.",
        "",
        "## Models and preprocessing",
        "",
        "The primary models were unweighted Logistic Regression and Random Forest with fixed parameters. The primary threshold for sensitivity, specificity, and balanced accuracy was 0.5.",
        "",
        "## Results",
        "",
        "### Experiment A: development-source repeated evaluation",
        "",
        "| Model | ROC-AUC mean ± SD | PR-AUC mean ± SD | Balanced accuracy mean ± SD | Brier mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        lines.append(
            f"| {model} | {fmt(summary[model]['roc_auc_mean'])} ± {fmt(summary[model]['roc_auc_std'])} | {fmt(summary[model]['pr_auc_mean'])} ± {fmt(summary[model]['pr_auc_std'])} | {fmt(summary[model]['balanced_accuracy_mean'])} ± {fmt(summary[model]['balanced_accuracy_std'])} | {fmt(summary[model]['brier_score_mean'])} ± {fmt(summary[model]['brier_score_std'])} |"
        )
    lines.extend([
        "",
        "### Experiment B: Long Beach VA external evaluation",
        "",
        "| Model | ROC-AUC | PR-AUC | Balanced accuracy | Sensitivity | Specificity | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model in MODEL_NAMES:
        values = external[model]["metrics"]
        lines.append(
            f"| {model} | {fmt(values['roc_auc'])} | {fmt(values['pr_auc'])} | {fmt(values['balanced_accuracy'])} | {fmt(values['sensitivity'])} | {fmt(values['specificity'])} | {fmt(values['brier_score'])} |"
        )
    lines.extend([
        "",
        "Confidence intervals are in `confidence_intervals.json`; invalid bootstrap replicate counts are reported there. Curves and source-missingness visualisations are SVG files in this directory.",
        "",
        "## Random versus external evaluation comparison",
        "",
        "Experiment A is a development-source random/group-stratified benchmark and Experiment B is a source-held-out VA evaluation. Any difference between them is evidence of benchmark sensitivity to source shift, but it cannot isolate population, measurement, missingness, or label-definition effects.",
        "",
        "## Reliability and calibration observations",
        "",
        "Brier score and probability outputs were recorded. This study did not fit a post-hoc calibration model, so calibration conclusions are limited to the uncalibrated model probabilities and their Brier performance.",
        "",
        "## Limitations",
        "",
        "The external VA sample is small and class-imbalanced. The dataset is historical, and source is confounded with population, clinical measurement, missingness, and label processes. Duplicate records reduce the effective sample size. The benchmark target is observed disease status rather than prospective clinical risk, so no clinical validity claim is made.",
        "",
        "## Conclusion",
        "",
        "The results support only the empirical comparison shown above. They should not be interpreted as clinical performance or as evidence that source shift has one isolated cause.",
        "",
        "## Reproducibility details",
        "",
        f"Seeds: {metadata['seeds']}. Bootstrap repetitions: {metadata['bootstrap_resamples']}. Dataset checksums, package versions, preprocessing configuration, duplicate audit, and source counts are in `run_metadata.json`.",
    ])
    (output / "research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_experiment_summary(output: Path, metrics: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        return "undefined" if value is None else f"{float(value):.4f}"

    lines = [
        "# UCI Heart Disease experiment summary",
        "",
        "Experiment A is the repeated grouped-stratified development-source baseline. Experiment B is the source-held-out Long Beach VA evaluation.",
        "",
        "## Experiment A: mean +/- standard deviation",
        "",
        "| Model | ROC-AUC | PR-AUC | Balanced accuracy | Sensitivity | Specificity | Brier score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        values = metrics["experiment_a_summary"][model]
        lines.append(
            f"| {model} | {fmt(values['roc_auc_mean'])} +/- {fmt(values['roc_auc_std'])} | {fmt(values['pr_auc_mean'])} +/- {fmt(values['pr_auc_std'])} | {fmt(values['balanced_accuracy_mean'])} +/- {fmt(values['balanced_accuracy_std'])} | {fmt(values['sensitivity_mean'])} +/- {fmt(values['sensitivity_std'])} | {fmt(values['specificity_mean'])} +/- {fmt(values['specificity_std'])} | {fmt(values['brier_score_mean'])} +/- {fmt(values['brier_score_std'])} |"
        )
    lines.extend([
        "",
        "## Experiment B: Long Beach VA",
        "",
        "| Model | Class counts | ROC-AUC | PR-AUC | Balanced accuracy | Sensitivity | Specificity | Brier score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for model in MODEL_NAMES:
        result = metrics["experiment_b"][model]
        values = result["metrics"]
        lines.append(
            f"| {model} | {values['class_counts']} | {fmt(values['roc_auc'])} | {fmt(values['pr_auc'])} | {fmt(values['balanced_accuracy'])} | {fmt(values['sensitivity'])} | {fmt(values['specificity'])} | {fmt(values['brier_score'])} |"
        )
    lines.extend([
        "",
        "Bootstrap confidence intervals and invalid replicate counts are recorded in `confidence_intervals.json`.",
    ])
    (output / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(
    output_dir: Path | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Run both approved experiments and write reproducible result artifacts."""

    seeds = tuple(int(seed) for seed in seeds)
    data = load_uci_heart_sources()
    development = development_data(data)
    external = external_data(data)
    output = output_dir or repository_root() / "results" / "uci_heart_disease"
    output.mkdir(parents=True, exist_ok=True)
    quality = dataset_quality_report(data)
    splits = repeated_group_stratified_splits(development, seeds)
    prediction_frames: list[pd.DataFrame] = []
    experiment_a: list[dict[str, Any]] = []
    experiment_b: dict[str, Any] = {}
    split_records: list[dict[str, Any]] = []

    for seed, train_index, test_index in splits:
        train = development.iloc[train_index]
        test = development.iloc[test_index]
        split_records.append({
            "seed": seed,
            "train_indices": [int(index) for index in train_index],
            "test_indices": [int(index) for index in test_index],
            "train_record_count": int(len(train)),
            "test_record_count": int(len(test)),
            "train_class_counts": {str(label): int((train[TARGET] == label).sum()) for label in (0, 1)},
            "test_class_counts": {str(label): int((test[TARGET] == label).sum()) for label in (0, 1)},
            "duplicate_groups_overlap": int(
                len(set(train["duplicate_group"]) & set(test["duplicate_group"]))
            ),
        })

    for model_name in MODEL_NAMES:
        for seed, train_index, test_index in splits:
            train = development.iloc[train_index]
            test = development.iloc[test_index]
            model = fit_model(model_name, train, random_state=seed)
            probabilities = model.predict_proba(test[list(FEATURES)])[:, 1]
            metrics = classification_metrics(test[TARGET], probabilities)
            experiment_a.append({"model": model_name, "seed": seed, **metrics})
            prediction_frames.append(
                _prediction_frame(test, probabilities, "A_random_group_stratified", model_name, seed)
            )

        model = fit_model(model_name, development, random_state=42)
        external_probabilities = model.predict_proba(external[list(FEATURES)])[:, 1]
        experiment_b[model_name] = {
            "metrics": classification_metrics(external[TARGET], external_probabilities),
            "confidence_intervals": stratified_group_bootstrap_ci(
                external[TARGET],
                external_probabilities,
                external["duplicate_group"],
                n_resamples=bootstrap_resamples,
            ),
        }
        prediction_frames.append(
            _prediction_frame(external, external_probabilities, "B_external_va", model_name, None)
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    experiment_a_summary: dict[str, dict[str, float]] = {}
    for model_name in MODEL_NAMES:
        model_runs = [run for run in experiment_a if run["model"] == model_name]
        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [run[metric_name] for run in model_runs if run[metric_name] is not None],
                dtype=float,
            )
            experiment_a_summary.setdefault(model_name, {})[f"{metric_name}_mean"] = float(values.mean())
            experiment_a_summary[model_name][f"{metric_name}_std"] = float(values.std(ddof=1))
    metrics_payload = {
        "experiment_a": experiment_a,
        "experiment_a_summary": experiment_a_summary,
        "experiment_b": experiment_b,
    }
    (output / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (output / "confidence_intervals.json").write_text(
        json.dumps(
            {model: result["confidence_intervals"] for model, result in experiment_b.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata = {
        "features": list(FEATURES),
        "target_mapping": {"0": 0, "1-4": 1},
        "development_sources": list(DEVELOPMENT_SOURCES),
        "external_source": EXTERNAL_SOURCE,
        "seeds": [int(seed) for seed in seeds],
        "bootstrap_resamples": int(bootstrap_resamples),
        "duplicate_audit": duplicate_audit(data),
        "source_row_counts": data.groupby("source").size().astype(int).to_dict(),
        "models": list(MODEL_NAMES),
        "weighted_models": False,
        "preprocessing_configuration": PREPROCESSING_CONFIGURATION,
        "model_parameters": {
            "logistic_regression": {"max_iter": 2000, "solver": "lbfgs", "class_weight": None},
            "random_forest": {"n_estimators": 300, "n_jobs": 1, "class_weight": None, "random_state_experiment_a": "split seed", "random_state_experiment_b": 42},
        },
        "dataset_checksums": _checksum_manifest(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "split_records": split_records,
        "bootstrap_method": "class-stratified sampling of duplicate groups as clusters",
    }
    (output / "dataset_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in MODEL_NAMES:
        for experiment, label in (("A_random_group_stratified", "A"), ("B_external_va", "B")):
            subset = predictions[(predictions["model"] == model_name) & (predictions["experiment"] == experiment)]
            if len(subset) and subset["true_target"].nunique() == 2:
                if experiment.startswith("A"):
                    curve_label = f"{label} {model_name} pooled"
                else:
                    curve_label = f"{label} {model_name} VA"
                fpr, tpr, _ = roc_curve(subset["true_target"], subset["predicted_probability"])
                curves[curve_label] = (fpr, tpr)
    _plot_lines_svg(output / "roc_curves.svg", "ROC curves", curves, "False-positive rate", "True-positive rate", diagonal=True)
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in MODEL_NAMES:
        for experiment, label in (("A_random_group_stratified", "A"), ("B_external_va", "B")):
            subset = predictions[(predictions["model"] == model_name) & (predictions["experiment"] == experiment)]
            if len(subset) and subset["true_target"].nunique() == 2:
                precision, recall, _ = precision_recall_curve(subset["true_target"], subset["predicted_probability"])
                pr_curves[f"{label} {model_name}"] = (recall, precision)
    _plot_lines_svg(output / "precision_recall_curves.svg", "Precision-recall curves", pr_curves, "Recall", "Precision")
    _plot_comparison_svg(output / "experiment_comparison.svg", experiment_a_summary, experiment_b)
    _plot_missingness_svg(output / "missingness_by_source.svg", quality)
    _write_experiment_summary(output, metrics_payload)
    _write_research_report(output, metrics_payload, metadata, quality)
    return {"metrics": metrics_payload, "metadata": metadata, "predictions": predictions}


if __name__ == "__main__":
    run_study()
