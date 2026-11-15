from __future__ import annotations

import hashlib

import numpy as np

from genrisk_research.robustness_analysis import (
    METRICS,
    bootstrap_metric_intervals,
    run_robustness_analysis,
)
from genrisk_research.uci_heart_disease import repository_root


def test_bootstrap_is_reproducible_and_has_valid_replicates() -> None:
    result_a = bootstrap_metric_intervals([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_resamples=50, seed=11)
    result_b = bootstrap_metric_intervals([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_resamples=50, seed=11)
    assert result_a == result_b
    assert set(result_a["metrics"]) == set(METRICS)
    assert result_a["metrics"]["roc_auc"]["valid_replicates"] == 50
    assert result_a["metrics"]["roc_auc"]["lower"] <= result_a["metrics"]["roc_auc"]["point_estimate"]


def test_grouped_bootstrap_keeps_duplicate_records_in_clusters() -> None:
    result = bootstrap_metric_intervals(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["g0", "g0", "g1", "g1"], n_resamples=30, seed=4
    )
    assert result["duplicate_group_count"] == 2
    assert result["metrics"]["accuracy"]["valid_replicates"] == 30


def test_one_class_bootstrap_reports_undefined_auc() -> None:
    result = bootstrap_metric_intervals([1, 1, 1], [0.1, 0.2, 0.3], n_resamples=20, seed=5)
    assert result["metrics"]["roc_auc"]["point_estimate"] is None
    assert result["metrics"]["roc_auc"]["valid_replicates"] == 0
    assert result["metrics"]["roc_auc"]["invalid_replicates"] == 20
    assert result["metrics"]["roc_auc"]["lower"] is None


def test_analysis_uses_existing_predictions_without_modifying_them(tmp_path) -> None:
    prediction_path = repository_root() / "results" / "uci_heart_disease" / "source_generalization" / "predictions.csv"
    before = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    result = run_robustness_analysis(tmp_path / "robustness", n_resamples=20, seed=9)
    after = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    assert before == after
    assert len(result["bootstrap"]) == 6
    assert {entry["model"] for entry in result["bootstrap"]} == {"logistic_regression", "random_forest"}
