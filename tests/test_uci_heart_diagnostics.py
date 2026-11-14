from __future__ import annotations

import numpy as np
import pandas as pd

from genrisk_research.uci_heart_diagnostics import (
    _calibration_curve,
    _prevalence_references,
    jensen_shannon_divergence,
    missingness_analysis,
    standardized_mean_difference,
)
from genrisk_research.uci_heart_disease import FEATURES, load_uci_heart_sources


def test_standardized_mean_difference_is_zero_for_identical_values() -> None:
    values = pd.Series([1.0, 2.0, 3.0])
    assert standardized_mean_difference(values, values) == 0.0


def test_jensen_shannon_divergence_is_symmetric_and_zero_for_identical_distributions() -> None:
    left = {"a": 0.75, "b": 0.25}
    right = {"a": 0.25, "b": 0.75}
    assert jensen_shannon_divergence(left, left) == 0.0
    assert jensen_shannon_divergence(left, right) == jensen_shannon_divergence(right, left)


def test_missingness_analysis_reports_all_features_and_complete_records() -> None:
    report = missingness_analysis(load_uci_heart_sources())
    assert set(report["by_source"]) == {"Cleveland", "Hungary", "Switzerland", "Long Beach VA"}
    assert set(report["by_source"]["Long Beach VA"]["feature_missing_counts"]) == set(FEATURES)
    assert report["by_source"]["Cleveland"]["complete_record_count"] == 303
    assert report["by_source"]["Long Beach VA"]["complete_record_count"] == 140
    assert report["largest_va_vs_development_differences"]["trestbps"] > 20


def test_calibration_bins_are_ordered_and_cover_all_records() -> None:
    y = pd.Series([0, 1, 0, 1, 1, 0])
    probabilities = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    curve = _calibration_curve(y, probabilities, bins=3)
    assert len(curve) == 3
    assert sum(point["count"] for point in curve) == len(y)
    assert [point["mean_predicted_probability"] for point in curve] == sorted(
        point["mean_predicted_probability"] for point in curve
    )


def test_prevalence_references_are_defined_from_labels() -> None:
    references = _prevalence_references(pd.Series([0, 0, 1, 1]))
    assert references["prevalence"] == 0.5
    assert references["prevalence_only_brier"] == 0.25
    assert references["prevalence_pr_auc_baseline"] == 0.5
