from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from genrisk_research.distribution_shift import (
    _standardized_mean_difference,
    _js_divergence,
    feature_summaries,
    missingness_comparisons,
    pairwise_feature_distances,
    prepare_distribution_data,
    run_distribution_shift_analysis,
    sentinel_audit,
    target_prevalence_comparisons,
)
from genrisk_research.uci_heart_disease import load_uci_heart_sources, repository_root


def test_numeric_summary_matches_known_source_values() -> None:
    data = load_uci_heart_sources()
    summaries = feature_summaries(data)
    row = summaries[(summaries["source"] == "Cleveland") & (summaries["feature"] == "age")].iloc[0]
    expected = data.loc[data["source"].eq("Cleveland"), "age"].dropna()
    assert row["observed_count"] == len(expected)
    assert row["mean"] == expected.mean()
    assert row["median"] == expected.median()
    assert row["interquartile_range"] == expected.quantile(0.75) - expected.quantile(0.25)


def test_categorical_distance_and_js_are_correct() -> None:
    data = pd.DataFrame({"source": ["A"] * 4 + ["B"] * 4, "cp": [1, 1, 1, 2, 1, 2, 2, 2]})
    proportions_a = {"1": 0.75, "2": 0.25}
    proportions_b = {"1": 0.25, "2": 0.75}
    assert _js_divergence(proportions_a, proportions_b) > 0
    assert _js_divergence({"1": 1.0}, {"1": 1.0}) == 0.0
    assert 0.0 <= 0.5 * abs(0.75 - 0.25) + 0.5 * abs(0.25 - 0.75) <= 1.0


def test_missingness_is_not_imputed_and_comparisons_are_absolute() -> None:
    data = load_uci_heart_sources()
    data.loc[data.index[0], "chol"] = np.nan
    comparisons = missingness_comparisons(data)
    row = comparisons[(comparisons["source_a"] == "Cleveland") & (comparisons["source_b"] == "Hungary") & (comparisons["feature"] == "chol")].iloc[0]
    assert row["missing_count_a"] == 1
    assert row["absolute_missing_percentage_difference"] == abs(row["missing_percentage_a"] - row["missing_percentage_b"])
    assert data.loc[data.index[0], "chol"] != data.loc[data.index[0], "chol"]


def test_target_prevalence_has_source_and_pairwise_rows() -> None:
    data = load_uci_heart_sources()
    result = target_prevalence_comparisons(data)
    assert set(result["row_type"]) == {"source", "pairwise"}
    cleveland = result[(result["row_type"] == "source") & (result["source"] == "Cleveland")].iloc[0]
    assert cleveland["observation_count"] == 303
    assert cleveland["positive_count"] == 139
    assert cleveland["positive_prevalence"] == 139 / 303


def test_pairwise_comparisons_cover_all_sources_and_feature_types() -> None:
    data = load_uci_heart_sources()
    result = pairwise_feature_distances(data)
    assert len(result) == 6 * 7
    assert set(result["feature_type"]) == {"numeric", "categorical"}
    assert set(result["comparison_type"]) == {"development_development", "development_external"}


def test_constant_and_empty_values_are_handled_without_fabrication() -> None:
    assert _standardized_mean_difference(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    assert _standardized_mean_difference(np.array([1.0, 1.0]), np.array([2.0, 2.0])) is None
    assert _standardized_mean_difference(np.array([]), np.array([1.0])) is None


def test_smd_uses_sample_size_weighted_pooled_standard_deviation() -> None:
    # var(A)=4, var(B)=0; pooled variance=((3-1)*4+(5-1)*0)/(3+5-2)=4/3.
    expected = 2.0 / np.sqrt(4.0 / 3.0)
    assert _standardized_mean_difference(np.array([0.0, 2.0, 4.0]), np.zeros(5)) == expected


def test_impossible_numeric_zero_sentinels_are_analysis_missing_only() -> None:
    raw = load_uci_heart_sources()
    prepared = prepare_distribution_data(raw)
    assert int((raw.loc[raw["source"].eq("Switzerland"), "chol"] == 0).sum()) == 123
    assert prepared.loc[prepared["source"].eq("Switzerland"), "chol"].isna().all()
    assert int(prepared.loc[prepared["source"].eq("Long Beach VA"), "chol"].isna().sum()) == 56
    assert int(prepared.loc[prepared["source"].eq("Long Beach VA"), "trestbps"].isna().sum()) == 57
    audit = sentinel_audit(raw)
    assert {tuple(row[key] for key in ("source", "feature", "sentinel_value", "count")) for row in audit} >= {
        ("Switzerland", "chol", 0.0, 123),
        ("Long Beach VA", "chol", 0.0, 49),
        ("Long Beach VA", "trestbps", 0.0, 1),
    }
    assert int((raw.loc[raw["source"].eq("Switzerland"), "chol"] == 0).sum()) == 123


def test_cholesterol_extreme_result_is_not_reported_as_observed_distribution_shift() -> None:
    data = load_uci_heart_sources()
    distances = pairwise_feature_distances(data)
    row = distances[(distances["source_a"] == "Cleveland") & (distances["source_b"] == "Switzerland") & (distances["feature"] == "chol")].iloc[0]
    assert row["observed_count_a"] == 303
    assert row["observed_count_b"] == 0
    assert pd.isna(row["standardized_mean_difference"])
    assert pd.isna(row["kolmogorov_smirnov_statistic"])
    assert pd.isna(row["wasserstein_distance"])


def test_distance_bounds_and_category_alignment() -> None:
    data = load_uci_heart_sources()
    result = pairwise_feature_distances(data)
    categorical = result[result["feature_type"] == "categorical"]
    assert categorical["total_variation_distance"].dropna().between(0.0, 1.0).all()
    assert categorical["jensen_shannon_divergence"].dropna().between(0.0, 1.0).all()
    assert result["kolmogorov_smirnov_statistic"].dropna().between(0.0, 1.0).all()


def test_analysis_is_reproducible_and_does_not_modify_primary_inputs(tmp_path) -> None:
    primary = repository_root() / "results" / "uci_heart_disease" / "predictions.csv"
    before = hashlib.sha256(primary.read_bytes()).hexdigest()
    first = run_distribution_shift_analysis(tmp_path / "one")
    second = run_distribution_shift_analysis(tmp_path / "two")
    after = hashlib.sha256(primary.read_bytes()).hexdigest()
    assert before == after
    assert first == second
