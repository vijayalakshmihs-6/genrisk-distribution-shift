from __future__ import annotations

import numpy as np
import pandas as pd

from genrisk_research.uci_heart_disease import (
    CATEGORICAL_FEATURES,
    DEVELOPMENT_SOURCES,
    FEATURES,
    METRIC_NAMES,
    TARGET,
    build_model_pipeline,
    classification_metrics,
    development_data,
    duplicate_audit,
    external_data,
    load_uci_heart_sources,
    repeated_group_stratified_splits,
    stratified_group_bootstrap_ci,
)


def test_dataset_schema_target_mapping_and_source_counts() -> None:
    data = load_uci_heart_sources()
    assert tuple(FEATURES) == ("age", "sex", "cp", "trestbps", "chol", "restecg", "thalach")
    assert data.groupby("source").size().to_dict() == {
        "Cleveland": 303,
        "Hungary": 294,
        "Long Beach VA": 200,
        "Switzerland": 123,
    }
    assert set(data[TARGET].unique()) == {0, 1}
    assert data[TARGET].isna().sum() == 0


def test_duplicate_audit_retains_records_and_finds_internal_duplicates() -> None:
    data = load_uci_heart_sources()
    audit = duplicate_audit(data)
    assert audit["record_count"] == 920
    assert audit["duplicate_group_count"] == 2
    assert audit["duplicate_record_count"] == 4
    assert audit["within_source_duplicate_groups"] == 2
    assert audit["cross_source_duplicate_groups"] == 0


def test_experiment_a_keeps_duplicate_groups_together() -> None:
    development = development_data(load_uci_heart_sources())
    for _, train_index, test_index in repeated_group_stratified_splits(development, seeds=(1, 2, 3)):
        train_groups = set(development.iloc[train_index]["duplicate_group"])
        test_groups = set(development.iloc[test_index]["duplicate_group"])
        assert train_groups.isdisjoint(test_groups)
        assert set(development.iloc[train_index][TARGET]) == {0, 1}
        assert set(development.iloc[test_index][TARGET]) == {0, 1}


def test_group_stratified_splits_are_reproducible() -> None:
    development = development_data(load_uci_heart_sources())
    first = repeated_group_stratified_splits(development, seeds=(11, 12))
    second = repeated_group_stratified_splits(development, seeds=(11, 12))
    assert [(seed, train.tolist(), test.tolist()) for seed, train, test in first] == [
        (seed, train.tolist(), test.tolist()) for seed, train, test in second
    ]


def test_preprocessing_is_fitted_on_training_rows_only() -> None:
    data = development_data(load_uci_heart_sources())
    train = data.iloc[:200].copy()
    test = data.iloc[200:220].copy()
    train.loc[train.index[0], "age"] = np.nan
    test.loc[test.index[0], "age"] = 999.0
    pipeline = build_model_pipeline("logistic_regression")
    pipeline.fit(train[list(FEATURES)], train[TARGET])
    transformed = pipeline.named_steps["preprocess"].transform(test[list(FEATURES)])
    assert transformed.shape[0] == len(test)
    assert float(pipeline.named_steps["preprocess"].transformers_[0][1].named_steps["imputer"].statistics_[0]) != 999.0


def test_model_pipelines_are_unweighted_and_have_expected_preprocessing() -> None:
    logistic = build_model_pipeline("logistic_regression")
    forest = build_model_pipeline("random_forest")
    assert logistic.named_steps["model"].class_weight is None
    assert forest.named_steps["model"].class_weight is None
    assert set(CATEGORICAL_FEATURES) == {"sex", "cp", "restecg"}
    assert logistic.named_steps["preprocess"].transformers[0][0] == "numeric"


def test_external_data_isolated_from_development_sources() -> None:
    data = load_uci_heart_sources()
    development = development_data(data)
    external = external_data(data)
    assert set(development["source"]) == set(DEVELOPMENT_SOURCES)
    assert set(external["source"]) == {"Long Beach VA"}
    assert set(development["record_id"]).isdisjoint(external["record_id"])


def test_group_bootstrap_does_not_split_duplicate_groups() -> None:
    y = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    groups = np.array(["a", "a", "b", "c", "c"])
    result = stratified_group_bootstrap_ci(y, probabilities, groups, n_resamples=20, seed=9)
    assert result["resamples_valid"] == 20
    assert set(result["intervals"]) == set(METRIC_NAMES)


def test_metrics_are_reproducible() -> None:
    y = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    assert classification_metrics(y, probabilities) == classification_metrics(y, probabilities)
