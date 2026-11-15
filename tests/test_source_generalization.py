from __future__ import annotations

import numpy as np

from genrisk_research.source_generalization import (
    SOURCE_HOLDOUT_FOLDS,
    _extended_metrics,
    evaluate_model,
    run_source_generalization,
    source_holdout_partitions,
)
from genrisk_research.uci_heart_disease import FEATURES, TARGET, build_model_pipeline, load_uci_heart_sources


def test_all_source_holdout_assignments_are_exact_and_disjoint() -> None:
    data = load_uci_heart_sources()
    partitions = source_holdout_partitions(data)
    assert set(partitions) == set(SOURCE_HOLDOUT_FOLDS)
    for fold, (train, test) in partitions.items():
        expected_train, expected_test = SOURCE_HOLDOUT_FOLDS[fold]
        assert set(train["source"]) == set(expected_train)
        assert set(test["source"]) == {expected_test}
        assert set(train["record_id"]).isdisjoint(test["record_id"])
        assert set(train[TARGET]) == {0, 1}
        assert set(test[TARGET]) == {0, 1}


def test_source_holdout_preprocessing_uses_training_rows_only() -> None:
    data = load_uci_heart_sources()
    train, test = source_holdout_partitions(data)["holdout_cleveland"]
    train = train.copy()
    test = test.copy()
    train.loc[train.index[0], "age"] = np.nan
    test.loc[test.index[0], "age"] = 999.0
    pipeline = build_model_pipeline("logistic_regression")
    pipeline.fit(train[list(FEATURES)], train[TARGET])
    age_imputer = pipeline.named_steps["preprocess"].transformers_[0][1].named_steps["imputer"]
    assert float(age_imputer.statistics_[0]) != 999.0
    metrics, probabilities = evaluate_model("logistic_regression", train, test, random_state=17)
    assert len(probabilities) == len(test)
    assert metrics["record_count"] == len(test)


def test_source_holdout_evaluation_is_reproducible() -> None:
    data = load_uci_heart_sources()
    train, test = source_holdout_partitions(data)["holdout_hungary"]
    first_metrics, first_probabilities = evaluate_model("random_forest", train, test, random_state=23)
    second_metrics, second_probabilities = evaluate_model("random_forest", train, test, random_state=23)
    assert first_metrics == second_metrics
    assert np.array_equal(first_probabilities, second_probabilities)


def test_undefined_metrics_are_reported_as_none() -> None:
    metrics = _extended_metrics([1, 1, 1], [0.1, 0.2, 0.3])
    assert metrics["roc_auc"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["precision"] is None
    assert metrics["f1_score"] is None
    assert metrics["accuracy"] == 0.0


def test_run_evaluates_both_models_in_each_fold(tmp_path) -> None:
    payload = run_source_generalization(tmp_path / "source_generalization", seeds=(7,))
    rows = payload["source_holdout"]["folds"]
    assert len(rows) == len(SOURCE_HOLDOUT_FOLDS) * 2
    assert {(row["evaluation"], row["model"]) for row in rows} == {
        (fold, model) for fold in SOURCE_HOLDOUT_FOLDS for model in ("logistic_regression", "random_forest")
    }
    assert len(payload["random_split_benchmark"]["runs"]) == 2
