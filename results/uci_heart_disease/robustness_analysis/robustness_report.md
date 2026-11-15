# Statistical uncertainty and robustness analysis

This analysis quantifies conditional test-sample uncertainty for the source-held-out predictions and describes variability across the fixed random-split seeds. It does not replace external validation or quantify training/model-selection uncertainty.

## Bootstrap method

A 2,000-replicate 95% percentile bootstrap was applied separately within each held-out source and model. Duplicate groups were sampled as clusters, and clusters were sampled separately within each class. Undefined metric replicates were counted as invalid rather than imputed. This preserves class representation but conditions the interval on the observed class composition and does not capture training or source-selection uncertainty.

## Random-split uncertainty

The random-split intervals summarize the ten fixed seeds. They describe split-to-split variability, not ten independent datasets and not an independent external-validation interval.

## Source robustness

| Model | Metric | Random mean | Source-held-out mean | Source range | All sources worse than random mean |
|---|---|---:|---:|---:|---|
| logistic_regression | roc_auc | 0.8646 | 0.8235 | 0.0114 | True |
| logistic_regression | balanced_accuracy | 0.8011 | 0.7356 | 0.0496 | True |
| logistic_regression | sensitivity | 0.7772 | 0.6660 | 0.1975 | True |
| logistic_regression | specificity | 0.8249 | 0.8052 | 0.1516 | False |
| logistic_regression | brier_score | 0.1501 | 0.2173 | 0.1334 | True |
| random_forest | roc_auc | 0.8589 | 0.8063 | 0.0465 | True |
| random_forest | balanced_accuracy | 0.7808 | 0.7435 | 0.0263 | True |
| random_forest | sensitivity | 0.7454 | 0.7382 | 0.1661 | False |
| random_forest | specificity | 0.8162 | 0.7488 | 0.1134 | True |
| random_forest | brier_score | 0.1539 | 0.1902 | 0.0388 | True |

## Interpretation

Logistic Regression has a smaller source-held-out ROC-AUC range than Random Forest in this analysis. That is a descriptive consistency result, not evidence that Logistic Regression is universally more robust. Metric-specific patterns differ: discrimination, threshold classification, and calibration should not be conflated.

## Limitations

The intervals are conditional on fixed fitted-model predictions and the observed held-out samples. They do not include uncertainty from model training, feature selection, hyperparameter selection, source selection, or dataset shift itself. Stratified cluster resampling preserves class representation and duplicate records, but it conditions on the observed prevalence and may not represent future-source prevalence uncertainty. No causal or clinical claim is made.
