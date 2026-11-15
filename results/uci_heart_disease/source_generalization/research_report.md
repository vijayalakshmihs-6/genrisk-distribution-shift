# Source-held-out generalisation experiment

This experiment compares pooled development-source random evaluation with leave-one-source-out evaluation. It does not alter the primary study or make causal or clinical claims.

## Protocol

Cleveland, Hungary, and Switzerland are used as development sources. Each source is held out once while the other two are used for training. The random benchmark uses the existing ten-seed grouped stratified split procedure. All preprocessing is fit inside a fresh model pipeline using training records only.

## Results

| Model | Random ROC-AUC mean | Random ROC-AUC SD | Source-held-out ROC-AUC mean | Difference |
|---|---:|---:|---:|---:|
| logistic_regression | 0.8646 | 0.0280 | 0.8235 | 0.0411 |
| random_forest | 0.8589 | 0.0288 | 0.8063 | 0.0526 |

### Held-out folds

| Fold | Held-out source | Model | Records | ROC-AUC | Balanced accuracy | Brier score |
|---|---|---|---:|---:|---:|---:|
| holdout_cleveland | Cleveland | logistic_regression | 303 | 0.8287 | 0.7611 | 0.1687 |
| holdout_cleveland | Cleveland | random_forest | 303 | 0.8116 | 0.7298 | 0.1774 |
| holdout_hungary | Hungary | logistic_regression | 294 | 0.8244 | 0.7343 | 0.1809 |
| holdout_hungary | Hungary | random_forest | 294 | 0.8269 | 0.7561 | 0.1772 |
| holdout_switzerland | Switzerland | logistic_regression | 123 | 0.8174 | 0.7114 | 0.3021 |
| holdout_switzerland | Switzerland | random_forest | 123 | 0.7804 | 0.7446 | 0.2160 |

## Interpretation

Lower source-held-out performance than the random benchmark is consistent with source-associated generalisation degradation. The source folds differ in population, measurement, missingness, and label processes, so these results do not identify a causal source effect or establish universal model robustness.

## Limitations

The three held-out sources are small and heterogeneous, and the experiment uses the same historical binary disease-status target and seven-feature representation as the primary study. Random-split and source-held-out results answer different questions; their difference is a descriptive comparison rather than an unbiased causal effect.

## Reproducibility

Metrics, predictions, seeds, source assignments, and preprocessing conventions are recorded in `metrics.json`, `fold_metrics.csv`, `random_split_metrics.csv`, and `predictions.csv`. No held-out source was used for preprocessing, feature selection, threshold selection, or model selection.
