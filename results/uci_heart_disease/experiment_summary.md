# UCI Heart Disease experiment summary

Experiment A is the repeated grouped-stratified development-source baseline. Experiment B is the source-held-out Long Beach VA evaluation.

## Experiment A: mean +/- standard deviation

| Model | ROC-AUC | PR-AUC | Balanced accuracy | Sensitivity | Specificity | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8646 +/- 0.0280 | 0.8669 +/- 0.0310 | 0.8011 +/- 0.0365 | 0.7772 +/- 0.0569 | 0.8249 +/- 0.0418 | 0.1501 +/- 0.0182 |
| random_forest | 0.8589 +/- 0.0288 | 0.8571 +/- 0.0326 | 0.7808 +/- 0.0266 | 0.7454 +/- 0.0574 | 0.8162 +/- 0.0414 | 0.1539 +/- 0.0167 |

## Experiment B: Long Beach VA

| Model | Class counts | ROC-AUC | PR-AUC | Balanced accuracy | Sensitivity | Specificity | Brier score |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | {'0': 51, '1': 149} | 0.6668 | 0.8344 | 0.6290 | 0.8658 | 0.3922 | 0.1851 |
| random_forest | {'0': 51, '1': 149} | 0.6556 | 0.8230 | 0.6457 | 0.8993 | 0.3922 | 0.1770 |

Bootstrap confidence intervals and invalid replicate counts are recorded in `confidence_intervals.json`.
