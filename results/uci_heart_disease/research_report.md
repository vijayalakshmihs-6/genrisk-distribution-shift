# GenRisk Research: UCI Heart Disease Source-Shift Study

## Abstract

This study evaluates how a hospital/source shift affects classical ML generalisation and reliability on a structured healthcare benchmark. Models are trained on Cleveland, Hungary, and Switzerland sources and evaluated both with repeated grouped stratified development splits and on the held-out Long Beach VA source.

## Research question and motivation

How does cross-source distribution shift affect the generalisation and reliability of classical ML models? The random development-source evaluation estimates an IID-style benchmark, while the VA evaluation measures external source generalisation.

## Dataset and source description

The study used {'Cleveland': 303, 'Hungary': 294, 'Long Beach VA': 200, 'Switzerland': 123} records from the verified UCI Heart Disease archive. Features were age, sex, cp, trestbps, chol, restecg, thalach. The target was mapped as 0 versus 1–4. No records were excluded.

## Data quality and missingness

Missingness and class distributions are recorded in `dataset_quality_report.json`. The duplicate audit found 2 duplicate groups affecting 4 records. Duplicate groups were kept intact for splitting and clustered bootstrap.

## Experimental methodology

Experiment A used five-fold StratifiedGroupKFold with ten fixed seeds, selecting one approximately 20% held-out fold per seed. Experiment B trained on all development records and evaluated once on Long Beach VA. Numeric values used training-only median imputation; categorical values used training-only mode imputation and one-hot encoding. Logistic Regression used training-only numeric standardisation; Random Forest did not.

## Leakage prevention

VA records were excluded from fitting, preprocessing, threshold selection, calibration, and model selection. Source identity was not a feature. Duplicate groups could not cross Experiment A partitions. Bootstrap resampling sampled duplicate groups as clusters.

## Models and preprocessing

The primary models were unweighted Logistic Regression and Random Forest with fixed parameters. The primary threshold for sensitivity, specificity, and balanced accuracy was 0.5.

## Results

### Experiment A: development-source repeated evaluation

| Model | ROC-AUC mean ± SD | PR-AUC mean ± SD | Balanced accuracy mean ± SD | Brier mean ± SD |
|---|---:|---:|---:|---:|
| logistic_regression | 0.8646 ± 0.0280 | 0.8669 ± 0.0310 | 0.8011 ± 0.0365 | 0.1501 ± 0.0182 |
| random_forest | 0.8589 ± 0.0288 | 0.8571 ± 0.0326 | 0.7808 ± 0.0266 | 0.1539 ± 0.0167 |

### Experiment B: Long Beach VA external evaluation

| Model | ROC-AUC | PR-AUC | Balanced accuracy | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.6668 | 0.8344 | 0.6290 | 0.8658 | 0.3922 | 0.1851 |
| random_forest | 0.6556 | 0.8230 | 0.6457 | 0.8993 | 0.3922 | 0.1770 |

Confidence intervals are in `confidence_intervals.json`; invalid bootstrap replicate counts are reported there. Curves and source-missingness visualisations are SVG files in this directory.

## Random versus external evaluation comparison

Experiment A is a development-source random/group-stratified benchmark and Experiment B is a source-held-out VA evaluation. Any difference between them is evidence of benchmark sensitivity to source shift, but it cannot isolate population, measurement, missingness, or label-definition effects.

## Reliability and calibration observations

Brier score and probability outputs were recorded. This study did not fit a post-hoc calibration model, so calibration conclusions are limited to the uncalibrated model probabilities and their Brier performance.

## Limitations

The external VA sample is small and class-imbalanced. The dataset is historical, and source is confounded with population, clinical measurement, missingness, and label processes. Duplicate records reduce the effective sample size. The benchmark target is observed disease status rather than prospective clinical risk, so no clinical validity claim is made.

## Conclusion

The results support only the empirical comparison shown above. They should not be interpreted as clinical performance or as evidence that source shift has one isolated cause.

## Reproducibility details

Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. Bootstrap repetitions: 2000. Dataset checksums, package versions, preprocessing configuration, duplicate audit, and source counts are in `run_metadata.json`.
