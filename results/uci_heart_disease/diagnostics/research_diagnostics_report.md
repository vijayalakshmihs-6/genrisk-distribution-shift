# Phase 2 diagnostic investigation: UCI Heart Disease

## Summary of the first study

The primary study found development-source mean ROC-AUC of 0.8646 for Logistic Regression and 0.8589 for Random Forest, versus 0.6668 and 0.6556 on the held-out Long Beach VA source. This phase investigates descriptive explanations without changing the original protocol or results.

## Feature distribution shift

The machine-readable distribution comparisons are in `distribution_shift.json`. Numeric summaries include means, standard deviations, medians, interquartile ranges, and standardized mean differences. Categorical summaries include proportions, absolute differences, and Jensen-Shannon divergence.

The largest standardized differences and categorical divergences should be interpreted as source-associated distribution differences, not causal explanations. Source is not used as a model feature.


| Numeric feature | VA versus pooled-development SMD |
|---|---:|
| age | 0.8746 |
| trestbps | 0.0975 |
| chol | -0.2331 |
| thalach | -0.7438 |

| Categorical feature | VA versus pooled-development JS divergence |
|---|---:|
| sex | 0.0872 |
| cp | 0.0377 |
| restecg | 0.1104 |
## Missingness differences

Missingness counts, percentages, complete-record percentages, and missingness patterns are in `missingness_analysis.json`. The analysis identifies whether VA missingness differs from pooled development missingness, but cannot establish that missingness caused performance deterioration.


Largest absolute VA versus pooled-development missingness differences:

| Feature | Difference in missingness percentage points |
|---|---:|
| trestbps | 27.58 |
| thalach | 26.22 |
| chol | 0.31 |
## Source-specific performance

Source-specific Experiment A results use the existing repeated-split predictions. Because records appear in multiple repeated test splits, these descriptive rows are not independent observations. The small Switzerland source should not be treated as conclusive.


| Source | Model | Unique records | Class counts | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---|---:|---:|---:|
| Cleveland | logistic_regression | 210 | {'0': 315, '1': 302} | 0.8550 | 0.8296 | 0.1548 |
| Cleveland | random_forest | 210 | {'0': 315, '1': 302} | 0.8286 | 0.8238 | 0.1703 |
| Hungary | logistic_regression | 201 | {'0': 387, '1': 245} | 0.8154 | 0.7208 | 0.1712 |
| Hungary | random_forest | 201 | {'0': 387, '1': 245} | 0.8264 | 0.7022 | 0.1644 |
| Switzerland | logistic_regression | 73 | {'0': 11, '1': 173} | 0.8760 | 0.9912 | 0.0617 |
| Switzerland | random_forest | 73 | {'0': 11, '1': 173} | 0.7782 | 0.9841 | 0.0624 |
## Reliability analysis

Saved probability distributions, ROC/PR metrics, calibration bins, Brier scores, prevalence-only Brier references, and prevalence PR baselines are in `reliability_analysis.json`. VA prevalence is 74.5%, so a prevalence-only Brier reference is 0.1900 and the PR baseline is 0.7450. These baselines make the high VA PR-AUC values less informative than they appear without prevalence context.

| Model | VA ROC-AUC | VA PR-AUC | VA Brier | VA prevalence-only Brier |
|---|---:|---:|---:|---:|
| logistic_regression | 0.6668 | 0.8344 | 0.1851 | 0.1900 |
| random_forest | 0.6556 | 0.8230 | 0.1770 | 0.1900 |

## Complete-case sensitivity analysis

The complete-case sensitivity analysis retained 832 of 920 records and excluded 88 records only for this sensitivity analysis. The primary imputation-based study was not changed, and VA was not used to select a preferred approach.


| Model | Complete-case VA records | Class counts | ROC-AUC | PR-AUC | Balanced accuracy | Brier |
|---|---:|---|---:|---:|---:|---:|
| logistic_regression | 140 | {'0': 32, '1': 108} | 0.6117 | 0.8332 | 0.6036 | 0.1790 |
| random_forest | 140 | {'0': 32, '1': 108} | 0.5946 | 0.8214 | 0.6331 | 0.1704 |
## What the evidence supports

The results support that the held-out VA source differs in feature distributions and missingness patterns from development sources, and that external discrimination and threshold performance are lower than the random development-source benchmark. Reliability and prevalence-aware baselines must be considered when interpreting VA PR-AUC and Brier scores.

## What remains uncertain

Source, population, measurement, missingness, and label processes are confounded. These diagnostics cannot identify which factor caused deterioration. The small VA sample and small Switzerland source also produce unstable subgroup estimates.

## Limitations

The dataset is historical, the target is observed disease status rather than prospective risk, and duplicate records reduce effective sample size. Complete-case results change the analysis population and are sensitivity evidence only. No clinical validity claim is made.

## Follow-up hypotheses

1. Source-level feature and missingness shifts jointly explain a material portion of the external reliability deterioration.
2. Calibration degradation under source shift is more pronounced than ranking degradation for at least one classical model.
3. Source-aware, leakage-safe missingness handling may improve cross-source reliability, but this requires a separately pre-registered experiment and must not use VA for selection.
