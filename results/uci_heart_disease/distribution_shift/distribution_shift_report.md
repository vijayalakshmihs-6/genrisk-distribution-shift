# Source-level distribution-shift analysis

This analysis describes observable differences across the four UCI Heart Disease sources. It separates observed feature distributions, missingness, and target prevalence. It does not establish causes of model deterioration or concept shift.

## Definitions and method

For numeric features, the analysis reports source summaries, standardised mean difference SMD = (mean_A - mean_B) / pooled_SD, where pooled_SD = sqrt(((n_A - 1)s_A^2 + (n_B - 1)s_B^2) / (n_A + n_B - 2)), two-sample Kolmogorov–Smirnov statistic, Wasserstein distance, and Wasserstein distance divided by pooled_SD when that scale is non-zero. For categorical features, it reports category frequencies/proportions, total variation distance, and Jensen–Shannon divergence using base-2 logarithms. Missing values are excluded only from observed-value distribution metrics and are measured separately.

If pooled_SD is zero, SMD is 0 when the two means are equal and otherwise undefined; the normalised Wasserstein distance follows the same zero-scale rule. If there are insufficient observations to estimate sample variances, the standardised metrics are undefined.

The raw loader converts documented `?` tokens to missing values. This analysis additionally treats impossible zero cholesterol and zero resting-blood-pressure values, and documented -9 codes, as missing for descriptive summaries only. The raw data, loader, model inputs, and previous experiment outputs are unchanged. Zero remains a valid category code for selected categorical variables.

Covariate shift refers to changes in P(X); prevalence or label shift refers to changes in P(Y); concept shift refers to changes in P(Y|X). This analysis directly measures aspects of P(X), missingness, and P(Y). It cannot establish P(Y|X) changes.

## Largest observed differences

### Numeric features

| Comparison | Feature | Absolute SMD | KS statistic | Wasserstein | Normalised Wasserstein |
|---|---|---:|---:|---:|---:|
| Hungary vs Long Beach VA | age | 1.4751 | 0.5927 | 11.5235 | 1.4751 |
| Cleveland vs Long Beach VA | thalach | 1.1868 | 0.5160 | 26.8113 | 1.1868 |
| Cleveland vs Switzerland | thalach | 1.1784 | 0.4642 | 28.0499 | 1.1784 |
| Hungary vs Switzerland | age | 0.9147 | 0.4066 | 7.4905 | 0.9147 |
| Cleveland vs Hungary | age | 0.7819 | 0.3554 | 6.6124 | 0.7819 |
| Hungary vs Switzerland | thalach | 0.7227 | 0.3301 | 17.5723 | 0.7227 |
| Hungary vs Long Beach VA | thalach | 0.7080 | 0.3186 | 16.3338 | 0.7080 |
| Cleveland vs Long Beach VA | age | 0.5729 | 0.2619 | 4.9111 | 0.5729 |

### Categorical features

| Comparison | Feature | TV distance | JS divergence |
|---|---|---:|---:|
| Cleveland vs Hungary | restecg | 0.4680 | 0.2791 |
| Cleveland vs Long Beach VA | restecg | 0.4518 | 0.2762 |
| Cleveland vs Switzerland | restecg | 0.4311 | 0.2444 |
| Hungary vs Switzerland | cp | 0.3784 | 0.1613 |
| Hungary vs Long Beach VA | restecg | 0.4020 | 0.1316 |
| Cleveland vs Long Beach VA | sex | 0.2901 | 0.1197 |
| Hungary vs Long Beach VA | cp | 0.2905 | 0.0987 |
| Hungary vs Long Beach VA | sex | 0.2455 | 0.0949 |

### Missingness

| Comparison | Feature | Absolute difference (percentage points) |
|---|---|---:|
| Cleveland vs Switzerland | chol | 100.00 |
| Hungary vs Switzerland | chol | 92.18 |
| Switzerland vs Long Beach VA | chol | 72.00 |
| Cleveland vs Long Beach VA | trestbps | 28.50 |
| Hungary vs Long Beach VA | trestbps | 28.16 |
| Cleveland vs Long Beach VA | chol | 28.00 |
| Switzerland vs Long Beach VA | trestbps | 26.87 |
| Cleveland vs Long Beach VA | thalach | 26.50 |

### Target prevalence

| Source | Observations | Negative | Positive | Positive prevalence |
|---|---:|---:|---:|---:|
| Cleveland | 303.0 | 164.0 | 139.0 | 0.4587 |
| Hungary | 294.0 | 188.0 | 106.0 | 0.3605 |
| Switzerland | 123.0 | 8.0 | 115.0 | 0.9350 |
| Long Beach VA | 200.0 | 51.0 | 149.0 | 0.7450 |

## Interpretation

The measurements establish source-associated covariate and missingness differences, and the prevalence table establishes outcome-composition differences. They are relevant candidate explanations for why performance may vary across sources, but they do not identify which difference caused deterioration. External-source comparisons are descriptive and were not used for model fitting or selection.

## Limitations

The data is historical and the sources differ in population, measurement, missingness, and label processes. Standardised distances can be undefined for zero-variance features; scale-normalised Wasserstein distance is undefined when the pooled standard deviation is zero. Effect sizes and distances are descriptive and no p-values are used as evidence. These analyses do not establish concept shift, causality, or clinical validity.

The extreme raw cholesterol contrast between Cleveland and Switzerland is not a valid observed-cholesterol contrast: all 123 Switzerland cholesterol values are zero, while the Cleveland values are positive. The zero values are therefore treated as missing here; the analysis reports no cholesterol distribution distance for that pair rather than interpreting the sentinel as measured cholesterol.
