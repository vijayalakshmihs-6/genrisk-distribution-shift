# Distribution-shift methodology audit

## Scope

This audit validates the source-level analysis in
`src/genrisk_research/distribution_shift.py` against the preserved UCI Heart
Disease files. It does not alter the raw data, model inputs, previous
experiment outputs, or target harmonisation.

## Data validation

The existing loader maps the four processed files explicitly:

| Source | File | Rows | Binary target counts (0, 1) |
|---|---|---:|---:|
| Cleveland | `processed.cleveland.data` | 303 | 164, 139 |
| Hungary | `processed.hungarian.data` | 294 | 188, 106 |
| Switzerland | `processed.switzerland.data` | 123 | 8, 115 |
| Long Beach VA | `processed.va.data` | 200 | 51, 149 |

The selected feature types come directly from the established schema:
numeric `age`, `trestbps`, `chol`, `thalach`; categorical `sex`, `cp`,
`restecg`. The loader converts `?` to missing values and maps targets 0 to 0
and 1–4 to 1. The loaded data contains 920 rows and 920 unique record IDs.
There are two exact duplicate groups containing four records total; both are
within-source and are retained. They are documented in
`distribution_shift_summary.json` and are not silently deduplicated.

## Sentinel audit and cholesterol investigation

The UCI documentation identifies `-9` as a missing-value code. No selected
feature contains `-9` after loading. The processed data also contains
impossible zeros for measured numeric variables:

- Switzerland `chol`: 123 zeros out of 123 rows.
- Long Beach VA `chol`: 49 zeros, plus 7 `?` values.
- Long Beach VA `trestbps`: 1 zero, plus 56 `?` values.

Zero remains a valid code for selected categorical variables, so categorical
zeros are not recoded. For this descriptive analysis only, impossible numeric
zeros and documented `-9` values are represented as missing. The loader and
all previous model experiments remain unchanged.

Before this correction, Cleveland cholesterol had:

- n = 303, missing = 0
- mean = 246.6931, standard deviation = 51.7769
- quartiles = 211, 241, 275; minimum = 126; maximum = 564

Switzerland had n = 123 raw zeros, mean = 0, standard deviation = 0, all
quartiles and the minimum/maximum equal to 0, and one unique value. Treating
those zeros as measurements produced the previous SMD 6.7381 and Wasserstein
distance 246.6931. Using the corrected sample-size-weighted pooled standard
deviation gives pooled SD 43.6975 and SMD 5.6455, but both are still invalid
as observed-cholesterol comparisons because the Switzerland values are
sentinels. The corrected analysis reports the Cleveland–Switzerland
cholesterol distances as undefined rather than fabricating a measured
distribution.

## Metric definitions and validation

- **SMD:** `(mean_A - mean_B) / pooled_SD`, with
  `pooled_SD = sqrt(((n_A - 1)s_A² + (n_B - 1)s_B²) / (n_A + n_B - 2))`.
  It is undefined when variance estimation is impossible or the pooled scale
  is zero with unequal means; equal constant means give SMD 0.
- **KS statistic:** maximum absolute difference between empirical cumulative
  distributions. It is bounded in `[0, 1]` and uses non-missing numeric values.
- **Wasserstein distance:** one-dimensional empirical transport distance in
  the original feature units. It is non-negative and scale-dependent.
- **Normalised Wasserstein:** Wasserstein distance divided by the pooled
  standard deviation when positive. It is undefined for a zero pooled scale
  unless the raw distance is also zero.
- **Total variation:** `0.5 * sum(|p_A(c) - p_B(c)|)` over the union of
  observed categories. It is bounded in `[0, 1]`.
- **Jensen–Shannon divergence:** symmetric divergence from the midpoint
  distribution, using base-2 logarithms. It is bounded in `[0, 1]` and aligns
  categories over the union of observed values.
- **Missingness difference:** absolute difference in source missingness
  percentages, reported in percentage points without imputation.
- **Prevalence difference:** absolute difference in positive-target
  proportions. It is outcome-composition evidence, not covariate-shift
  evidence.

Empty samples, absent categories, zero denominators, constant-valued features,
and undefined metrics are represented as null/undefined rather than replaced
with arbitrary values.

## Verified prevalence

The harmonised positive prevalence is Cleveland 139/303 = 45.87%, Hungary
106/294 = 36.05%, Switzerland 115/123 = 93.50%, and Long Beach VA 149/200 =
74.50%. These differences establish observed `P(Y)` differences only. They do
not establish concept shift `P(Y|X)`.

## Existing-model context

`source_performance_shift_context.csv` joins the corrected descriptive shift
summaries with the existing source-level performance and available reliability
metrics. It is descriptive only. Development-source predictions are repeated
split predictions and are not independent observations; Long Beach VA remains
external and was not used for model fitting or selection. No correlations or
causal feature explanations are calculated.

## Identified correction

The initial analysis incorrectly treated impossible zero cholesterol and blood
pressure values as observed numeric measurements. This inflated the reported
cholesterol contrast, especially the Cleveland–Switzerland result. The
correction is isolated to the distribution-shift analysis and is covered by
regression tests. Raw files, the loader's established harmonisation, model
configurations, predictions, and previous result directories were preserved.

## Scientific limits

The analysis establishes observable differences consistent with covariate
shift, missingness differences, and prevalence/label shift. It cannot identify
which difference caused performance deterioration and cannot establish concept
shift. The sources differ in population, measurement process, missingness
process, and label process. Distances and effect sizes are descriptive, not
causal evidence; no clinical validity claim is made.
