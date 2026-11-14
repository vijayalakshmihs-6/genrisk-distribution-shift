# UCI Heart Disease source verification

Official source: <https://archive.ics.uci.edu/dataset/45/heart%2Bdisease>

- Archive URL: <https://archive.ics.uci.edu/static/public/45/heart%2Bdisease.zip>
- DOI: `10.24432/C52P4X`
- Licence: CC BY 4.0
- Retrieval date: 2026-11-13
- Raw archive and extracted files are retained unchanged.

## Sources and targets

The official `heart-disease.names` file reports 76 attributes including the
target and identifies these source files:

| Source | Rows | Target distribution | Binary 0 | Binary 1–4 |
|---|---:|---|---:|---:|
| Cleveland | 303 | 0:164, 1:55, 2:36, 3:35, 4:13 | 164 | 139 |
| Hungary | 294 | 0:188, 1:106 | 188 | 106 |
| Switzerland | 123 | 0:8, 1:48, 2:32, 3:30, 4:5 | 8 | 115 |
| Long Beach VA | 200 | 0:51, 1:56, 2:41, 3:42, 4:10 | 51 | 149 |

The processed files expose the same 13 features plus `target`: `age`, `sex`,
`cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`,
`slope`, `ca`, and `thal`. The documented binary conversion `0 -> 0` and
`1,2,3,4 -> 1` is consistent across the four files.

## Missingness and feasibility

Complete rows for all 13 features are Cleveland 297/303, Hungary 1/294,
Switzerland 0/123, and Long Beach VA 1/200. The largest gaps are `ca`, `thal`,
and `slope`, especially in Hungary and Long Beach VA. Therefore, complete-case
use of all 13 features is not feasible.

For analysis only, omitting `slope`, `ca`, and `thal` gives complete rows of
303/303, 261/294, 46/123, and 130/200. A conservative seven-feature set
(`age`, `sex`, `cp`, `trestbps`, `chol`, `restecg`, `thalach`) gives 303/303,
269/294, 120/123, and 140/200. Alternatively, missing-value imputation must be
fit on the training sources only. No raw rows are removed from this archive.

The source-shift design is conceptually valid: train on Cleveland, Hungary,
and Switzerland, then test on Long Beach VA. It is feasible only with a
pre-specified reduced feature set or training-only imputation. The VA test set
is small and positively imbalanced, so per-source metrics and uncertainty
intervals are required.

No models were trained and no experiments were run.
