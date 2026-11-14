# GenRisk Research

Technical study: **Evaluating the generalisation and reliability of classical
machine-learning models under distribution shift in healthcare datasets.**

This repository is a non-clinical research artifact. Healthcare data is used
as a structured benchmark; the results are not clinical predictions or a
validated medical system.

## Study design

The study uses the official UCI Heart Disease collection and preserves its
four sources: Cleveland, Hungary, Switzerland, and Long Beach VA. Cleveland,
Hungary, and Switzerland form the development pool. Long Beach VA is held out
as an external source. The seven features are `age`, `sex`, `cp`, `trestbps`,
`chol`, `restecg`, and `thalach`; the target maps 0 to 0 and 1–4 to 1.

The models are unweighted Logistic Regression and Random Forest. Numeric
features use training-only median imputation; categorical features use
training-only mode imputation and one-hot encoding. Logistic Regression also
uses training-only numeric standardisation. The primary study uses repeated
group-aware stratified development splits and an isolated VA evaluation.

ROC-AUC and PR-AUC are primary metrics. Balanced accuracy, sensitivity,
specificity, Brier score, calibration diagnostics, prevalence baselines, and
duplicate-group bootstrap uncertainty are also reported.

## Main empirical findings

Development mean ROC-AUC was 0.8646 for Logistic Regression and 0.8589 for
Random Forest. External Long Beach VA ROC-AUC fell to 0.6668 and 0.6556,
respectively. Phase 2 diagnostics found substantial source-associated feature
and missingness differences. These are descriptive findings, not causal or
clinical conclusions.

## Reproduction

From the repository root:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest -q
python -m genrisk_research.uci_heart_disease
python -m genrisk_research.uci_heart_diagnostics
```

The preserved outputs must not be overwritten when checking execution. Use a
temporary output directory and the module functions' `output_dir` parameters
for reruns. The raw archive and extracted source files are immutable research
inputs.

## Repository structure

- `src/genrisk_research/` — experiment and diagnostics modules.
- `tests/` — focused experiment and diagnostics tests.
- `data/raw/uci_heart_disease/` — original archive, extracted sources,
  checksums, and verification record.
- `results/uci_heart_disease/` — preserved first-study and Phase 2 outputs.
- `docs/` — provenance, protocol, and diagnostics documentation.

## Limitations

The data is historical, source/population/measurement processes are
confounded, the external source is small, and duplicate records reduce the
effective sample size. The target represents observed disease status rather
than prospective risk. No clinical validity is claimed.
