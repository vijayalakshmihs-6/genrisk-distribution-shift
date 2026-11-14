# Experimental protocol

Development sources are pooled Cleveland, Hungary, and Switzerland records.
Long Beach VA is an untouched external test source. The seven selected
features are `age`, `sex`, `cp`, `trestbps`, `chol`, `restecg`, and `thalach`.
The binary target is 0 versus 1–4.

Experiment A uses five-fold `StratifiedGroupKFold` to select one approximate
80/20 held-out fold for each of ten fixed seeds. Exact duplicate groups remain
within one partition. Preprocessing is refit for every split. Experiment B
fits only on all eligible development records and evaluates once on VA.

Models are unweighted Logistic Regression and Random Forest with explicit
fixed parameters. Primary metrics are ROC-AUC and PR-AUC; secondary metrics
are balanced accuracy, sensitivity, specificity, and Brier score. Confidence
intervals use duplicate-group-aware stratified bootstrap resampling where
applicable.
