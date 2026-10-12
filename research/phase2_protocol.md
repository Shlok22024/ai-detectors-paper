# Phase 2 protocol

Recorded before training on October 12, 2026, following owner authorization to proceed with Phase 2. This fixes the approved lexical baseline before any detector outcome is inspected.

- Inputs: Phase 1's checksum-verified corpus and frozen split manifest. Only `text` enters the classifier.
- Training: 3,000 abstract texts / 1,500 paired source items. Validation: 1,020 abstract texts / 510 paired items.
- Familiar final test: 1,980 abstract texts / 990 paired items. No news predictions will be produced in Phase 2.
- Representation: word unigrams and bigrams, maximum 30,000 features, minimum document frequency 2, lowercase, default two-or-more-character word tokenization, no stop-word list, smoothed IDF, raw term frequency and L2 row normalization. No additional cleaning, truncation, length feature or metadata feature.
- Estimator: logistic regression, `C=1`, L2 regularization (`l1_ratio=0` in the installed scikit-learn API), `solver=lbfgs`, `max_iter=1000`, `tol=1e-4`, fitted intercept, no class weighting, seed 42. No hyperparameter search or refit on validation data.
- Decision: AI is positive (`1`); predict AI at score >= 0.5. The threshold is fixed and will carry unchanged into Phase 3.
- Convergence warnings are fatal: resolve convergence using training/validation only before accessing the final test. Any parameter change must be recorded here before final evaluation.
- Evaluation: accuracy, positive-class precision/recall/F1, confusion counts, human false-positive rate, AI false-negative rate and ROC-AUC. Include sample sizes and class balance.
- Uncertainty: 1,000 percentile bootstrap draws, resampling intact leakage components within each evaluation split. Use seed 42 for validation and 43 for the familiar test. Report 95% intervals; these condition on the fitted model and do not capture retraining or domain uncertainty.
- Artifacts: local saved pipeline; text-free validation and familiar-test predictions; metric table; confusion matrix; runtime/parameter record; baseline report and paper-comparison table.

This is a new baseline within a methodological reproduction. The paper does not report a TF-IDF logistic regression baseline. Its neural in-domain scores use different architectures, token limits and original split membership, so they are contextual results, not matched performance targets.
