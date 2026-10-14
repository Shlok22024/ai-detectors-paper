# Phase 4: interpretation of the frozen lexical model

Recorded October 13, 2026 after Phase 3 approval. The abstract and news evaluation outcomes are already known; this is an interpretation plan, not a preregistration of those outcomes.

- Keep the Phase 2 model, vocabulary, final-test assignments and 0.5 threshold unchanged. Verify their checksums and agreement with saved scores before analysis. Do not fit another classifier, optimize a threshold, add SHAP or run new benchmarks.
- Select the 15 largest positive and 15 most negative logistic coefficients, with lexical tie-breaking. Report training document frequencies by class and presence in each final-test domain/class. Coefficient selection uses the already fitted training model, not target-test performance.
- Summarize final-test AI scores by domain and class. Plot empirical score distributions and the fixed threshold, preserving the distinction between ranking (news ROC-AUC 0.8073) and thresholded detection (news recall 0.20%).
- Compute the exact linear-score accounting `intercept + sum(TF-IDF_j * coefficient_j)`. Separate positive and negative coefficient contributions, verify reconstruction against the pipeline's decision function and probabilities, and summarize means by domain/class. This is arithmetic within the fitted model, not causal identification.
- For AI texts only, identify the 10 terms with the largest negative and 10 with the largest positive changes in mean weighted contribution from abstracts to news. Retain the full contribution sum for verification; publish the selected table. These are descriptive explanations of the score shift, not estimates of an isolated domain effect.
- Inspect the authors' generation descriptions and domain-specific zero-shot templates to distinguish controlled factors from bundled changes. Preserve model identity and prompt-style controls without claiming identical literal prompts or generation tasks. Human source corpora, text lengths and released class-specific cleaning differ.
- Produce two figures (coefficients and score distributions), compact tables and `report/interpretation_report.md`. Clearly separate observed associations, exact score accounting and unverified explanations. The word-token model has no explicit punctuation or capitalization features, so it cannot directly explain formatting effects.

No manual author review or independent validation is implied. This phase remains a narrow interpretation of the completed experiment; final portfolio packaging follows separately.
