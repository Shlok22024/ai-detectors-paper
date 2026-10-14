"""Verify saved baseline outputs, then regenerate the report and confusion chart."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from prepare_data import ROOT, file_hash


def main():
    tables = ROOT / "outputs/tables"
    run = json.loads((tables / "baseline_run.json").read_text(encoding="utf-8"))
    for name in ["predictions", "metrics"]:
        if file_hash(tables / f"baseline_{name}.csv") != run[f"{name}_sha256"]:
            raise ValueError(f"Baseline {name} no longer matches the run record.")
    predictions = pd.read_csv(tables / "baseline_predictions.csv", float_precision="round_trip", dtype={"item_id": str})
    metrics = pd.read_csv(tables / "baseline_metrics.csv", float_precision="round_trip")
    if not predictions.row_id.is_unique or not predictions.domain.eq("abstracts").all():
        raise ValueError("Unexpected prediction scope or duplicated rows.")
    for row in metrics.itertuples(index=False):
        part = predictions.loc[predictions.split.eq(row.split)]
        np.testing.assert_array_equal(part.predicted_label, (part.score_ai >= row.threshold).astype(int))
        tn, fp, fn, tp = confusion_matrix(part.label, part.predicted_label, labels=[0,1]).ravel()
        for field, actual in {"tn":tn,"fp":fp,"fn":fn,"tp":tp,"n_texts":len(part),
                              "n_human":int(part.label.eq(0).sum()),"n_ai":int(part.label.eq(1).sum())}.items():
            assert getattr(row, field) == actual, (row.split, field)
        expected = {"accuracy":accuracy_score(part.label,part.predicted_label),
                    "precision":precision_score(part.label,part.predicted_label,zero_division=0),
                    "recall":recall_score(part.label,part.predicted_label),
                    "f1":f1_score(part.label,part.predicted_label),
                    "false_positive_rate":fp/(fp+tn),"false_negative_rate":fn/(fn+tp),
                    "roc_auc":roc_auc_score(part.label,part.score_ai)}
        for field, actual in expected.items():
            np.testing.assert_allclose(getattr(row, field), actual, atol=1e-14, rtol=0)
    result = metrics.set_index("split").loc["test_familiar"]
    validation = metrics.set_index("split").loc["validation"]
    matrix = np.array([[result.tn,result.fp],[result.fn,result.tp]],dtype=int)
    fig, ax = plt.subplots(figsize=(7,5.4))
    ax.imshow(matrix,cmap="Blues",vmin=0,vmax=int(matrix.max()))
    for (i,j), value in np.ndenumerate(matrix):
        label = [["Correct human","Human flagged as AI"],["AI missed","Correct AI"]][i][j]
        ax.text(j,i,f"{value:,}\n{label}",ha="center",va="center",fontsize=11,
                color="white" if value>matrix.max()/2 else "#183549")
    ax.set_xticks([0,1],["Human","AI"])
    ax.set_yticks([0,1],["Human","AI"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Released label")
    ax.set_title(f"Familiar test: {int(result.fp+result.fn):,} errors across {int(result.n_texts):,} abstracts",loc="left",pad=18,fontweight="bold")
    fig.text(.08,.035,f"{int(result.n_source_items):,} paired source items · {int(result.n_human):,} human / {int(result.n_ai):,} AI · threshold: {result.threshold:.1f}",fontsize=9,color="#53616D")
    fig.subplots_adjust(left=.17,right=.96,bottom=.18,top=.87)
    fig.savefig(ROOT / "outputs/figures/baseline_confusion_matrix.png",dpi=180,facecolor="white")
    plt.close(fig)
    # Original numbers checked in the official Table 7 before fitting this model.
    comparison = pd.DataFrame([
        {"experiment":"TF-IDF + logistic regression; Mistral zero-shot abstracts",
         "original_paper":"Not reported", "our_result":f"Accuracy {result.accuracy:.6f}",
         "comparable":"No matched original baseline", "explanation":"New lexical baseline; newly grouped split; different tokenization and detector."},
        {"experiment":"DeBERTa-V3-small; Mistral zero-shot abstracts",
         "original_paper":"Accuracy 0.9860 (Table 7)", "our_result":"Not run",
         "comparable":"No", "explanation":"Original neural architecture and split not reproduced; context only."},
        {"experiment":"XLM-RoBERTa-base; Mistral zero-shot abstracts",
         "original_paper":"Accuracy 0.9835 (Table 7)", "our_result":"Not run",
         "comparable":"No", "explanation":"Original neural architecture and split not reproduced; context only."},
    ])
    comparison.to_csv(tables / "paper_baseline_comparison.csv",index=False,lineterminator="\n")
    metric_rows = []
    names = {"accuracy":"Accuracy","precision":"AI precision","recall":"AI recall","f1":"AI F1",
             "false_positive_rate":"Human false-positive rate","false_negative_rate":"AI false-negative rate","roc_auc":"ROC-AUC"}
    for field,name in names.items():
        metric_rows.append(f"| {name} | {validation[field]:.6f} | {result[field]:.6f} | {result[field+'_ci_low']:.6f}–{result[field+'_ci_high']:.6f} |")
    comparison_rows = "\n".join("| " + " | ".join(map(str,row)) + " |" for row in comparison.itertuples(index=False,name=None))
    report = f"""# Baseline model: familiar-distribution evaluation

Phase 2 completed October 12, 2026. This historical report documents the familiar-domain lexical baseline within the approved methodological reproduction. It does not reproduce the original paper's neural classifier or original split assignments. Subsequent transfer results are in the [Phase 3 report](generalization_report.md).

## Main result

On **{int(result.n_texts):,} held-out abstracts** from **{int(result.n_source_items):,} source pairs**, accuracy is **{result.accuracy*100:.2f}%** and AI F1 is **{result.f1:.4f}**. The detector incorrectly flags **{int(result.fp)} of {int(result.n_human)} human abstracts** as AI ({result.false_positive_rate*100:.2f}%) and misses **{int(result.fn)} of {int(result.n_ai)} AI abstracts** ({result.false_negative_rate*100:.2f}%). Both classes are equally represented.

The source-pair bootstrap 95% accuracy interval is {result.accuracy_ci_low*100:.2f}%–{result.accuracy_ci_high*100:.2f}%; the human false-positive interval is {result.false_positive_rate_ci_low*100:.2f}%–{result.false_positive_rate_ci_high*100:.2f}%. Strong familiar-domain performance does not establish reliability on news, other generators or real-world human writing.

![Baseline confusion matrix](../outputs/figures/baseline_confusion_matrix.png)

## Fixed model and evaluation design

The [protocol](../research/phase2_protocol.md) was committed before model fitting. Its commit precedes the baseline implementation and any generated detector outcomes. The model fits **{run['training_texts']:,} abstract texts**, including {run['training_class_counts']['0']:,} human and {run['training_class_counts']['1']:,} AI texts, from {run['training_source_items']:,} paired sources. Abstract validation contains {int(validation.n_texts):,} texts from {int(validation.n_source_items):,} pairs. Familiar test contains {int(result.n_texts):,} texts from {int(result.n_source_items):,} pairs. Source items and linked duplicate components remain disjoint across splits.

A scikit-learn Pipeline fits TF-IDF only on training text, then logistic regression on the resulting sparse matrix. Representation: word unigrams/bigrams, `min_df=2`, at most 30,000 features, lowercasing, no stop-word list, smoothed IDF, raw term frequencies and L2 document normalization. The default token rule selects word sequences of at least two characters; it does not explicitly model capitalization or punctuation. There is no truncation, added length feature, released linguistic feature or provenance feature.

Logistic regression uses `C=1`, L2 regularization (`l1_ratio=0`), `solver=lbfgs`, `max_iter=1000`, `tol=1e-4`, an intercept, no class weighting and seed 42. AI is positive (`1`). Probabilities at or above **0.5** predict AI. This threshold was not tuned to the final test or to a desired false-positive rate. There was no hyperparameter search, no validation refit and no change to the prespecified configuration. Validation was evaluated before final test evaluation. No news text was transformed for scoring in Phase 2.

The fitted vocabulary contains {run['fitted_features']:,} features. The training matrix has {run['training_matrix_nonzero_values']:,} nonzero values. The optimizer converged in {run['solver_iterations'][0]} iterations without a convergence warning. The recorded fit time was {run['fit_seconds']:.2f} seconds and total fitting/evaluation/output time was {run['total_seconds']:.2f} seconds on the local Windows CPU, with numeric threads limited to one. These are observations from this run, not universal runtime guarantees.

## Metrics and uncertainty

| Metric | Validation | Familiar final test | Test 95% bootstrap interval |
| --- | --- | --- | --- |
{chr(10).join(metric_rows)}

Precision, recall and F1 use AI as positive. Human FPR is `FP / (FP + TN)`; AI FNR is `FN / (FN + TP)`. ROC-AUC uses continuous AI scores, not thresholded labels. Validation is a development check, not an additional final test. The difference between validation and test scores is descriptive and was not used to choose a different model.

Each interval uses 1,000 percentile bootstrap draws that resample whole leakage components, retaining both texts of every human/AI source pair. Seeds are 42 for validation and 43 for the familiar test. There are no linked multi-source components in this dataset subset, so the resampling unit is currently the source pair. The intervals condition on this fitted model and sampled abstract sources; they do not quantify uncertainty from retraining, generator choice or domain transfer. They are not hypothesis tests.

## Original-paper comparison

[Xia et al., EACL 2026](https://aclanthology.org/2026.eacl-long.307.pdf), Table 7, printed page 6542 (PDF page 19), reports the corresponding Mistral zero-shot abstract in-domain neural scores below. The row and column were checked against the rendered official table before training. No original neural detector was trained here.

| Experiment | Original paper | Our result | Comparable? | Explanation |
| --- | --- | --- | --- | --- |
{comparison_rows}

These are context, not a matched competition. Architecture, original split membership, sequence handling and implementation differ. We cannot attribute score differences to detector quality alone, claim neural-score reproduction or infer statistical agreement with the paper. Our Phase 3 experiment will adapt its cross-dataset evaluation logic using this already fitted lexical detector.

## Reproducibility and validation

After Phase 1 preparation, run `python src/train_baseline.py`, then `python src/report_baseline.py`. The pipeline is saved locally at `models/baseline.joblib`; it is ignored by Git and can be regenerated. `outputs/tables/baseline_predictions.csv` contains row IDs, source groups, released labels and scores/predictions without text. `baseline_metrics.csv` stores full-precision metrics and confidence intervals. `baseline_run.json` records input/output checksums, parameters, versions, convergence, the pre-run code commit and timings. The stored model produces identical validation probabilities after reloading.

The input corpus and manifest were checked against frozen Phase 1 checksums. Executed regression tests cover hand-calculated rates on an imbalanced example, threshold ties, no-positive predictions, intact-pair bootstrap sampling and unchanged vocabulary when transforming held-out words, in addition to the three preparation tests. The report script separately recounts confusion cells and recalculates metrics using saved predictions. A repeated run of the same fixed configuration produced byte-identical predictions and metric files; serialized model bytes differed, so the current saved model is identified by the checksum in the run record. These are internal code/output checks, not independent research validation or human authorship review.

## Limits and next phase

The familiar test consists of Mistral zero-shot outputs and released human abstracts. Class balance is artificial; precision would change with different prevalence. Labels inherit the release's provenance. Text-length differences and class-specific collection/cleaning artifacts remain possible shortcuts. The score is a model output, not a verified probability of authorship or evidence against an individual writer. No claim about native language, authorship subgroups or current commercial detectors is supported by these fields.

At the end of Phase 2, the central generalization question remained unanswered. The subsequent Phase 3 report applies this same detector and threshold to news and analyzes actual errors; the Phase 4 report interprets its learned associations. This report preserves the baseline checkpoint and its original measured results.

AI assistance was used for implementation, testing, source-table verification and drafting. Results are generated by the retained code and pinned dataset. The project owner has not been represented as manually reviewing every retained prediction, and no independent replication is claimed.
"""
    (ROOT / "report/baseline_report.md").write_text(report,encoding="utf-8",newline="\n")
    print("Saved-prediction metrics verified; report, comparison table and confusion chart generated.")


if __name__ == "__main__":
    main()
