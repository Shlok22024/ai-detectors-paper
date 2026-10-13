"""Verify Phase 3 outputs and generate its charts and research report."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from prepare_data import ROOT, file_hash
from train_baseline import METRICS, binary_metrics

COLORS = {"abstracts":"#2A6381","news":"#D17A43"}
DOMAIN_NAMES = {"abstracts":"Familiar abstracts","news":"Unseen news"}


def table(frame):
    lines = ["| " + " | ".join(map(str,frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(map(str,row)) + " |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def save(fig,name,footer):
    fig.text(.06,.025,footer,fontsize=8,color="#53616D")
    fig.savefig(ROOT / "outputs/figures" / name,dpi=180,facecolor="white")
    plt.close(fig)


def main():
    tables = ROOT / "outputs/tables"
    record = json.loads((tables / "generalization_run.json").read_text(encoding="utf-8"))
    for name,expected in record["output_sha256"].items():
        if file_hash(tables / f"generalization_{name}.csv") != expected:
            raise ValueError(f"Phase 3 {name} checksum mismatch.")
    predictions = pd.read_csv(tables / "generalization_predictions.csv",float_precision="round_trip",dtype={"item_id":str})
    metrics = pd.read_csv(tables / "generalization_metrics.csv",float_precision="round_trip").set_index("domain")
    changes = pd.read_csv(tables / "generalization_changes.csv",float_precision="round_trip").set_index("metric")
    bins = pd.read_csv(tables / "error_rates_by_length.csv")
    errors = pd.read_csv(tables / "domain_error_rates.csv").set_index(["domain","error_type"])
    examples = pd.read_csv(tables / "error_example_selection.csv",dtype={"item_id":str})
    notes = json.loads((ROOT / "research/error_example_notes.json").read_text(encoding="utf-8"))
    note_map = {note["row_id"]:note["description"] for note in notes["examples"]}
    if set(note_map) != set(examples.row_id):
        raise ValueError("Example descriptions do not match the selected IDs.")
    for domain in ["abstracts","news"]:
        part = predictions.loc[predictions.domain.eq(domain)]
        actual = binary_metrics(part.label,part.score_ai)
        for name in METRICS+["tn","fp","fn","tp","n_texts","n_human","n_ai"]:
            np.testing.assert_allclose(metrics.loc[domain,name],actual[name],atol=1e-14,rtol=0)
        for label,kind in [(0,"false_positive"),(1,"false_negative")]:
            cells = bins.loc[bins.domain.eq(domain)&bins.error_type.eq(kind)]
            assert cells.n_class_texts.sum()==int(part.label.eq(label).sum())
            assert cells.n_errors.sum()==int((part.label.eq(label)&part.predicted_label.ne(label)).sum())
    familiar,news = metrics.loc["abstracts"],metrics.loc["news"]
    for metric in METRICS:
        np.testing.assert_allclose(changes.loc[metric,"news_minus_abstracts"],news[metric]-familiar[metric],atol=1e-14,rtol=0)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fields = ["accuracy","recall","f1","roc_auc"]
    fig,ax = plt.subplots(figsize=(9,5.4))
    x = np.arange(len(fields))
    for domain,offset in [("abstracts",-.19),("news",.19)]:
        row = metrics.loc[domain]
        values = np.array([row[field] for field in fields])
        lows = np.array([row[field+"_ci_low"] for field in fields])
        highs = np.array([row[field+"_ci_high"] for field in fields])
        bars = ax.bar(x+offset,values,.34,color=COLORS[domain],label=DOMAIN_NAMES[domain],
                      yerr=[values-lows,highs-values],capsize=3,error_kw={"elinewidth":1})
        ax.bar_label(bars,labels=[f"{v:.3f}" for v in values],padding=9,fontsize=9)
    ax.set_xticks(x,["Accuracy","AI recall","AI F1","ROC-AUC"])
    ax.set_ylabel("Metric value (0–1)")
    ax.set_ylim(0,1.18)
    ax.set_yticks(np.arange(0,1.01,.2))
    ax.set_title("The fixed detector loses detection ability on news",loc="left",fontweight="bold",pad=15)
    ax.legend(frameon=False,loc="upper right",ncol=2)
    ax.yaxis.grid(alpha=.15)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=.17,top=.87)
    save(fig,"generalization_performance.png","1,980 texts / 990 source pairs per domain. Same model and 0.5 threshold. Error bars: source-component bootstrap 95% intervals.")

    matrix = np.array([[news.tn,news.fp],[news.fn,news.tp]],dtype=int)
    fig,ax = plt.subplots(figsize=(7,5.4))
    ax.imshow(matrix,cmap="Blues",vmin=0,vmax=int(matrix.max()))
    labels = [["Correct human","Human flagged as AI"],["AI missed","Correct AI"]]
    for (i,j),value in np.ndenumerate(matrix):
        ax.text(j,i,f"{value:,}\n{labels[i][j]}",ha="center",va="center",fontsize=11,
                color="white" if value>matrix.max()/2 else "#183549")
    ax.set_xticks([0,1],["Human","AI"])
    ax.set_yticks([0,1],["Human","AI"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Released label")
    ax.set_title(f"Unseen news: {int(news.fn):,} of {int(news.n_ai):,} AI texts missed",loc="left",fontweight="bold",pad=18)
    fig.subplots_adjust(left=.17,right=.96,bottom=.18,top=.87)
    save(fig,"news_confusion_matrix.png","Frozen news test: 990 human texts and 990 AI texts. No retraining or threshold calibration.")

    fig,axes = plt.subplots(1,2,figsize=(10,5))
    ordered = ["<1,000","1,000–1,999","2,000–3,999",">=4,000"]
    for ax,kind,title in zip(axes,["false_positive","false_negative"],["Human texts flagged as AI","AI texts missed"]):
        for domain,offset in [("abstracts",-.08),("news",.08)]:
            cells = bins.loc[bins.domain.eq(domain)&bins.error_type.eq(kind)&bins.n_class_texts.ge(30)]
            xx = np.array([ordered.index(value) for value in cells.length_bin])+offset
            values = cells.error_rate.to_numpy()
            ax.errorbar(xx,values,yerr=[values-cells.exact_ci_low.to_numpy(),cells.exact_ci_high.to_numpy()-values],
                        fmt="o",capsize=3,color=COLORS[domain],label=DOMAIN_NAMES[domain])
        ax.set_xticks(range(4),ordered,rotation=20)
        ax.set_xlabel("Characters per text")
        ax.set_title(title,loc="left",fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.grid(alpha=.15)
    axes[0].set_ylim(0,.055)
    axes[0].set_ylabel("Class-specific error rate")
    axes[1].set_ylim(0,1.06)
    axes[1].set_yticks(np.arange(0,1.01,.2))
    axes[1].legend(frameon=False,loc="center right")
    fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.84,wspace=.23)
    save(fig,"error_rates_by_length.png","Only cells with at least 30 class-specific texts shown. Error bars: exact binomial 95% intervals; full denominators in the table.")

    metric_names = {"accuracy":"Accuracy","precision":"AI precision","recall":"AI recall","f1":"AI F1",
                    "false_positive_rate":"Human FPR","false_negative_rate":"AI FNR","roc_auc":"ROC-AUC"}
    metric_rows = [{"Metric":metric_names[field],"Abstracts":f"{familiar[field]:.6f}","News":f"{news[field]:.6f}",
                    "Change (news minus abstracts)":f"{changes.loc[field,'news_minus_abstracts']:+.6f}"} for field in METRICS]
    example_rows = [{"Row ID":row.row_id,"Error":row.error_type,"AI score":f"{row.score_ai:.6f}",
                    "Characters":row.length_characters,"AI-assisted description":note_map[row.row_id]}
                   for row in examples.itertuples(index=False)]
    bin_rows=[]
    for row in bins.itertuples(index=False):
        bin_rows.append({"Domain":row.domain,"Error type":row.error_type,"Characters":row.length_bin,
                         "Errors / class texts":f"{row.n_errors}/{row.n_class_texts}",
                         "Rate":f"{row.error_rate*100:.2f}%" if row.n_class_texts else "Undefined",
                         "Sample note":"Small (<30)" if 0<row.n_class_texts<30 else ("No texts" if not row.n_class_texts else "")})
    accuracy_change = changes.loc["accuracy"]
    fpr_news = errors.loc[("news","false_positive")]
    report = f"""# Domain transfer and error analysis

Phase 3 completed October 13, 2026. The frozen Phase 2 detector was applied to the frozen news test without refitting, expanding its vocabulary or changing the 0.5 decision threshold.

## Main finding

**Accuracy fell from {familiar.accuracy*100:.2f}% on familiar abstracts to {news.accuracy*100:.2f}% on unseen news**, a change of {accuracy_change.news_minus_abstracts*100:.2f} percentage points. The independent source-component bootstrap 95% interval for this change is {accuracy_change.ci_low*100:.2f} to {accuracy_change.ci_high*100:.2f} percentage points. Each test has 1,980 texts, 990 human and 990 AI, from 990 matched source pairs.

The detector identified only **{int(news.tp)} of {int(news.n_ai)} AI news texts** and missed **{int(news.fn)}** ({news.false_negative_rate*100:.2f}% FNR). It labeled all {int(news.n_human)} human news texts correctly. It predicted human for {int(news.tn+news.fn):,} of {int(news.n_texts):,} texts, making accuracy close to the 50% always-human reference on this balanced sample. That reference is a mathematical class-balance comparison, not another fitted model.

This is evidence of poor fixed-threshold transfer for this lexical detector in this experiment. It is not evidence that every AI detector fails or that writing domain alone caused the change.

![Domain performance](../outputs/figures/generalization_performance.png)

{table(pd.DataFrame(metric_rows))}

ROC-AUC remains {news.roc_auc:.4f}, while recall at the fixed threshold is {news.recall*100:.2f}%. Thus score ranking and thresholded detection tell different stories. The experiment has not tested whether target-domain calibration would improve performance. Changing the threshold after seeing this test would invalidate the prespecified comparison.

News precision is {news.precision:.2f}, based on only {int(news.tp+news.fp)} predicted-AI texts. It should not be read as reliable detection: almost all AI texts were missed. The raw precision bootstrap interval is uninformative here because resamples without predicted positives use the previously documented zero-precision convention. Full metric and difference intervals are retained in `generalization_metrics.csv` and `generalization_changes.csv`; no significance claim is made.

![News confusion matrix](../outputs/figures/news_confusion_matrix.png)

## Design and uncertainty

The [Phase 3 protocol](../research/phase3_protocol.md) was committed before news scoring. The run record's protocol checksum captures that pre-outcome version at its recorded code-base commit; the current protocol additionally contains the explicitly dated reporting amendment below. The saved model was checked against Phase 2's SHA-256 and remained unchanged afterward. Phase 1 corpus/manifest checksums and Phase 2 output checksums were verified. The loaded model reproduced the saved familiar-test scores within numerical tolerance. Only `test_shifted` news was newly scored; unused news, training and validation are absent from Phase 3 summaries. No model selection, recalibration or new feature fitting occurred.

News intervals use 1,000 intact-component bootstrap draws (seed 44). Difference intervals independently resample each domain's components (seed 45); these are disjoint source samples, not paired predictions. Familiar intervals are the existing Phase 2 intervals. They condition on the fitted detector and source samples, not the universe of domains or retraining outcomes.

Zero news false positives produce a degenerate bootstrap FPR interval of [0,0]. A separately labeled exact binomial interval was added after this outcome was observed, with the reason recorded in the protocol amendment. The two-sided 95% interval for news human FPR is **{fpr_news.exact_ci_low*100:.3f}%–{fpr_news.exact_ci_high*100:.3f}%**, assuming independent sampled human source items. This avoids treating zero observed errors as proof of zero population risk. Exact class-specific intervals are in `domain_error_rates.csv` and the length table. They use [SciPy's Clopper–Pearson implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html) and do not replace the primary component-bootstrap accuracy-change analysis.

## Errors by text length

These bins were fixed in Phase 0. All rates use the relevant class as denominator: human texts for false positives, AI texts for false negatives. Zero-sample bins are undefined; fewer than 30 texts is flagged as a small sample. The chart omits small cells while the table preserves them.

![Length error rates](../outputs/figures/error_rates_by_length.png)

{table(pd.DataFrame(bin_rows))}

The observed news misses occur across the supported length bins. The two detected AI news texts are in the longest bin; this is too little evidence to establish that length improves detection. Abstract errors and news errors cannot isolate a causal length effect: the domains have different content, generation conditions and class-conditional length distributions. No author, native-language or demographic fields support subgroup claims.

## Identified examples

The deterministic rule selected the highest-scoring false-positive human and lowest-scoring false-negative AI text in each domain, where such an error existed. **Three complete texts were inspected with AI assistance**; there was no news false positive to select. Full text stays local in `work/phase3/error_examples.json`; the following IDs map to the pinned corpus. Scientific claims, events and attributed quotations in these passages were not independently verified.

{table(pd.DataFrame(example_rows))}

These are intentionally extreme errors, not a representative sample. Their descriptive content does not establish what features caused classification. Coefficient interpretation belongs to Phase 4. No manual project-owner review or independent validation is claimed.

## Limits, reproduction and next phase

This adapts Xia et al.'s cross-dataset design to one TF-IDF/logistic detector, one Mistral generator, one zero-shot prompt and a newly grouped split. Original neural scores are not matched benchmarks. Author identities, original source URLs and model pretraining records remain unavailable. Lexical near-duplicate checks cannot eliminate paraphrases or all source overlap. Released cleaning and length differences can provide shortcuts, and equal class prevalence is artificial.

After Phases 1 and 2, run `python src/evaluate_generalization.py`, `python src/analyze_errors.py`, then `python src/report_generalization.py`. Retain the saved Phase 2 pipeline: its checksum is verified. The evaluation script never calls fit. Published outputs contain identifiers, scores, counts, summaries and figures, not text. The report verifies saved metric counts and length-bin totals against the predictions. Eleven regression tests cover the original checks plus independent-change sign, component exclusion, length-bin boundaries/empty cells, deterministic error selection and the exact zero-error upper bound. These are internal checks, not independent research replication.

The generalization question now has a measured answer for the approved experiment: strong familiar-domain performance did not survive this fixed-detector domain transfer. Phase 4 will inspect coefficients and distinguish predictive associations from proposed explanations. The overall portfolio project is not yet complete.
"""
    (ROOT / "report/generalization_report.md").write_text(report,encoding="utf-8",newline="\n")
    print("Verified metrics and bin totals; generated three charts and report/generalization_report.md.")


if __name__ == "__main__":
    main()
