"""Generate descriptive tables, figures and the Phase 1 overview from prepared data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_data import ROOT, SHA256, REVISION, file_hash

COLORS = {0: "#2A6381", 1: "#D17A43"}
CLASS_NAMES = {0: "Human", 1: "AI (Mistral)"}
SPLIT_NAMES = {"train": "Abstracts: training", "validation": "Abstracts: validation",
               "test_familiar": "Abstracts: familiar test", "test_shifted": "News: shifted test",
               "unused_news": "News: unused", "excluded_cross_domain": "Excluded: linked domains"}


def markdown_table(frame):
    # Avoid an optional tabulate dependency for these small tables.
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def save_figure(fig, name, footer):
    fig.text(.06, .02, footer, fontsize=8, color="#53616D")
    fig.savefig(ROOT / "outputs/figures" / name, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    tables = ROOT / "outputs/tables"
    prepared = ROOT / "data/processed/selected_texts.csv"
    audit = json.loads((tables / "preparation_audit.json").read_text(encoding="utf-8"))
    if file_hash(prepared) != audit["processed_sha256"]:
        raise ValueError("Prepared data no longer matches the preparation audit.")
    df = pd.read_csv(prepared, dtype={"item_id": str}, keep_default_na=False)
    (ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "report").mkdir(parents=True, exist_ok=True)
    composition = df.groupby(["domain", "model", "label"]).size().reset_index(name="texts")
    splits = df.groupby(["domain", "split"]).agg(texts=("row_id", "size"),
                                                  source_items=("source_group", "nunique"),
                                                  components=("leakage_group", "nunique")).reset_index()
    lengths = df.groupby(["domain", "model"])["length_characters"].agg(["count", "min", "median", "mean", "max"]).reset_index()
    lengths["mean"] = lengths["mean"].round(1)
    length_words = df.groupby(["domain", "model"])["length_words"].agg(["count", "min", "median", "mean", "max"]).reset_index()
    length_words["mean"] = length_words["mean"].round(1)
    quality = [{"check": f"missing_or_sentinel_{key}", "count": val} for key,val in audit["missing_or_sentinel_fields"].items()]
    quality.extend({"check": key, "count": val} for key,val in audit["duplicate_checks"].items())
    quality.extend({"check": key, "count": audit[key]} for key in ["invalid_labels", "unpaired_source_items", "near_duplicate_pairs", "cross_source_near_duplicate_pairs", "source_group_split_overlap", "component_split_overlap"])
    for name, frame in [("dataset_composition.csv",composition),("split_summary.csv",splits),
                        ("text_length_summary.csv",lengths),("word_length_summary.csv",length_words),
                        ("quality_checks.csv",pd.DataFrame(quality))]:
        frame.to_csv(tables / name, index=False, lineterminator="\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(8, 5))
    domains = ["abstracts", "news"]
    x = np.arange(len(domains))
    for label, offset in [(0,-.19),(1,.19)]:
        counts = [int(((df.domain==domain) & (df.label==label)).sum()) for domain in domains]
        bars = ax.bar(x + offset, counts, .34, color=COLORS[label], label=CLASS_NAMES[label])
        ax.bar_label(bars, labels=[f"{n:,}" for n in counts], padding=5)
    ax.set_xticks(x, ["Scientific abstracts", "News articles"])
    ax.set_ylabel("Texts")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_title("Selected data: balanced classes in both domains", loc="left", pad=15)
    ax.legend(frameon=False, loc="upper right")
    ax.yaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=.17, top=.86)
    save_figure(fig, "class_distribution.png", f"Eligible subset: {len(df):,} texts / {df.source_group.nunique():,} paired source items. AI: Mistral zero-shot.")

    fig, axes = plt.subplots(1,2,figsize=(10,4.8),sharey=True)
    bins = np.geomspace(df.length_characters.min(), df.length_characters.max()*1.01, 30)
    for ax,domain in zip(axes,domains):
        for label in [0,1]:
            values = df.loc[df.domain.eq(domain) & df.label.eq(label), "length_characters"]
            ax.hist(values, bins=bins, histtype="step", linewidth=2, color=COLORS[label], label=CLASS_NAMES[label])
            ax.axvline(values.median(), color=COLORS[label], ls="--", lw=1, alpha=.75)
        ax.set_xscale("log")
        ax.set_xlabel("Characters per text (log scale)")
        ax.set_title("Scientific abstracts" if domain=="abstracts" else "News articles",loc="left")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Texts per common logarithmic bin")
    axes[1].legend(frameon=False)
    fig.suptitle("Text lengths differ by class and domain", x=.06,ha="left",fontweight="bold")
    fig.subplots_adjust(bottom=.2,top=.81,wspace=.1)
    save_figure(fig,"text_length_distribution.png","3,000 texts per class in each domain. Dashed lines show class medians; both panels use the same bins.")

    present = [key for key in SPLIT_NAMES if key in set(df.split)]
    fig, ax = plt.subplots(figsize=(9,5))
    left = np.zeros(len(present))
    for label in [0,1]:
        counts = np.array([int((df.split.eq(key) & df.label.eq(label)).sum()) for key in present])
        ax.barh(range(len(present)),counts,left=left,color=COLORS[label],label=CLASS_NAMES[label])
        left += counts
    for y,value in enumerate(left):
        ax.text(value+35,y,f"{int(value):,}",va="center",fontsize=9)
    ax.set_yticks(range(len(present)),[SPLIT_NAMES[key] for key in present])
    ax.invert_yaxis()
    ax.set_xlim(0,max(left)*1.18)
    ax.set_xlabel("Texts")
    ax.set_title("Source pairs stay together across the frozen splits",loc="left",pad=40)
    ax.legend(frameon=False,loc="lower right",bbox_to_anchor=(1,1.01),ncol=2)
    fig.subplots_adjust(left=.29,bottom=.18,top=.84)
    save_figure(fig,"split_composition.png","Only abstracts enter training and validation. Unused news is retained for transparency and excluded from the main experiment.")

    active = df.loc[df.split.isin(["train","validation","test_familiar","test_shifted"])]
    candidate_description = (
        "The lossless size/prefix filter ruled out every pair at this threshold; no exact Jaccard candidate comparison was needed."
        if audit['candidate_comparisons'] == 0 else
        f"Size/prefix filtering selected {audit['candidate_comparisons']:,} candidates, which were verified by exact set intersection and union."
    )
    report = f"""# Dataset overview and split design

Phase 1 completed October 12, 2026. The owner approved the Phase 0 paper, dataset and scope by requesting Phase 1. This report records preparation and descriptive analysis; model performance is documented separately in the [Phase 2 baseline report](baseline_report.md).

## Data and provenance

This methodological reproduction uses the public release accompanying [Xia et al. (EACL 2026)](https://aclanthology.org/2026.eacl-long.307/). The input is [the pinned main CSV](https://huggingface.co/datasets/yuxixia/AI-detector-generalization-benchmark/blob/{REVISION}/data/linguistic_analysis.csv), revision `{REVISION}`. Its dataset card declares Apache-2.0; the separate paper repository declares MIT. The release includes preprocessed human and generated texts, not original source-document metadata or split membership. No text is committed in this project.

The full input checksum was verified: `{SHA256}`. It contains {audit['source_rows']:,} rows. The approved filter retains human texts and `model=mistral, prompt_style=zero_shot_baseline` texts in `abstracts` and `news`: **{len(df):,} texts**, **{df.source_group.nunique():,} paired source items**. AI is positive (`1`), human is `0`. The model name is release metadata mapped by the authors to Mistral-Large-Instruct-2411; it was not independently reconstructed from text.

{markdown_table(composition)}

![Class distribution](../outputs/figures/class_distribution.png)

## Preparation and integrity checks

Only six released columns are read: text, label, generator, prompt style, domain/source and item ID. The unnamed export index and precomputed feature columns are not detector inputs. Required fields were checked for blanks and common missing-value sentinels; labels were checked against generator identity; human prompt metadata was checked; every source item was checked for exactly one human and one AI row.

Text processing only normalizes line endings and strips boundary whitespace; {audit['formatting_modified_rows']:,} rows changed. Raw punctuation, capitalization and internal whitespace remain available to the future detector. No stop-word removal or linguistic cleanup was applied. There were no invalid rows or exact duplicates requiring removal.

{markdown_table(pd.DataFrame(quality))}

Near duplication was evaluated across all {len(df):,} texts, not just within an already chosen split. The audit casefolds words and builds sets of word 5-grams. A pair is linked if Jaccard similarity is at least 0.80. {candidate_description} This is lexical duplicate detection, not a semantic paraphrase detector. Identical sets qualify; repeated shingle frequency is ignored.

The audit found {audit['near_duplicate_pairs']:,} qualifying text pairs, including {audit['cross_source_near_duplicate_pairs']:,} pairs connecting different source items. All texts sharing `(domain, item_id)` are grouped. Qualifying near-duplicate links merge source groups transitively. The resulting {audit['leakage_components']:,} components remain intact during split assignment. Any component spanning both domains would be excluded from the main experiment; {audit['split_strategy']['cross_domain_components_excluded']:,} such components were found. Pair IDs and similarity values, without text, are in `outputs/tables/near_duplicate_pairs.csv`.

## Frozen splits

Seed: **42**. Source groups are sorted before deterministic shuffling. Abstract components are allocated toward 50% training, 17% validation and 33% familiar test by source-item weight; a component is never divided to satisfy an exact percentage. A separate shuffle of news components selects a similarly sized shifted test. All remaining news texts are explicitly marked unused.

{markdown_table(splits)}

There are {len(active):,} texts in the main experiment's active splits. News is absent from training and validation. The stored manifest freezes every selected row's source group, linked component and split. No group or qualifying duplicate component crosses split boundaries. Familiar and shifted test sets contain distinct sources; their performance difference will not be treated as a paired prediction comparison. Bootstrap sampling in later phases should use the linked component if it joins multiple source items.

![Split composition](../outputs/figures/split_composition.png)

These are newly constructed project splits, not the original paper's assignments. Whole-subset duplicate auditing and descriptive EDA occurred before model fitting and are disclosed. The vocabulary, TF-IDF weights, model parameters and any learned preprocessing must subsequently be fitted on abstract training data only. Validation may guide convergence or allowed settings; neither final test may guide selection. No news data may guide vocabulary or thresholds.

## Text lengths and potential shortcuts

Character counts use the retained Unicode strings. Word counts use `\\w+` token matches and are a descriptive convention, not the future detector's tokenizer.

{markdown_table(lengths)}

![Text lengths](../outputs/figures/text_length_distribution.png)

Human text is longer at the median in both domains. News is substantially longer than abstracts. Therefore domain transfer changes topic, writing context, text length and the released generation/cleaning conditions together. An apparent performance difference cannot isolate one cause. Later error analysis will report class-specific denominators in the approved fixed character bins, and coefficient inspection will describe predictive associations without causal claims.

## Remaining limitations

- Word-shingle similarity below 0.80, paraphrases and shared facts/topics can remain undetected. The rule does not prove the absence of all leakage.
- Author identities, original source URLs, publication identifiers and model pretraining records are not present in the selected fields. Author overlap and generator pretraining contamination cannot be eliminated from this release.
- Human and AI passages share source items but may differ in length, task and class-specific cleaning. The detector may learn collection artifacts.
- Labels and generator identities come from the research release. No independent human authorship verification occurred.
- This scope covers one generator, one prompt and one direction of domain transfer. It does not assess current commercial AI detectors or all AI writing.

## Reproduction and checkpoint

From the repository root, install `requirements.txt`, run `python src/prepare_data.py`, then `python src/eda.py`. The first command downloads the pinned source if absent and verifies its checksum. An existing source file can be passed with `--input`. Run `python -m unittest discover -s tests -v` for the candidate-filter and group-splitting regression checks. See `INSTRUCTIONS.md` for setup and validation commands.

Executed validation includes input checksum verification, label/pair checks, exact and near-duplicate audits and group-disjoint split assertions. Regression tests compare candidate filtering against exhaustive pair comparison on synthetic sets, check transitive source linking and check cross-domain exclusion. These are code checks, not an independent research replication. Generated figures were visually inspected before committing the Phase 1 checkpoint; the actual Git status and commit hash are reported in chat.

AI assistance was used for implementation, debugging, research interpretation and documentation. These tables and figures are generated from the documented input by the retained scripts. No claim is made that the project owner manually reviewed all texts or independently validated the analyses.

At the Phase 1 checkpoint, the next step was training the approved TF-IDF + logistic regression baseline. Its completed familiar-test evaluation is documented in the linked Phase 2 report. This report retains the preparation checks and descriptive analysis.
"""
    (ROOT / "report/data_overview.md").write_text(report,encoding="utf-8")
    print(f"Generated five summary tables, three figures and report/data_overview.md from {len(df):,} texts.")


if __name__ == "__main__":
    main()
