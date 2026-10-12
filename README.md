# Can AI Detectors Actually Detect AI?

Research reproduction and generalization study. **Status: Phase 2 complete — the familiar-domain baseline is trained and evaluated.** Cross-domain generalization remains unmeasured.

The question is whether a detector's performance survives a change in writing domain. We trained TF-IDF + logistic regression on scientific abstracts. Next, we will compare familiar abstracts with unseen news, keeping the fitted detector, generator, prompt and threshold fixed.

This is a **methodological reproduction and extension** of the cross-dataset design in [Xia et al. (EACL 2026)](https://aclanthology.org/2026.eacl-long.307/), using their released Mistral zero-shot texts. The detector and split assignments differ from the paper; its neural performance scores are not direct comparison targets.

The familiar test contains 1,980 abstracts, equally split between human and AI. The fixed baseline achieved **97.78% accuracy** (source-pair bootstrap 95% interval: **97.07%–98.43%**) and **0.9778 AI F1**. It flagged **25 of 990 human texts** as AI (**2.53% FPR**) and missed **19 of 990 AI texts** (**1.92% FNR**). These results describe this familiar-domain dataset; they do not establish generalization or real-world authorship reliability.

![Familiar-test confusion matrix](outputs/figures/baseline_confusion_matrix.png)

The [baseline report](report/baseline_report.md) documents metrics, uncertainty, the prespecified configuration and the original-paper comparison. The threshold was fixed at 0.5 before training; there was no hyperparameter search or validation refit. No news predictions have been produced.

Phase 1 retained 12,000 texts from 6,000 human/AI source pairs, with balanced classes in each domain. Labels and pairing passed consistency checks. No exact duplicates or qualifying near duplicates were found under the specified 0.80 word-5-gram Jaccard rule. This does not rule out paraphrases, shared topics or model pretraining overlap.

| Split | Domain | Source pairs | Texts |
|---|---|---:|---:|
| Training | Abstracts | 1,500 | 3,000 |
| Validation | Abstracts | 510 | 1,020 |
| Familiar test | Abstracts | 990 | 1,980 |
| Shifted test | News | 990 | 1,980 |
| Unused | News | 2,010 | 4,020 |

![Text-length distributions](outputs/figures/text_length_distribution.png)

Human passages are longer at the median in both domains, and news is substantially longer than abstracts. Later analysis must account for these potential shortcuts. Domain transfer will not isolate a causal effect of topic or style.

Read the [data overview](report/data_overview.md) for preparation checks and limitations, [feasibility report](research/paper_feasibility.md) for the original selection, and [Phase 0 evidence](research/feasibility_evidence.json) for the download inspection. [Setup and reproduction instructions](INSTRUCTIONS.md) explain how to regenerate Phase 1 and 2 outputs. Start with Python 3.12 and `requirements.txt`; run preparation, EDA, baseline training and baseline reporting in that order.

The working clone is in `E:\Personal Projects\AI Detectors`. Code is covered by this repository's MIT license; the upstream data card declares Apache-2.0. Raw and processed text remain local and are ignored by Git. Source URLs, license distinctions and acquisition instructions are in [data/README.md](data/README.md).

Repository contents: `src/` contains preparation, EDA, baseline training and reporting; `tests/` contains leakage and metric regression checks; `outputs/` contains text-free tables, predictions and figures; `report/` contains generated reports; `research/` contains the Phase 0 decision record and Phase 2 protocol. The prepared text-bearing CSV is local under `data/processed/`, and the saved pipeline is local under `models/`.

AI assistance was used for research retrieval, coding, debugging, source-table verification and documentation. Tables and charts come from executed code and the pinned public dataset. No independent research validation or manual review of every passage is claimed. Next phase: apply the frozen baseline to unseen news and analyze actual errors.
