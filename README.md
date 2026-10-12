# Can AI Detectors Actually Detect AI?

Research reproduction and generalization study. **Status: Phase 1 complete — data preparation, leakage checks and exploratory analysis.** The owner approved the paper, dataset and scope. No model results exist yet.

The question is whether a detector's performance survives a change in writing domain. We will train TF-IDF + logistic regression on scientific abstracts and compare held-out abstracts with unseen news, keeping the generator and prompt fixed.

This is a **methodological reproduction and extension** of the cross-dataset design in [Xia et al. (EACL 2026)](https://aclanthology.org/2026.eacl-long.307/), using their released Mistral zero-shot texts. The detector and split assignments differ from the paper; its neural performance scores are not direct comparison targets.

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

Read the [data overview](report/data_overview.md) for actual checks and limitations, [feasibility report](research/paper_feasibility.md) for the original selection, and [Phase 0 evidence](research/feasibility_evidence.json) for the download inspection. [Setup and reproduction instructions](INSTRUCTIONS.md) explain how to regenerate Phase 1 outputs. Start with Python 3.12 and `requirements.txt`; run `src/prepare_data.py`, then `src/eda.py`.

The working clone is in `E:\Personal Projects\AI Detectors`. Code is covered by this repository's MIT license; the upstream data card declares Apache-2.0. Raw and processed text remain local and are ignored by Git. Source URLs, license distinctions and acquisition instructions are in [data/README.md](data/README.md).

Repository contents: `src/` contains preparation and EDA; `tests/` contains leakage-logic regression checks; `outputs/` contains text-free tables and figures; `report/` contains the generated overview; `research/` contains the Phase 0 decision record. The prepared text-bearing CSV is local under `data/processed/`.

AI assistance was used for research retrieval, coding, debugging and documentation. Tables and charts come from executed code and the pinned public dataset. No independent research validation or manual review of every passage is claimed. Next phase: the approved baseline model.
