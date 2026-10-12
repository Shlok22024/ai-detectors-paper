# Paper and dataset feasibility: AI detector generalization

Verified: October 12, 2026. Status: Phase 0 complete; owner approval pending. No detector has been trained or evaluated.

## Recommendation

**GO for a methodological reproduction of one cross-dataset experiment from Xia, Stańczak, and Roth (EACL 2026), using their released text and a TF-IDF + logistic regression detector. NO-GO for reproducing the full neural-model study on a laptop.**

Use `yuxixia/AI-detector-generalization-benchmark`, revision `a8bcd10fbe1f522fe47fd56f4fc75a54f36a276c`, specifically `data/linguistic_analysis.csv`. Fix the generator to `mistral` and AI prompting to `zero_shot_baseline`. Train on scientific abstracts; compare held-out abstracts with held-out news. Extend the evaluation with human false-positive rates by text length and inspection of linear coefficients.

The eligible subset was actually inspected: **12,000 texts, 6,000 human and 6,000 AI**, across **6,000 source items**. Each domain has 3,000 source items with one human text and one Mistral text. These are eligibility counts before Phase 1 near-duplicate checks, not final experimental sample sizes.

This preserves the paper's controlled cross-domain design but changes its detector and split assignment. TF-IDF + logistic regression is **our baseline**, not a baseline reported by that paper. The portfolio should say “methodological reproduction and extension,” not “exact replication” or “reproduced the paper's accuracy.”

## 1. Candidate comparison

| Candidate | Public release verified | Original computation | Practical decision |
|---|---|---|---|
| Xia et al., 2026: linguistic explanation of generalization | Official full paper; code; downloadable text with labels, generator, domain and item IDs | XLM-RoBERTa-base and DeBERTa-V3-small; HGX H100; about 400 GPU hours reported | **Recommended with a narrower detector and one shift** |
| Yu et al., 2025: EvoBench | Official full paper; three complete paired JSON files downloaded; repository mainly data | Authors report four A100 GPUs; several methods score text with billion-parameter models | Defer: small repeated reference pool, provenance reconciliation and incomplete license release |
| Dugan et al., 2024: RAID | Official full paper; code; labeled non-adversarial training CSV endpoint checked; prefix downloaded | Full generation used 32 A6000 GPUs and 14,872 GPU hours; released text avoids regeneration | Credible fallback; larger scope and weaker match to an original trained-detector experiment |

### Candidate A: recommended paper

**Title:** Explaining Generalization of AI-Generated Text Detectors Through Linguistic Analysis. **Authors:** Yuxi Xia, Kinga Stańczak, Benjamin Roth. **Venue/year:** EACL 2026, Volume 1: Long Papers, pp. 6524–6546. [Official publication](https://aclanthology.org/2026.eacl-long.307/), [full paper](https://aclanthology.org/2026.eacl-long.307.pdf), [code](https://github.com/Yuuxii/Generalization-of-AI-text-Detector), [dataset](https://huggingface.co/datasets/yuxixia/AI-detector-generalization-benchmark).

The paper studies transfer across prompts, generators and domains, then correlates transfer accuracy with shifts in 80 linguistic features. It reports 12,000 human texts and 504,000 generated texts, using seven generators, six prompt strategies and four sources: arXiv abstracts, CNN/Daily Mail news, AmazonReviews2023 reviews and ASQA answers. Its source-item split ratio is 50:17:33. The detectors are XLM-RoBERTa-base and DeBERTa-V3-small; the reported setup uses three epochs, learning rate 2e-5, batch size 16, maximum sequence length 512 and weight decay 0.01. Its principal detector metric is accuracy; its linguistic analysis uses Pearson correlations.

**What we can reproduce:** the experimental logic of same-generator, same-prompt cross-dataset evaluation on released text. **What we cannot reproduce within scope:** neural detector scores, original split membership, the full condition grid, generation or the complete linguistic correlation study. **Extension:** length-stratified false positives and coefficient-based examination of an inexpensive detector.

The code repository has an [MIT license](https://github.com/Yuuxii/Generalization-of-AI-text-Detector/blob/316184f7d87659f8d348685dc6967c3b0e0cb235/LICENSE). The dataset's pinned card declares [Apache-2.0](https://huggingface.co/datasets/yuxixia/AI-detector-generalization-benchmark/blob/a8bcd10fbe1f522fe47fd56f4fc75a54f36a276c/README.md). The card contains almost no documentation beyond this declaration, so the paper and inspected fields supply the provenance description. These release declarations support proceeding with local research; they do not establish that every upstream text has identical redistribution terms. Publish code, aggregate outputs and acquisition instructions; keep downloaded corpora out of Git and review upstream terms before publishing text excerpts.

### Candidate B: EvoBench

**Title:** EvoBench: Towards Real-world LLM-Generated Text Detection Benchmarking for Evolving Large Language Models. **Authors:** Xiao Yu, Yi Yu, Dongrui Liu, Kejiang Chen, Weiming Zhang, Nenghai Yu, Jing Shao. **Venue/year:** Findings of ACL 2025, pp. 14605–14620. [Official publication](https://aclanthology.org/2025.findings-acl.754/), [full paper](https://aclanthology.org/2025.findings-acl.754.pdf), [release repository](https://github.com/happy-Moer/EvoBench).

The paper reports 29 evolving versions across seven families and five datasets, with 150 human/AI pairs per version–dataset condition: 21,750 AI texts and 21,750 pair occurrences. This is not evidence of 21,750 distinct human documents. Tasks include XSum and WritingPrompts continuation, PubMed QA, PeerRead reviews and SocialMedia paraphrasing. Existing supervised detectors include OpenAI's GPT-2 detectors, RADAR, Text Fluoroscopy and ImBD; statistical methods include likelihood, rank measures, entropy, DetectGPT and Fast-DetectGPT variants. Metrics are AUROC and evolving model generalization (EMG). The original evaluation uses pretrained detectors, not a random train/test split of all benchmark rows. Appendix A's optimization uses 200 HC3 entries and an 8:1 train/validation ratio with regenerated examples.

**Actual release checks:** at repository revision `4c866cd211499744d21d34cbf2f32594064047c0`, downloaded two Claude XSum files and one Claude PubMed file. Each has `original` and `sampled` lists of 150 strings. Generator/version and dataset are recorded in paths, rather than row fields; there are no explicit source IDs or split fields in these sampled JSONs. Two XSum files share only three exact human strings, and their sequences are unequal. Thus neither positional alignment across versions nor a single identical reference pool can be assumed. Within the first XSum file, human strings are unique and both lists have no empty entries. Median character counts are 960 human and 980.5 generated; this is one file, not a benchmark-wide summary. Continuation examples retain overlapping opening text, another grouping risk.

The README says 30 versions, whereas the published paper says 29. The repository tree contains no Python implementation files and no LICENSE file, although the README declares MIT and links to a missing LICENSE. Data can be downloaded, but the specific release license and benchmark-to-file mapping are insufficiently documented for our preferred target. A narrowed model-version adaptation is possible after these issues are resolved; exact detector reproduction and the optimization experiments are outside this laptop scope. Domain transfer also changes generation task for some domains, complicating interpretation.

### Candidate C: RAID

**Title:** RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. **Authors:** Liam Dugan, Alyssa Hwang, Filip Trhlík, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch. **Venue/year:** ACL 2024, Volume 1: Long Papers, pp. 12463–12492. [Official publication](https://aclanthology.org/2024.acl-long.674/), [full paper](https://aclanthology.org/2024.acl-long.674.pdf), [code](https://github.com/liamdugan/raid), [dataset card](https://huggingface.co/datasets/liamdugan/raid).

The original core dataset has 14,971 human documents, 509,014 non-adversarial generations and 6,287,820 texts including attacks, covering 11 generators, eight domains, four decoding conditions and 11 attacks. Ten percent is reserved for a hidden-label test release. The paper evaluates eight open and four commercial detectors rather than training a TF-IDF detector: RoBERTa variants, RADAR, GLTR, Binoculars, Fast-DetectGPT, LLMDet, GPTZero, Originality, Winston and ZeroGPT. Its central metric is the fraction of AI generations detected at a target human FPR of 5%, with thresholds tuned per domain. This is AI recall at constrained FPR, not ordinary overall accuracy.

The live release advertises an expanded dataset; do not confuse it with the 2024 snapshot. Its card declares MIT and the code has an MIT license. The official loader offers `train_none.csv` without adversarial examples, avoiding the 11.8 GB attacked training release. The endpoint reports **801,662,741 bytes**. A 4,000,000-byte prefix was downloaded and parsed, confirming `generation`, `model`, `domain`, `source_id`, `id`, `adv_source_id`, `decoding`, `repetition_penalty`, `attack`, `title` and `prompt`. Human rows use `model=human`; no separate binary label is needed. The prefix covers abstracts and includes human, llama-chat and mpt rows. It does not verify every domain or the entire file's integrity.

**Feasible target:** a grouped, non-adversarial cross-domain or cross-generator adaptation with generation settings fixed. **Not feasible within this scope:** reproducing generation, all original detectors or commercial services. **Possible extension:** compare a threshold fixed on familiar validation data against domain-specific calibration, clearly marking target-domain calibration as an adapted setting. Source groups and source-related generations must stay together; labels in the official hidden test cannot be assumed accessible. RAID remains a fallback rather than an additional benchmark to implement now.

## 2. Recommended dataset: actual download inspection

Two complete files were downloaded at the pinned Hugging Face revision. SHA-256 and detailed count tables are retained in `feasibility_evidence.json`. No access token or paid API was required. The third file was sampled only.

| Released file | Bytes | Inspection | Actual rows |
|---|---:|---|---:|
| `linguistic_analysis.csv` | 743,557,706 | Complete download and row inspection | 431,431 |
| `linguistic_analysis_multistage.csv` | 137,012,028 | Complete download and row inspection | 95,991 |
| `stylometric_data_full.csv` | 1,168,220,706 | Header and 500,000-byte prefix only | Not counted |

The main file contains **12,000 human and 419,431 AI rows**, with five AI prompting strategies. The multistage file contains **12,000 human and 83,991 self-refine AI rows**. All 12,000 multistage human rows match the main file's human text by `(source, item_id)`. Therefore simply concatenating these files duplicates the human class. Combining the AI rows and counting those human rows once gives **515,422 released text rows**, 578 fewer than the paper's stated 516,000. This is a release-count discrepancy; its cause has not been established. It does not reduce the proposed zero-shot Mistral cells, which are complete.

The main file has `text`, `label`, `prompt_style`, `model`, `item_id`, `source`, precomputed linguistic columns and an unnamed exported index. Labels are `0` and `1`; sampled human rows have `model=human, label=0`, and generated rows have `label=1`. No blank values were observed in the six required fields in either complete file. Phase 1 should also check semantic label consistency and string sentinels such as `nan`, rather than treating this blank check as complete validation.

**Critical code/release mismatch:** the [published training script](https://github.com/Yuuxii/Generalization-of-AI-text-Detector/blob/316184f7d87659f8d348685dc6967c3b0e0cb235/src/train_classifiers.py#L20) filters on `df.split`. None of the three inspected CSV headers has `split`. Original split membership was not located in the inspected release. We must construct and document new grouped splits, rather than reconstructing membership from item-number ranges or suggesting that the released training command runs unchanged.

### Eligibility and text length

| Domain | Human | Mistral zero-shot | Source items | Median human characters | Median AI characters |
|---|---:|---:|---:|---:|---:|
| Abstracts | 3,000 | 3,000 | 3,000 | 1,317 | 1,044.5 |
| News | 3,000 | 3,000 | 3,000 | 3,570.5 | 2,929 |

All selected source groups contain exactly one human and one AI example. There are **zero excess exact-text duplicates**, **zero excess whitespace-normalized duplicates** within each selected domain, and **zero exact-text overlaps across these domains**. Near duplicates, author reuse, related source articles and language-model pretraining contamination have not been ruled out.

These are reasonably substantial texts for lexical classification, but neither class nor domain has identical length distributions. Abstract AI lengths range from 571 to 2,542 characters; news AI lengths range from 645 to 16,834. News human text can fall below the paper's stated pre-cleaning minimum. We will use actual released text lengths and will not assume the paper's sampling thresholds describe every cleaned row. Length differences are a measured design risk, not an established explanation of detector behavior.

The matched item IDs support grouping and indicate comparable source topics. They do not make human writing and generated writing interchangeable: their tasks and the authors' class-specific cleaning can still create distinguishable artifacts. Original source URLs, authors and the original split are not supplied in the inspected main header.

## 3. Scope proposed for approval

### Main experiment

1. Use only human and `mistral` zero-shot rows from `abstracts` and `news`. Feed **only `text`** to the detector; metadata and released feature columns are for grouping and analysis.
2. In Phase 1, resolve exact and near-duplicate connections before assigning splits. Keep each `(source, item_id)` pair in one split and merge linked groups when duplicates require it. Use a simple shingle-based candidate check with a documented similarity rule; do not build an elaborate deduplication system.
3. With seed 42, assign abstract groups to train/validation/test in the paper's 50:17:33 proportions. If all current pairs survive, this means 1,500/510/990 groups, or 3,000/1,020/1,980 texts. These are **planned counts conditional on cleaning**.
4. Select a similarly sized news test set of 990 groups using a fixed seed before fitting anything. The remaining news groups are unused in the main experiment. Use no news text to fit vocabulary, choose parameters or select thresholds. Phase 0 feasibility inspection is disclosed; it has not measured detector outcomes.
5. Fit a scikit-learn pipeline with word unigram/bigram TF-IDF (initial cap 30,000 features, `min_df=2`) and logistic regression (`C=1`, fixed seed). Preserve raw punctuation and capitalization in stored data; document the tokenizer and lowercase setting. This is a conventional lexical baseline and does not model all punctuation signals.
6. Fix the decision threshold at 0.5 for the main analysis and evaluate the same fitted detector on held-out abstracts and news. If parameters need adjustment for convergence or memory, use abstract training/validation data only and document the adjustment before final evaluation.

Report AI as positive (`1`): accuracy, precision, recall, F1, confusion matrix, human FPR and AI FNR; add ROC-AUC using continuous scores. Report denominators and class balance. Accuracy difference between domains is the primary shift summary. The evaluation sets contain distinct sources, so this is not a paired prediction comparison.

Use source-group bootstrap intervals for the primary difference and human FPR, resampling whole human/AI pairs within each test domain. Around 1,000 bootstrap draws is a proposed computational choice, not a performed analysis. Intervals measure uncertainty within these sampled sources, not variability across all domains or model training runs.

### Controlled extension and outputs

Analyze error rates within prespecified character-length bins (`<1,000`, `1,000–1,999`, `2,000–3,999`, `>=4,000`), always showing class-specific sample counts and omitting unsupported comparisons. Inspect top positive and negative logistic coefficients and a small identified error sample. These analyses can reveal possible shortcuts; they do not establish causal explanations or intrinsic properties of AI writing.

Produce the familiar-versus-shifted metric table, confusion matrices, false-positive rates by length and a coefficient chart. Keep the report and README explicit that this tests one lexical detector, one generator and one direction of domain transfer. A good result is as publishable as a poor result. Do not add unseen-generator evaluation, SHAP, neural training, attacks or a second dataset to the approved default scope.

### Practical requirements and effort

Expected tools: Python, pandas, NumPy, scikit-learn, matplotlib and scipy; standard-library download and hashing are sufficient. Read the 744 MB source CSV in chunks with selected columns; retain only the 12,000 eligible rows. Avoid loading all precomputed feature files or joining the stylometric release. A CPU laptop with 8–16 GB RAM should be adequate for this capped sparse model; that is an engineering estimate, not a measured training benchmark. The current machine was inspected: approximately 31.7 GiB RAM and a six-core Intel i5-11400H. No paid API or GPU is required for the proposal.

Allow roughly 2 GB temporary disk space for the recommended source file, paper copies and working outputs; current feasibility downloads for alternatives add storage. Estimated remaining hands-on effort: 18–28 hours across preparation (4–6), baseline/evaluation (5–8), error analysis/charts (4–6), documentation and reproduction checks (5–8). Download speed, near-duplicate findings and owner review may change this estimate.

## 4. Limitations, decision and checkpoint

Known limitations: new split assignment; a different detector; released preprocessing rather than independently recreated preprocessing; source/domain and length changes together; limited upstream authorship metadata; unverified near duplicates and pretraining overlap; sparse dataset documentation; no independent human validation; no measured performance yet. Original neural scores cannot be used as expected scores for our logistic baseline.

**Approval requested:** select Xia et al. (EACL 2026), the pinned released main CSV, Mistral zero-shot abstracts-to-news methodological reproduction, and the error/length/coefficient extension above. Approval covers Phase 1 preparation and subsequent phased implementation within this scope. Phase 1 must pause or revise scope if pairing, label or leakage checks materially invalidate it. Full neural replication is a no-go; RAID is an alternative if this release becomes unusable.

Phase 0 validation performed: accessed official full papers; extracted relevant methods, limitations and appendices; visually inspected methodology/statistics pages; inspected public repository trees and training source; checked license declarations; downloaded two complete recommended CSVs and counted their rows/conditions; inspected pairing, blank required fields, exact duplicates and selected length distributions; downloaded three complete EvoBench JSON files; inspected a RAID CSV prefix and endpoint size. No model performance, hypothesis tests, independent replication or project-owner review occurred.

Files at this checkpoint: this report, `research/feasibility_evidence.json`, planning README and `.gitignore`. Research scratch files and downloaded data remain outside the tracked repository. A local Git checkpoint records the report; the actual hash and working-tree state are reported in the chat after committing. No GitHub remote was supplied, so this checkpoint cannot be pushed yet. Next phase: dependency setup, dataset preparation, EDA and final split construction **after explicit owner approval**.

AI assistance was used for research retrieval, feasibility inspection and drafting this report. Dataset counts came from executed inspection code and downloaded files. The project owner has not yet reviewed or approved this report, and no independent validation is claimed.
