# Reproduce Phases 1–3

Use Python 3.12. The current execution was verified on Python 3.12.14 on Windows, using the versions in `requirements.txt`. Run commands from the repository root.

## Windows PowerShell

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe src\prepare_data.py
& .\.venv\Scripts\python.exe src\eda.py
& .\.venv\Scripts\python.exe src\train_baseline.py
& .\.venv\Scripts\python.exe src\report_baseline.py
& .\.venv\Scripts\python.exe src\evaluate_generalization.py
& .\.venv\Scripts\python.exe src\analyze_errors.py
& .\.venv\Scripts\python.exe src\report_generalization.py
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe -m pip check
```

The checked-out working project is in `E:\Personal Projects\AI Detectors`. The two preparation commands use paths relative to their source files, so outputs are always written inside that repository even if the command is launched elsewhere.

## macOS/Linux

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/prepare_data.py
.venv/bin/python src/eda.py
.venv/bin/python src/train_baseline.py
.venv/bin/python src/report_baseline.py
.venv/bin/python src/evaluate_generalization.py
.venv/bin/python src/analyze_errors.py
.venv/bin/python src/report_generalization.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

These are portable command equivalents; they have not been executed on macOS/Linux.

## Inputs and outputs

The first preparation run downloads approximately 744 MB and requires no credentials. See `data/README.md` for the pinned URL, checksum and reuse of a local input. The processed corpus remains local and is not committed. Outputs are regenerated deterministically from the pinned input with seed 42. Preparation verifies the checksum before accepting the file and fails if label, pair or split integrity checks fail.

Generated artifacts:

- `data/processed/selected_texts.csv`: the 12,000 selected texts and their frozen split assignments; ignored by Git.
- `outputs/tables/split_manifest.csv`: row identifiers, provenance, lengths, groups and splits, without text.
- `outputs/tables/preparation_audit.json`: integrity checks, duplicate counts and source/output checksums.
- `outputs/tables/near_duplicate_pairs.csv`: qualifying pair identifiers and Jaccard similarities; header-only if no pairs qualify.
- Composition, split, character/word-length and quality-check summary tables in `outputs/tables/`.
- Three descriptive charts in `outputs/figures/` and the generated `report/data_overview.md`.

The split manifest is the authority for subsequent phases: abstracts supply training, validation and familiar testing; news supplies shifted testing and an explicitly unused remainder. Any connected component spanning domains would be excluded. Do not resplit after seeing model scores. Read only `text` as a detector feature. Fit the vectorizer and all learned preprocessing on the abstract training split; metadata and precomputed release features are not detector inputs.

The three regression tests address candidate-filter correctness versus exhaustive synthetic comparisons, transitive source grouping and cross-domain split exclusion. They do not validate dataset authorship or independently replicate the paper. Phase 1 validation also reran preparation in the requested E: clone and confirmed identical processed-file and manifest hashes, checked dependency consistency and visually inspected all three final charts.

## Phase 2 outputs and validation

Baseline training accepts the frozen Phase 1 corpus and manifest only if their checksums agree with the preparation record. It fits on abstract training text, evaluates abstract validation and then the familiar final test, and produces no news predictions. Its configuration and threshold were fixed in `research/phase2_protocol.md` before fitting. The convergence check fails on optimizer warnings; there is no hyperparameter search or validation refit.

- `models/baseline.joblib`: local saved TF-IDF/logistic pipeline, ignored by Git. Regenerate it with the training command; keep it for Phase 3.
- `outputs/tables/baseline_predictions.csv`: text-free validation and familiar-test identifiers, released labels, probabilities and decisions.
- `outputs/tables/baseline_metrics.csv`: class counts, confusion counts, metrics and source-component bootstrap intervals.
- `outputs/tables/baseline_run.json`: input/output checksums, code commit, parameters, runtime and package versions.
- `outputs/tables/paper_baseline_comparison.csv`: original neural values and the explicit limits of comparison.
- `outputs/figures/baseline_confusion_matrix.png` and `report/baseline_report.md`: generated familiar-test result presentation.

The report command verifies prediction and metric hashes and separately recalculates confusion counts and metrics from the saved predictions. Reloading the saved model was checked against the original validation probabilities. Four additional tests cover hand-calculated error rates, threshold ties and no-positive predictions, intact-pair bootstrap sampling and unchanged vocabulary when transforming held-out-only words. Together with Phase 1, seven regression tests pass.

Repeating training with the same locked input and parameters is a reproducibility check, not another tuning run. Record timestamps and durations may change; compare prediction and metric hashes to assess deterministic results. Numeric operations use one thread, and exact cross-platform equality has not been tested.

The original paper's neural detector, original split membership and complete benchmark are not reproduced. Phase 2 reports the familiar-domain baseline; Phase 3 measures news transfer.

## Phase 3 outputs and validation

The domain evaluation command loads the Phase 2 model only after checking its checksum, verifies the frozen data/manifest and Phase 2 outputs, reproduces familiar scores within numerical tolerance, and scores only the frozen news test. It never calls fit or changes the threshold. The saved model checksum is checked again afterward. It excludes unused news and model-development rows from Phase 3 summaries.

- `generalization_predictions.csv`, `generalization_metrics.csv`, `generalization_changes.csv` and `generalization_run.json` under `outputs/tables/`: both final tests, metric intervals, independently bootstrapped news-minus-abstracts changes and integrity evidence.
- `error_rates_by_length.csv` and `domain_error_rates.csv`: class-specific denominators, errors and separately labeled exact binomial intervals. Zero-sample cells remain undefined; cells with fewer than 30 class texts are flagged.
- `test_errors.csv` and `error_example_selection.csv`: all final-test errors and the deterministic extreme-example selection, without text.
- `work/phase3/error_examples.json`: selected full texts for local inspection, ignored by Git. `research/error_example_notes.json` contains the three actual AI-assisted descriptions, not independent authorship validation. Reporting verifies that their IDs match the selected examples; if a different release/scope changes those IDs, inspect its new examples before writing descriptions.
- Three new figures under `outputs/figures/` and `report/generalization_report.md` present the measured comparison and errors.

News metric intervals use 1,000 intact-component bootstrap draws (seed 44). Difference intervals independently resample disjoint components in both domains (seed 45); this is not paired-domain testing. An exact Clopper–Pearson error-rate interval was added after zero news false positives exposed a degenerate bootstrap interval. The reason and timing are recorded in the protocol; no prediction or threshold changed. Precision bootstrap draws with no predicted positives retain the prior zero convention, and sparse news precision is explicitly treated as uninformative.

The four Phase 3 tests check independent-change direction and rejection of reused components, length-bin boundaries and empty cells, deterministic extreme-example selection, and the nonzero exact upper limit for zero observed errors. Eleven regression tests pass in total. Reporting checks metric/confusion counts and length-bin totals against saved predictions; all three Phase 3 charts were visually inspected.
