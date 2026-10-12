# Reproduce Phase 1

Use Python 3.12. The current execution was verified on Python 3.12.14 on Windows, using the versions in `requirements.txt`. Run commands from the repository root.

## Windows PowerShell

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe src\prepare_data.py
& .\.venv\Scripts\python.exe src\eda.py
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

The original paper's neural detector, original split membership and complete benchmark are not reproduced. No baseline or generalization performance has been computed during Phase 1.
