# Data acquisition and usage

The approved input is [`linguistic_analysis.csv`](https://huggingface.co/datasets/yuxixia/AI-detector-generalization-benchmark/blob/a8bcd10fbe1f522fe47fd56f4fc75a54f36a276c/data/linguistic_analysis.csv) from Xia et al.'s AI detector generalization benchmark.

- Dataset revision: `a8bcd10fbe1f522fe47fd56f4fc75a54f36a276c`
- File size: 743,557,706 bytes
- SHA-256: `3983ed4e82550f8ab3bdfbbb37a60b89389ae250e36830ef6919e5b297550afc`
- Released card declaration: Apache-2.0. The project's MIT code license does not relicense upstream text.

Run `python src/prepare_data.py` from the repository root. This downloads the pinned file into `data/raw/` if absent, verifies its checksum and selects human/Mistral zero-shot texts from abstracts and news. It writes the local text-bearing subset to `data/processed/selected_texts.csv` and text-free split/quality metadata to `outputs/tables/`.

For an existing complete source, use `python src/prepare_data.py --input "path/to/linguistic_analysis.csv"`. The checksum is verified regardless of acquisition method. The selected subset is balanced: 3,000 human/AI pairs per domain. It uses newly assigned grouped splits; original split membership is unavailable in the inspected release.

Raw and processed corpora are ignored by Git. Publish acquisition instructions and aggregate analysis; review original source terms before redistributing text examples. Related human/AI source items and any qualifying near duplicates share a leakage group. The word-shingle check does not detect every paraphrase or topical overlap.
