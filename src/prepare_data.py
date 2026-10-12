"""Acquire the pinned release, audit the approved subset, and freeze grouped splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a8bcd10fbe1f522fe47fd56f4fc75a54f36a276c"
URL = f"https://huggingface.co/datasets/yuxixia/AI-detector-generalization-benchmark/resolve/{REVISION}/data/linguistic_analysis.csv"
SHA256 = "3983ed4e82550f8ab3bdfbbb37a60b89389ae250e36830ef6919e5b297550afc"
FIELDS = ["text", "label", "model", "prompt_style", "source", "item_id"]
SEED = 42
SIMILARITY = Fraction(4, 5)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def obtain_input(path: Path, allow_download: bool) -> None:
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        print("Downloading pinned source CSV (approximately 744 MB)...", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if file_hash(temporary) != SHA256:
            raise ValueError("Downloaded source checksum differs from the approved release.")
        temporary.replace(path)
    if file_hash(path) != SHA256:
        raise ValueError("Input checksum differs from the approved release.")


def shingles(text: str) -> frozenset[str]:
    # Audit representation only: the detector's retained text is not lowercased here.
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 5:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i:i + 5]) for i in range(len(words) - 4))


def similar_pairs(documents: list[frozenset[str]], threshold: Fraction = SIMILARITY):
    """Exact Jaccard verification after lossless size/prefix candidate filtering.

    All sets use the same global shingle order. A set of size n contributes
    n - ceil(threshold*n) + 1 prefix entries. Similar sets must share a prefix
    entry. Integer arithmetic makes threshold and prefix boundaries exact.
    """
    frequencies = Counter(s for doc in documents for s in doc)
    postings: dict[str, list[int]] = defaultdict(list)
    edges = []
    comparisons = 0
    numerator, denominator = threshold.numerator, threshold.denominator
    for processed, i in enumerate(sorted(range(len(documents)), key=lambda i: (len(documents[i]), i)), start=1):
        doc = documents[i]
        if not doc:
            continue
        prefix_size = len(doc) - (numerator * len(doc) + denominator - 1) // denominator + 1
        prefix = sorted(doc, key=lambda s: (frequencies[s], s))[:prefix_size]
        candidates = {j for s in prefix for j in postings[s]
                      if denominator * len(documents[j]) >= numerator * len(doc)}
        for j in sorted(candidates):
            comparisons += 1
            intersection = len(doc & documents[j])
            union = len(doc) + len(documents[j]) - intersection
            if denominator * intersection >= numerator * union:
                edges.append((j, i, intersection / union))
        for s in prefix:
            postings[s].append(i)
        if processed % 2000 == 0:
            print(f"Near-duplicate audit: {processed:,} texts processed; {comparisons:,} candidates verified", flush=True)
    return edges, comparisons


def component_groups(source_groups: list[str], edges) -> list[str]:
    parents = {group: group for group in source_groups}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for a, b, _ in edges:
        left, right = find(source_groups[a]), find(source_groups[b])
        if left != right:
            parents[max(left, right)] = min(left, right)
    return [find(group) for group in source_groups]


def assign_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    domains_per_group = df.groupby("leakage_group")["domain"].nunique()
    crossed = set(domains_per_group[domains_per_group > 1].index)
    df["split"] = "unused_news"
    df.loc[df.leakage_group.isin(crossed), "split"] = "excluded_cross_domain"
    eligible = df.loc[~df.leakage_group.isin(crossed)]
    abstract_groups = sorted(eligible.loc[eligible.domain.eq("abstracts"), "leakage_group"].unique())
    news_groups = sorted(eligible.loc[eligible.domain.eq("news"), "leakage_group"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(abstract_groups)
    weights = eligible.groupby("leakage_group")["source_group"].nunique().to_dict()
    abstract_total = sum(weights[g] for g in abstract_groups)
    targets = {"train": round(abstract_total * .50), "validation": round(abstract_total * .17)}
    targets["test_familiar"] = abstract_total - sum(targets.values())
    current = Counter()
    assignment = {}
    # Greedy allocation never divides a source or linked component. With no
    # cross-source links this exactly attains the planned 1500/510/990 counts.
    for group in abstract_groups:
        split = max(targets, key=lambda s: (targets[s] - current[s]) / max(targets[s], 1))
        assignment[group] = split
        current[split] += weights[group]
    rng.shuffle(news_groups)
    news_target = current["test_familiar"]
    count = 0
    for group in news_groups:
        if count >= news_target:
            break
        assignment[group] = "test_shifted"
        count += weights[group]
    if not abstract_total or count < news_target:
        raise ValueError("Insufficient eligible groups for the approved experiment.")
    for group, split in assignment.items():
        df.loc[df.leakage_group.eq(group), "split"] = split
    if df.groupby("source_group").split.nunique().max() != 1:
        raise AssertionError("A source item crosses splits.")
    if df.groupby("leakage_group").split.nunique().max() != 1:
        raise AssertionError("A linked component crosses splits.")
    if not df.loc[df.split.isin(["train", "validation"]), "domain"].eq("abstracts").all():
        raise AssertionError("Target-domain text entered model-development splits.")
    return df, {"cross_domain_components_excluded": len(crossed),
                "abstract_source_group_targets": targets, "actual_abstract_source_groups": dict(current),
                "shifted_source_group_target": news_target, "actual_shifted_source_groups": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Existing pinned CSV; otherwise download into data/raw.")
    args = parser.parse_args()
    source = args.input or ROOT / "data/raw/linguistic_analysis.csv"
    obtain_input(source, args.input is None)
    selected, source_rows = [], 0
    for chunk in pd.read_csv(source, usecols=FIELDS, dtype=str, keep_default_na=False, chunksize=20000):
        source_rows += len(chunk)
        mask = chunk.source.isin(["abstracts", "news"]) & (
            chunk.model.eq("human") | (chunk.model.eq("mistral") & chunk.prompt_style.eq("zero_shot_baseline")))
        selected.append(chunk.loc[mask])
    df = pd.concat(selected, ignore_index=True)
    df = df.sort_values(["source", "item_id", "model"], kind="stable").reset_index(drop=True)
    missing = {field: int(df[field].str.strip().str.casefold().isin(["", "nan", "none", "null", "na", "n/a"]).sum())
               for field in FIELDS}
    if any(missing.values()):
        raise ValueError(f"Missing/sentinel values in approved subset: {missing}")
    df["label"] = pd.to_numeric(df.label, errors="raise").astype(int)
    if not df.label.isin([0, 1]).all() or not df.label.eq(df.model.ne("human").astype(int)).all():
        raise ValueError("Label/generator mismatch.")
    if not df.loc[df.model.eq("human"), "prompt_style"].eq("human").all():
        raise ValueError("Unexpected human provenance.")
    df = df.rename(columns={"source": "domain"})
    original_text = df.text.copy()
    df["text"] = df.text.str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False).str.strip()
    modified = int(df.text.ne(original_text).sum())
    normalized = df.text.map(lambda s: " ".join(s.split()))
    duplicates = {"exact_excess": int(df.text.duplicated().sum()),
                  "whitespace_normalized_excess": int(normalized.duplicated().sum())}
    if any(duplicates.values()):
        raise ValueError("Unexpected exact duplicates: revise deduplication and pairing before continuing.")
    df["source_group"] = df.domain + ":" + df.item_id
    pairing = df.groupby("source_group").label.agg(["size", "sum"])
    if not pairing["size"].eq(2).all() or not pairing["sum"].eq(1).all():
        raise ValueError("Every source must have one human and one AI text.")
    df["row_id"] = df.source_group + ":" + df.model
    print(f"Selected {len(df):,} texts. Auditing all selected pairs for near duplication...", flush=True)
    edges, comparisons = similar_pairs([shingles(text) for text in df.text])
    df["leakage_group"] = component_groups(df.source_group.tolist(), edges)
    df, split_audit = assign_splits(df)
    df["length_characters"] = df.text.str.len()
    df["length_words"] = df.text.map(lambda text: len(re.findall(r"\w+", text)))
    audit = {"source_revision": REVISION, "source_url": URL, "source_sha256": SHA256,
             "source_rows": source_rows, "selected_rows": len(df), "source_items": int(df.source_group.nunique()),
             "label_definition": {"0": "human", "1": "AI"}, "missing_or_sentinel_fields": missing,
             "invalid_labels": 0, "unpaired_source_items": 0, "formatting_modified_rows": modified,
             "duplicate_checks": duplicates, "near_duplicate_rule": "Jaccard >= 0.80 on sets of casefolded word 5-grams",
             "candidate_comparisons": comparisons, "near_duplicate_pairs": len(edges),
             "cross_source_near_duplicate_pairs": sum(df.iloc[a].source_group != df.iloc[b].source_group for a,b,_ in edges),
             "leakage_components": int(df.leakage_group.nunique()), "seed": SEED,
             "split_strategy": split_audit, "source_group_split_overlap": 0, "component_split_overlap": 0}
    processed = ROOT / "data/processed"
    tables = ROOT / "outputs/tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed / "selected_texts.csv", index=False, lineterminator="\n")
    df.drop(columns="text").to_csv(tables / "split_manifest.csv", index=False, lineterminator="\n")
    links = [{"row_id_a": df.iloc[a].row_id, "row_id_b": df.iloc[b].row_id, "jaccard": score,
              "same_source": df.iloc[a].source_group == df.iloc[b].source_group} for a,b,score in edges]
    pd.DataFrame(links, columns=["row_id_a", "row_id_b", "jaccard", "same_source"]).to_csv(tables / "near_duplicate_pairs.csv", index=False)
    audit["processed_sha256"] = file_hash(processed / "selected_texts.csv")
    audit["manifest_sha256"] = file_hash(tables / "split_manifest.csv")
    (tables / "preparation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
