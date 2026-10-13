"""Summarize final-test errors and select identified extreme examples for inspection."""
import json

import numpy as np
import pandas as pd

from prepare_data import ROOT, file_hash

LENGTH_BINS = ["<1,000","1,000–1,999","2,000–3,999",">=4,000"]


def length_error_summary(predictions):
    data = predictions.copy()
    data["length_bin"] = pd.cut(data.length_characters,[-np.inf,1000,2000,4000,np.inf],right=False,labels=LENGTH_BINS)
    rows = []
    for domain in ["abstracts","news"]:
        for label,error_type in [(0,"false_positive"),(1,"false_negative")]:
            for length_bin in LENGTH_BINS:
                part = data.loc[data.domain.eq(domain)&data.label.eq(label)&data.length_bin.eq(length_bin)]
                errors = int(part.predicted_label.ne(part.label).sum())
                rows.append({"domain":domain,"error_type":error_type,"length_bin":length_bin,
                             "n_class_texts":len(part),"n_errors":errors,
                             "error_rate":errors/len(part) if len(part) else np.nan,
                             "small_sample":len(part)<30})
    return pd.DataFrame(rows)


def select_examples(predictions):
    chosen = []
    for domain in ["abstracts","news"]:
        for label,error_type in [(0,"false_positive"),(1,"false_negative")]:
            part = predictions.loc[predictions.domain.eq(domain)&predictions.label.eq(label)&predictions.predicted_label.ne(label)]
            if not part.empty:
                row = part.sort_values(["score_ai","row_id"],ascending=[label==1,True]).iloc[0].to_dict()
                row["error_type"] = error_type
                chosen.append(row)
    return pd.DataFrame(chosen)


def main():
    tables = ROOT / "outputs/tables"
    record = json.loads((tables / "generalization_run.json").read_text(encoding="utf-8"))
    if file_hash(tables / "generalization_predictions.csv") != record["output_sha256"]["predictions"]:
        raise ValueError("Generalization predictions no longer match the recorded run.")
    predictions = pd.read_csv(tables / "generalization_predictions.csv",dtype={"item_id":str},float_precision="round_trip")
    summary = length_error_summary(predictions)
    summary.to_csv(tables / "error_rates_by_length.csv",index=False,lineterminator="\n",float_format="%.17g")
    errors = predictions.loc[predictions.label.ne(predictions.predicted_label)].copy()
    errors["error_type"] = np.where(errors.label.eq(0),"false_positive","false_negative")
    errors.to_csv(tables / "test_errors.csv",index=False,lineterminator="\n",float_format="%.17g")
    selected = select_examples(predictions)
    selected.to_csv(tables / "error_example_selection.csv",index=False,lineterminator="\n",float_format="%.17g")
    corpus_path = ROOT / "data/processed/selected_texts.csv"
    if file_hash(corpus_path) != record["processed_sha256"]:
        raise ValueError("Local texts differ from the evaluated corpus.")
    texts = pd.read_csv(corpus_path,keep_default_na=False).set_index("row_id").text
    local = selected.to_dict(orient="records")
    for row in local:
        row["text"] = texts.loc[row["row_id"]]
    work = ROOT / "work/phase3"
    work.mkdir(parents=True,exist_ok=True)
    (work / "error_examples.json").write_text(json.dumps(local,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
    print(summary.to_string(index=False))
    print(f"Selected {len(selected)} extreme errors for actual text inspection; full text remains local under work/phase3.")


if __name__ == "__main__":
    main()
