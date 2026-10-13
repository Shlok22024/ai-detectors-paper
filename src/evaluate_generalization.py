"""Evaluate the frozen baseline on news and compare independent domain samples."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from prepare_data import ROOT, file_hash
from train_baseline import METRICS, binary_metrics, cluster_intervals, load_verified_data


def component_indices(frame):
    return [np.asarray(indices) for indices in frame.reset_index(drop=True).groupby("leakage_group").indices.values()]


def independent_changes(familiar, shifted, draws=1000, seed=45):
    if set(familiar.leakage_group) & set(shifted.leakage_group):
        raise ValueError("Independent-domain comparison cannot reuse leakage components.")
    groups = [component_indices(frame) for frame in [familiar, shifted]]
    arrays = [(frame.label.to_numpy(), frame.score_ai.to_numpy()) for frame in [familiar, shifted]]
    point = [binary_metrics(*array) for array in arrays]
    rng = np.random.default_rng(seed)
    differences = {metric: [] for metric in METRICS}
    for _ in range(draws):
        sampled = []
        for group, (labels,scores) in zip(groups, arrays):
            indices = np.concatenate([group[i] for i in rng.integers(len(group),size=len(group))])
            sampled.append(binary_metrics(labels[indices],scores[indices]))
        for metric in METRICS:
            differences[metric].append(sampled[1][metric]-sampled[0][metric])
    return pd.DataFrame([{"metric":metric, "news_minus_abstracts":point[1][metric]-point[0][metric],
                          "ci_low":float(np.quantile(differences[metric],.025)),
                          "ci_high":float(np.quantile(differences[metric],.975))} for metric in METRICS])


def main():
    started = time.perf_counter()
    tables = ROOT / "outputs/tables"
    baseline = json.loads((tables / "baseline_run.json").read_text(encoding="utf-8"))
    model_path = ROOT / "models/baseline.joblib"
    if file_hash(model_path) != baseline["model_sha256"]:
        raise ValueError("Saved model differs from the Phase 2 artifact. Do not retrain to bypass this check.")
    for name in ["predictions", "metrics"]:
        if file_hash(tables / f"baseline_{name}.csv") != baseline[f"{name}_sha256"]:
            raise ValueError(f"Phase 2 {name} differs from its run record.")
    data, preparation = load_verified_data()
    if preparation["processed_sha256"] != baseline["processed_sha256"] or preparation["manifest_sha256"] != baseline["manifest_sha256"]:
        raise ValueError("Phase 1 inputs differ from the Phase 2 model inputs.")
    shifted = data.loc[data.split.eq("test_shifted")].copy()
    if len(shifted) != 1980 or not shifted.domain.eq("news").all() or shifted.label.value_counts().to_dict() != {0:990,1:990}:
        raise ValueError("News test differs from the approved 990 balanced source pairs.")
    pipeline = joblib.load(model_path)
    if pipeline.named_steps["classifier"].classes_.tolist() != [0,1] or baseline["threshold"] != .5:
        raise ValueError("Unexpected class direction or threshold.")
    phase2 = pd.read_csv(tables / "baseline_predictions.csv",dtype={"item_id":str},float_precision="round_trip")
    familiar = phase2.loc[phase2.split.eq("test_familiar")].copy()
    abstract_data = data.loc[data.split.eq("test_familiar")].set_index("row_id").loc[familiar.row_id]
    with threadpool_limits(limits=1):
        np.testing.assert_allclose(pipeline.predict_proba(abstract_data.text)[:,1],familiar.score_ai,atol=1e-14,rtol=0)
        scores = pipeline.predict_proba(shifted.text)[:,1]
    news = shifted.drop(columns="text").copy()
    news["score_ai"] = scores
    news["predicted_label"] = (scores >= baseline["threshold"]).astype(int)
    news["correct"] = news.label.eq(news.predicted_label)
    all_predictions = pd.concat([familiar,news],ignore_index=True)
    if not all_predictions.row_id.is_unique:
        raise ValueError("Duplicate evaluation IDs.")
    if set(all_predictions.leakage_group) & set(data.loc[data.split.isin(["train","validation"]),"leakage_group"]):
        raise ValueError("Evaluation components overlap model development.")
    news_values = binary_metrics(news.label,news.score_ai)
    news_values.update(split="test_shifted",domain="news",n_source_items=int(news.source_group.nunique()),
                       n_components=int(news.leakage_group.nunique()),threshold=baseline["threshold"])
    for name,(low,high) in cluster_intervals(news,draws=1000,seed=44).items():
        news_values[f"{name}_ci_low"] = low
        news_values[f"{name}_ci_high"] = high
    familiar_values = pd.read_csv(tables / "baseline_metrics.csv",float_precision="round_trip").query("split == 'test_familiar'").copy()
    familiar_values["domain"] = "abstracts"
    metrics = pd.concat([familiar_values,pd.DataFrame([news_values])],ignore_index=True)
    changes = independent_changes(familiar,news)
    all_predictions.to_csv(tables / "generalization_predictions.csv",index=False,lineterminator="\n",float_format="%.17g")
    metrics.to_csv(tables / "generalization_metrics.csv",index=False,lineterminator="\n",float_format="%.17g")
    changes.to_csv(tables / "generalization_changes.csv",index=False,lineterminator="\n",float_format="%.17g")
    if file_hash(model_path) != baseline["model_sha256"]:
        raise AssertionError("Saved model changed during evaluation.")
    record = {"completed_at_utc":datetime.now(timezone.utc).isoformat(),
              "code_base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "evaluation_script_sha256":file_hash(Path(__file__)),"protocol_sha256":file_hash(ROOT / "research/phase3_protocol.md"),
              "baseline_model_sha256":baseline["model_sha256"],"processed_sha256":baseline["processed_sha256"],
              "manifest_sha256":baseline["manifest_sha256"],"baseline_predictions_sha256":baseline["predictions_sha256"],
              "output_sha256":{name:file_hash(tables/f"generalization_{name}.csv") for name in ["predictions","metrics","changes"]},
              "threshold":.5,"bootstrap_draws":1000,"news_interval_seed":44,"independent_change_seed":45,
              "refit":False,"recalibration":False,"unused_news_scored":False,
              "phase2_familiar_predictions_reproduced":True,"model_unchanged_after_scoring":True,
              "seconds":time.perf_counter()-started}
    (tables / "generalization_run.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"News: accuracy={news_values['accuracy']:.6f}; F1={news_values['f1']:.6f}; FPR={news_values['false_positive_rate']:.6f}; FNR={news_values['false_negative_rate']:.6f}")
    print(changes.to_string(index=False))


if __name__ == "__main__":
    main()
