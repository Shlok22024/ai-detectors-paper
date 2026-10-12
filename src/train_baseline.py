"""Train the fixed lexical baseline and evaluate only abstract validation/test data."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from prepare_data import ROOT, SEED, file_hash

THRESHOLD = 0.5
BOOTSTRAP_DRAWS = 1000
METRICS = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "roc_auc"]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=2,
                                 lowercase=True, stop_words=None, norm="l2",
                                 use_idf=True, smooth_idf=True, sublinear_tf=False)),
        ("classifier", LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs",
                                          max_iter=1000, tol=1e-4, random_state=SEED,
                                          class_weight=None, fit_intercept=True)),
    ])


def binary_metrics(labels, scores, threshold=THRESHOLD) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not np.isin(labels, [0, 1]).all() or set(labels) != {0, 1}:
        raise ValueError("Evaluation requires both human (0) and AI (1) labels.")
    if len(labels) != len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid probability scores.")
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    return {"n_texts": len(labels), "n_human": tn + fp, "n_ai": fn + tp,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tn + tp) / len(labels),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "false_positive_rate": fp / (tn + fp), "false_negative_rate": fn / (tp + fn),
            "roc_auc": float(roc_auc_score(labels, scores))}


def cluster_intervals(predictions: pd.DataFrame, draws=BOOTSTRAP_DRAWS, seed=SEED) -> dict:
    # Each list contains *all* rows in a linked component, including both classes.
    groups = [np.asarray(indices) for indices in predictions.reset_index(drop=True).groupby("leakage_group").indices.values()]
    labels, scores = predictions.label.to_numpy(), predictions.score_ai.to_numpy()
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRICS}
    for _ in range(draws):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        values = binary_metrics(labels[indices], scores[indices])
        for name in METRICS:
            samples[name].append(values[name])
    return {name: [float(value) for value in np.quantile(samples[name], [.025, .975])] for name in METRICS}


def load_verified_data() -> tuple[pd.DataFrame, dict]:
    tables = ROOT / "outputs/tables"
    prepared = ROOT / "data/processed/selected_texts.csv"
    audit = json.loads((tables / "preparation_audit.json").read_text(encoding="utf-8"))
    if file_hash(prepared) != audit["processed_sha256"]:
        raise ValueError("Prepared corpus differs from the frozen Phase 1 checksum.")
    if file_hash(tables / "split_manifest.csv") != audit["manifest_sha256"]:
        raise ValueError("Split manifest differs from the frozen Phase 1 checksum.")
    df = pd.read_csv(prepared, dtype={"item_id": str}, keep_default_na=False)
    manifest = pd.read_csv(tables / "split_manifest.csv", dtype={"item_id": str}, keep_default_na=False)
    pd.testing.assert_frame_equal(df.drop(columns="text"), manifest)
    if not df.row_id.is_unique or df.groupby("leakage_group").split.nunique().max() != 1:
        raise ValueError("Duplicate identifiers or split overlap.")
    if df.groupby("source_group").split.nunique().max() != 1:
        raise ValueError("Source item crosses splits.")
    for split in ["train", "validation", "test_familiar"]:
        part = df.loc[df.split.eq(split)]
        if part.empty or not part.domain.eq("abstracts").all() or set(part.label) != {0, 1}:
            raise ValueError(f"Invalid abstract split: {split}")
    return df, audit


def main():
    started = time.perf_counter()
    df, preparation = load_verified_data()
    training = df.loc[df.split.eq("train")]
    validation = df.loc[df.split.eq("validation")]
    pipeline = build_pipeline()
    print(f"Fitting one fixed pipeline on {len(training):,} abstract training texts...", flush=True)
    fit_started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(training.text, training.label)
    fit_seconds = time.perf_counter() - fit_started
    classifier = pipeline.named_steps["classifier"]
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Unexpected classifier class order.")
    # Validate before accessing the final test. No model or threshold is selected.
    frames, metrics = [], []
    for split, bootstrap_seed in [("validation", SEED), ("test_familiar", SEED + 1)]:
        part = validation if split == "validation" else df.loc[df.split.eq(split)]
        with threadpool_limits(limits=1):
            scores = pipeline.predict_proba(part.text)[:, 1]
        predictions = part.drop(columns="text").copy()
        predictions["score_ai"] = scores
        predictions["predicted_label"] = (scores >= THRESHOLD).astype(int)
        predictions["correct"] = predictions.label.eq(predictions.predicted_label)
        # The intervals condition on this model; no bootstrap model is trained.
        intervals = cluster_intervals(predictions, seed=bootstrap_seed)
        values = binary_metrics(part.label, scores)
        values.update(split=split, n_source_items=int(part.source_group.nunique()),
                      n_components=int(part.leakage_group.nunique()), threshold=THRESHOLD)
        for name, (low, high) in intervals.items():
            values[f"{name}_ci_low"] = low
            values[f"{name}_ci_high"] = high
        frames.append(predictions)
        metrics.append(values)
        print(f"{split}: accuracy={values['accuracy']:.6f}; F1={values['f1']:.6f}; FPR={values['false_positive_rate']:.6f}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    if not predictions.domain.eq("abstracts").all():
        raise AssertionError("News was evaluated in Phase 2.")
    tables = ROOT / "outputs/tables"
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "baseline.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    with threadpool_limits(limits=1):
        np.testing.assert_array_equal(reloaded.predict_proba(validation.text)[:, 1], frames[0].score_ai.to_numpy())
    predictions.to_csv(tables / "baseline_predictions.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(metrics).to_csv(tables / "baseline_metrics.csv", index=False, lineterminator="\n", float_format="%.17g")
    record = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "training_script_sha256": file_hash(Path(__file__)),
        "protocol_sha256": file_hash(ROOT / "research/phase2_protocol.md"),
        "processed_sha256": preparation["processed_sha256"], "manifest_sha256": preparation["manifest_sha256"],
        "model_sha256": file_hash(model_path), "predictions_sha256": file_hash(tables / "baseline_predictions.csv"),
        "metrics_sha256": file_hash(tables / "baseline_metrics.csv"),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "package_versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"]},
        "training_texts": len(training), "training_source_items": int(training.source_group.nunique()),
        "training_class_counts": {str(k): int(v) for k,v in training.label.value_counts().sort_index().items()},
        "fitted_features": len(pipeline.named_steps["tfidf"].vocabulary_),
        "training_matrix_nonzero_values": int(pipeline.named_steps["tfidf"].transform(training.text).nnz),
        "converged": True, "solver_iterations": classifier.n_iter_.tolist(),
        "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - started, "numeric_threads": 1,
        "tfidf_parameters": {k: pipeline.named_steps["tfidf"].get_params()[k] for k in ["ngram_range","max_features","min_df","max_df","lowercase","token_pattern","stop_words","norm","smooth_idf","use_idf","sublinear_tf"]},
        "logistic_parameters": {k: classifier.get_params()[k] for k in ["C","l1_ratio","solver","max_iter","tol","class_weight","fit_intercept","random_state"]},
        "threshold": THRESHOLD, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seeds": {"validation": SEED, "test_familiar": SEED+1},
        "bootstrap_unit": "intact leakage component", "hyperparameter_search": False,
        "refit_on_validation": False, "news_predictions_produced": False,
        "saved_model_reload_predictions_identical": True,
    }
    (tables / "baseline_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved local pipeline, text-free predictions, metrics and run record in {record['total_seconds']:.1f} seconds.")


if __name__ == "__main__":
    main()
