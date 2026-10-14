"""Describe learned associations and account for scores without modifying the model."""
from datetime import datetime, timezone
import json
import subprocess

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from prepare_data import ROOT, file_hash
from train_baseline import load_verified_data


def score_components(matrix, coefficients, intercept):
    positive = np.asarray(matrix @ np.where(coefficients>0,coefficients,0)).ravel()
    negative = np.asarray(matrix @ np.where(coefficients<0,coefficients,0)).ravel()
    return positive,negative,intercept+positive+negative


def select_coefficients(features, coefficients, count=15):
    frame = pd.DataFrame({"feature":features,"coefficient":coefficients})
    positive = frame.loc[frame.coefficient.gt(0)].sort_values(["coefficient","feature"],ascending=[False,True]).head(count)
    negative = frame.loc[frame.coefficient.lt(0)].sort_values(["coefficient","feature"],ascending=[True,True]).head(count)
    return pd.concat([positive.assign(association="AI"),negative.assign(association="human")],ignore_index=True)


def main():
    tables = ROOT / "outputs/tables"
    phase2 = json.loads((tables / "baseline_run.json").read_text(encoding="utf-8"))
    phase3 = json.loads((tables / "generalization_run.json").read_text(encoding="utf-8"))
    model_path = ROOT / "models/baseline.joblib"
    if file_hash(model_path)!=phase2["model_sha256"] or phase3["baseline_model_sha256"]!=phase2["model_sha256"]:
        raise ValueError("Model differs from the evaluated frozen artifact.")
    if file_hash(tables / "generalization_predictions.csv")!=phase3["output_sha256"]["predictions"]:
        raise ValueError("Phase 3 prediction checksum mismatch.")
    data,preparation = load_verified_data()
    predictions = pd.read_csv(tables / "generalization_predictions.csv",dtype={"item_id":str},float_precision="round_trip")
    evaluated = data.set_index("row_id").loc[predictions.row_id].reset_index()
    if preparation["processed_sha256"]!=phase3["processed_sha256"]:
        raise ValueError("Corpus changed since evaluation.")
    training = data.loc[data.split.eq("train")].copy()
    pipeline = joblib.load(model_path)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    features = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    with threadpool_limits(limits=1):
        matrix = vectorizer.transform(evaluated.text)
        train_matrix = vectorizer.transform(training.text)
        positive,negative,logits = score_components(matrix,coefficients,intercept)
        np.testing.assert_allclose(logits,pipeline.decision_function(evaluated.text),atol=1e-12,rtol=0)
        np.testing.assert_allclose(expit(logits),predictions.score_ai,atol=1e-14,rtol=0)
    selected = select_coefficients(features,coefficients)
    vocabulary = {feature:i for i,feature in enumerate(features)}
    selected_indices = [vocabulary[feature] for feature in selected.feature]
    for label,class_name in [(0,"human"),(1,"ai")]:
        mask = training.label.to_numpy()==label
        selected[f"training_{class_name}_texts_with_term"] = np.asarray((train_matrix[mask][:,selected_indices]>0).sum(axis=0)).ravel()
    rows = []
    means = {}
    for domain in ["abstracts","news"]:
        for label,class_name in [(0,"human"),(1,"ai")]:
            mask = (evaluated.domain.to_numpy()==domain)&(evaluated.label.to_numpy()==label)
            values = predictions.loc[mask,"score_ai"].to_numpy()
            submatrix = matrix[mask]
            selected[f"{domain}_{class_name}_texts_with_term"] = np.asarray((submatrix[:,selected_indices]>0).sum(axis=0)).ravel()
            means[(domain,label)] = np.asarray(submatrix.mean(axis=0)).ravel()
            rows.append({"domain":domain,"class":class_name,"n_texts":int(mask.sum()),
                         "median_score_ai":float(np.median(values)),"mean_score_ai":float(np.mean(values)),
                         "score_p05":float(np.quantile(values,.05)),"score_p95":float(np.quantile(values,.95)),
                         "n_predicted_ai":int((values>=.5).sum()),"intercept":intercept,
                         "mean_positive_contribution":float(positive[mask].mean()),
                         "mean_negative_contribution":float(negative[mask].mean()),
                         "mean_logit":float(logits[mask].mean()),
                         "zero_vocabulary_vectors":int((submatrix.getnnz(axis=1)==0).sum())})
    summaries = pd.DataFrame(rows)
    delta = coefficients*(means[("news",1)]-means[("abstracts",1)])
    all_changes = pd.DataFrame({"feature":features,"coefficient":coefficients,
                                "abstract_ai_mean_tfidf":means[("abstracts",1)],
                                "news_ai_mean_tfidf":means[("news",1)],"mean_logit_contribution_change":delta})
    down = all_changes.loc[all_changes.mean_logit_contribution_change.lt(0)].sort_values(["mean_logit_contribution_change","feature"]).head(10)
    up = all_changes.loc[all_changes.mean_logit_contribution_change.gt(0)].sort_values(["mean_logit_contribution_change","feature"],ascending=[False,True]).head(10)
    changes = pd.concat([down.assign(direction="negative change"),up.assign(direction="positive change")],ignore_index=True)
    observed_logit_change = float(summaries.query("domain=='news' and `class`=='ai'").mean_logit.iloc[0]-summaries.query("domain=='abstracts' and `class`=='ai'").mean_logit.iloc[0])
    np.testing.assert_allclose(delta.sum(),observed_logit_change,atol=1e-12,rtol=0)
    for name,frame in [("coefficient_features",selected),("score_decomposition",summaries),("ai_feature_contribution_changes",changes)]:
        frame.to_csv(tables / f"{name}.csv",index=False,lineterminator="\n",float_format="%.17g")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    shown = selected.sort_values("coefficient")
    fig,ax = plt.subplots(figsize=(9,9))
    ax.barh(shown.feature,shown.coefficient,color=np.where(shown.coefficient.gt(0),"#D17A43","#2A6381"))
    ax.axvline(0,color="#697580",lw=.8)
    ax.set_xlabel("Logistic coefficient (negative: human; positive: AI)")
    ax.set_title("Words and phrases associated with the training labels",loc="left",fontweight="bold",pad=15)
    fig.subplots_adjust(left=.23,right=.96,bottom=.12,top=.91)
    fig.text(.06,.025,"15 coefficients per direction. Fitted on abstracts only. Magnitude is not causal importance or a text's contribution.",fontsize=8,color="#53616D")
    fig.savefig(ROOT / "outputs/figures/coefficient_features.png",dpi=180,facecolor="white")
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4.7),sharey=True)
    for ax,domain in zip(axes,["abstracts","news"]):
        for label,name,color in [(0,"Human","#2A6381"),(1,"AI (Mistral)","#D17A43")]:
            values = np.sort(predictions.loc[predictions.domain.eq(domain)&predictions.label.eq(label),"score_ai"])
            ax.step(np.r_[0,values,1],np.r_[0,np.arange(1,len(values)+1)/len(values),1],where="post",label=name,color=color,lw=2)
        ax.axvline(.5,color="#52616D",ls="--",lw=1.2,label="Fixed threshold")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1.04)
        ax.set_title("Familiar abstracts" if domain=="abstracts" else "Unseen news",loc="left",fontweight="bold")
        ax.set_xlabel("AI score")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Fraction of texts at or below score")
    axes[1].legend(frameon=False,loc="lower right")
    fig.subplots_adjust(left=.09,right=.98,bottom=.21,top=.83,wspace=.12)
    fig.text(.06,.025,"990 texts per class in each domain. Curves describe scores; the fixed 0.5 threshold was not optimized on news.",fontsize=8,color="#53616D")
    fig.savefig(ROOT / "outputs/figures/score_distributions.png",dpi=180,facecolor="white")
    plt.close(fig)
    if file_hash(model_path)!=phase2["model_sha256"]:
        raise AssertionError("Saved model changed during interpretation.")
    record = {"completed_at_utc":datetime.now(timezone.utc).isoformat(),
              "code_base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "script_sha256":file_hash(ROOT / "src/interpret_model.py"),"model_sha256":phase2["model_sha256"],
              "predictions_sha256":phase3["output_sha256"]["predictions"],"intercept":intercept,
              "ai_mean_logit_change_news_minus_abstracts":observed_logit_change,
              "all_feature_contribution_changes_sum":float(delta.sum()),
              "logit_reconstruction_max_abs_error":float(np.max(np.abs(logits-pipeline.decision_function(evaluated.text)))),
              "probability_reconstruction_max_abs_error":float(np.max(np.abs(expit(logits)-predictions.score_ai))),
              "model_unchanged":True,"refit":False,"threshold_optimized":False,
              "output_sha256":{name:file_hash(tables/f"{name}.csv") for name in ["coefficient_features","score_decomposition","ai_feature_contribution_changes"]}}
    (tables / "interpretation_run.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(selected[["feature","coefficient","association"]].to_string(index=False))
    print(summaries.to_string(index=False))
    print(changes[["feature","coefficient","mean_logit_contribution_change"]].to_string(index=False))


if __name__ == "__main__":
    main()
