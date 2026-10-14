"""Generate a bounded interpretation report from verified coefficient/score tables."""
import json

import numpy as np
import pandas as pd

from prepare_data import ROOT, file_hash
from report_generalization import table


def main():
    tables = ROOT / "outputs/tables"
    record = json.loads((tables / "interpretation_run.json").read_text(encoding="utf-8"))
    for name,expected in record["output_sha256"].items():
        if file_hash(tables / f"{name}.csv")!=expected:
            raise ValueError(f"Interpretation table changed: {name}")
    coefficients = pd.read_csv(tables / "coefficient_features.csv",float_precision="round_trip")
    scores = pd.read_csv(tables / "score_decomposition.csv",float_precision="round_trip")
    shifts = pd.read_csv(tables / "ai_feature_contribution_changes.csv",float_precision="round_trip")
    indexed = scores.set_index(["domain","class"])
    ai_abs,ai_news = indexed.loc[("abstracts","ai")],indexed.loc[("news","ai")]
    mean_shift = ai_news.mean_logit-ai_abs.mean_logit
    np.testing.assert_allclose(mean_shift,record["all_feature_contribution_changes_sum"],atol=1e-12,rtol=0)
    np.testing.assert_allclose(scores.intercept+scores.mean_positive_contribution+scores.mean_negative_contribution,scores.mean_logit,atol=1e-12,rtol=0)
    coef_rows=[]
    for association in ["AI","human"]:
        for row in coefficients.loc[coefficients.association.eq(association)].head(8).itertuples(index=False):
            coef_rows.append({"Training association":association,"Term":row.feature,"Coefficient":f"{row.coefficient:+.4f}",
                              "Human training texts":f"{row.training_human_texts_with_term}/1500",
                              "AI training texts":f"{row.training_ai_texts_with_term}/1500"})
    score_rows=[
                {"Domain":row['domain'],"Class":row['class'],"Median AI score":f"{row['median_score_ai']:.4f}",
                 "5th–95th percentile":f"{row['score_p05']:.4f}–{row['score_p95']:.4f}","Predicted AI / texts":f"{row['n_predicted_ai']}/{row['n_texts']}"}
                for row in scores.to_dict(orient="records")]
    decomposition_rows=[{"Domain":row['domain'],"Class":row['class'],"Intercept":f"{row['intercept']:.4f}",
                         "Mean positive contribution":f"{row['mean_positive_contribution']:+.4f}",
                         "Mean negative contribution":f"{row['mean_negative_contribution']:+.4f}","Mean logit":f"{row['mean_logit']:+.4f}"}
                        for row in scores.to_dict(orient="records")]
    shift_rows=[{"Term":row.feature,"Coefficient":f"{row.coefficient:+.4f}",
                 "Mean abstract-AI TF-IDF":f"{row.abstract_ai_mean_tfidf:.4f}","Mean news-AI TF-IDF":f"{row.news_ai_mean_tfidf:.4f}",
                 "News-minus-abstract contribution":f"{row.mean_logit_contribution_change:+.4f}"}
                for row in shifts.groupby("direction",sort=False).head(5).itertuples(index=False)]
    report=f"""# Coefficients, score shift and limits of explanation

Phase 4 completed October 13, 2026. This analysis interprets the same frozen TF-IDF/logistic detector evaluated in Phases 2 and 3. There is no new classifier, SHAP analysis, threshold optimization or target-domain refit.

## What the experiment establishes

The detector's fixed-threshold accuracy fell from 97.78% on abstracts to 50.10% on news, with news AI recall of 0.20%. News ROC-AUC remained 0.8073: the model retained meaningful ranking ability, although it missed almost every AI news text at the unchanged 0.5 threshold. The correct finding is a severe **fixed-threshold transfer failure**, accompanied by reduced ranking performance, rather than a complete loss of class discrimination.

The [Phase 4 plan](../research/phase4_plan.md) was recorded after those results were known. Its feature-selection and accounting rules guide this descriptive analysis; it is not a preregistration of the earlier experimental finding.

## Learned lexical associations

The largest positive coefficients include `our`, `insights`, `insights into`, `findings`, `additionally` and `we present`. Large negative coefficients include `the`, `is`, `of`, `be` and `are`. These include common function words and abstract-style phrases, not universal AI fingerprints. Positive weights move the fitted score toward AI; negative weights move it toward human, conditional on the other represented features.

![Selected coefficients](../outputs/figures/coefficient_features.png)

{table(pd.DataFrame(coef_rows))}

The complete selected table contains 15 terms in each direction, plus training document counts and final-test counts for each domain/class. Selection is by coefficient magnitude in the already fitted training model, with lexical tie-breaking. No news performance was used to choose coefficients. A term may occur in both classes; the detector operates on normalized TF-IDF values, not just presence. Correlated unigrams and bigrams share information, and L2 regularization distributes weight across them. Magnitude alone is neither causal importance nor the contribution of a particular text.

## Score distributions and retained ranking

![Empirical score distributions](../outputs/figures/score_distributions.png)

{table(pd.DataFrame(score_rows))}

The median AI score for AI-labeled text shifts from **{ai_abs.median_score_ai:.4f}** in abstracts to **{ai_news.median_score_ai:.4f}** in news. Human news scores also move downward. AI news generally scores above human news, consistent with the retained ROC-AUC. Nearly all news texts in both classes score below 0.5. This is a description of the observed scores; no alternative threshold has been chosen or evaluated.

Each domain/class has 990 final-test texts. The 5th–95th percentiles describe their spread and are not confidence intervals. Scores are outputs of this classifier, not validated authorship probabilities. Of {int(scores.n_texts.sum()):,} evaluated texts, {int(scores.zero_vocabulary_vectors.sum()):,} have all-zero vocabulary vectors; partial vocabulary coverage was not separately quantified.

## Exact accounting within the fitted model

For each text, the linear score is `intercept + sum(TF-IDF_j * coefficient_j)`, and the AI score is its logistic transform. A zero linear score corresponds to the fixed 0.5 threshold. The analysis separates positive and negative weighted contributions and checks their sum against the pipeline's decision function and saved predictions. Maximum reconstruction errors are {record['logit_reconstruction_max_abs_error']:.2g} in linear scores and {record['probability_reconstruction_max_abs_error']:.2g} in probabilities, within floating-point tolerance.

{table(pd.DataFrame(decomposition_rows))}

For AI text, the mean positive contribution falls from {ai_abs.mean_positive_contribution:.4f} to {ai_news.mean_positive_contribution:.4f} (change **{ai_news.mean_positive_contribution-ai_abs.mean_positive_contribution:.4f}**), and the mean negative contribution changes by **{ai_news.mean_negative_contribution-ai_abs.mean_negative_contribution:.4f}**. The intercept stays fixed. Together, the mean AI logit changes by **{mean_shift:.4f}**, from {ai_abs.mean_logit:.4f} to {ai_news.mean_logit:.4f}. These means reconstruct the mean linear score; applying the logistic transform to a mean logit does not equal the mean probability.

The selected term changes below use the coefficient multiplied by the difference in mean TF-IDF value for AI news versus AI abstracts. The full sum across all 30,000 features reconstructs the mean logit change; this abbreviated table shows five of the ten largest changes in each direction.

{table(pd.DataFrame(shift_rows))}

For example, `the` has a negative fitted coefficient and a larger mean normalized weight in AI news, adding a negative contribution change of {shifts.loc[shifts.feature.eq('the'),'mean_logit_contribution_change'].iloc[0]:.4f}. The positive weight on `our` contributes less in AI news than abstracts. This accounts for parts of the score shift arithmetically. It does not show that the underlying distribution changed because of domain alone, or that inserting/removing a word would produce a realistic causal explanation. TF-IDF normalization, word co-occurrence and correlated features complicate such interventions.

## Writing domain, generation design and human sources

The compared conditions retain the release's `mistral` generator identity and `zero_shot_baseline` prompt-style label. **The literal prompts and generation tasks differ.** [Xia et al.'s official paper](https://aclanthology.org/2026.eacl-long.307.pdf), Table 6, printed page 6539 (PDF page 16), describes abstract generation from a research category and title, and news generation from article highlights. Both templates request a character length based on the source item. The [pinned author prompt-construction code](https://github.com/Yuuxii/Generalization-of-AI-text-Detector/blob/316184f7d87659f8d348685dc6967c3b0e0cb235/src/generate_prompts.py) substitutes these different fields and rounds source character lengths to the nearest ten. The literal template JSON referenced by that code is absent from the inspected pinned repository tree; the published paper supplies the template descriptions used here.

| Factor | What is controlled or known | What changes or remains unverified |
| --- | --- | --- |
| Detector | Same fitted coefficients, vocabulary and 0.5 threshold | No refitting or target-domain calibration |
| Generator | Same release label: Mistral, mapped by the authors to Mistral-Large-Instruct-2411 | Per-text generator provenance is inherited from the release |
| Prompt style | Zero-shot baseline in both conditions | Category/title-to-abstract versus highlights-to-news tasks and literal template inputs differ |
| Human corpus | Human/AI pairs share an item within each source dataset | arXiv abstracts versus CNN/Daily Mail articles; authors and original source URLs are not available in the selected release |
| Length | Length requests depend on source items | Source and generated length distributions differ by class/domain |
| Cleaning | Analysis retains released text, with only Phase 1 boundary/line-ending normalization | Paper reports different cleaning steps for human and AI text; uncleaned counterparts are not analyzed here |
| Generation runtime | Model and prompt-style labels are available | Main CSV lacks per-text raw prompts, decoding settings and generation seeds; identical runtime conditions cannot be fully audited |

This is valid as a test of transfer between the released abstract and news conditions. It is not a factorial experiment that separates writing domain, prompt/task design, source-human differences, length and preprocessing. Our coefficient and score accounting cannot identify their separate causal contributions. The short name “domain transfer” refers to this bundled distribution change.

## Formatting and explanatory limits

The retained model lowercases word tokens and has no explicit punctuation, capitalization, layout or formatting coefficients. Its token boundaries and vocabulary can indirectly reflect text preparation, but this phase does not measure a formatting effect. No character model or extra classifier is introduced to fill that gap.

Observed findings are the coefficients, term distributions, score distributions and fixed-threshold outcomes. Exact arithmetic shows how represented features form the fitted scores. A plausible explanation is reliance on associations learned from the abstract collection that transfer poorly to the news collection. Whether prompting, human source composition, length or cleaning produced those feature changes remains unverified. Neither causal mechanisms nor intrinsic properties of AI writing have been established.

## Reproduction and checkpoint

After Phases 1–3, run `python src/interpret_model.py`, then `python src/report_interpretation.py`. Three text-free tables, two figures and `interpretation_run.json` are regenerated. The analysis checks the saved model, data and predictions, reconstructs logits/probabilities and verifies the sum of all feature contribution changes. It only transforms existing texts and does not call fit. The report independently checks aggregate accounting and table checksums. Two additional tests verify signed score reconstruction and feature-sign/tie selection; 13 regression tests pass in total. Figures were visually inspected before the checkpoint commit.

No experimental performance result changed in Phase 4. Full texts and the saved pipeline stay local. AI assistance was used for implementation, source inspection and interpretation; no independent validation or manual review of all texts is claimed. Remaining work is final figure selection, the summary card, consolidated reporting and a reproduction check, rather than expanding the experiment.
"""
    (ROOT / "report/interpretation_report.md").write_text(report,encoding="utf-8",newline="\n")
    print("Verified table hashes and mean-score accounting; generated report/interpretation_report.md.")


if __name__=="__main__":
    main()
