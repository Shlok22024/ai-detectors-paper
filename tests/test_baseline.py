"""Check error-rate semantics, threshold handling and group-aware uncertainty."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_baseline import binary_metrics, build_pipeline, cluster_intervals


class BaselineTests(unittest.TestCase):
    def test_hand_calculated_imbalanced_confusion_and_rates(self):
        result = binary_metrics([0,0,0,0,1,1], [.1,.6,.8,.2,.9,.4])
        self.assertEqual([result[k] for k in ["tn","fp","fn","tp"]], [2,2,1,1])
        for metric, expected in {"accuracy": .5, "precision": 1/3, "recall": .5,
                                 "f1": .4, "false_positive_rate": .5,
                                 "false_negative_rate": .5, "roc_auc": .75}.items():
            self.assertAlmostEqual(result[metric], expected)

    def test_threshold_tie_and_no_positive_predictions(self):
        at_tie = binary_metrics([0,1],[.5,.5])
        self.assertEqual([at_tie[k] for k in ["tn","fp","fn","tp"]], [0,1,0,1])
        none = binary_metrics([0,1],[.1,.2])
        self.assertEqual(none["precision"], 0)
        self.assertEqual(none["f1"], 0)

    def test_group_resampling_keeps_complementary_pair_errors_together(self):
        # Every source has exactly one error. Whole-group accuracy must stay 0.5;
        # independently resampling rows would introduce incorrect uncertainty.
        predictions = pd.DataFrame({"label":[0,1,0,1],"score_ai":[.8,.8,.2,.2],
                                    "leakage_group":["a","a","b","b"]})
        intervals = cluster_intervals(predictions,draws=100,seed=7)
        self.assertEqual(intervals["accuracy"],[.5,.5])

    def test_transforming_heldout_text_does_not_fit_its_vocabulary(self):
        pipe = build_pipeline()
        pipe.fit(["humanword common term", "humanword common text",
                  "machineword common term", "machineword common text"],[0,0,1,1])
        original = pipe.named_steps["tfidf"].vocabulary_.copy()
        pipe.predict_proba(["holdoutonlyzzz machineword"])
        self.assertEqual(pipe.named_steps["tfidf"].vocabulary_, original)
        self.assertNotIn("holdoutonlyzzz",original)


if __name__ == "__main__":
    unittest.main()
