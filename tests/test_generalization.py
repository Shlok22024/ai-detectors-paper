from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from analyze_errors import exact_error_interval, length_error_summary, select_examples
from evaluate_generalization import independent_changes


class GeneralizationTests(unittest.TestCase):
    def test_zero_errors_still_have_nonzero_binomial_upper_limit(self):
        low,high = exact_error_interval(0,990)
        self.assertEqual(low,0)
        self.assertAlmostEqual(high,1-.025**(1/990),places=12)

    def test_independent_change_sign_and_intact_pair_resampling(self):
        familiar = pd.DataFrame({"label":[0,1,0,1],"score_ai":[.1,.9,.1,.9],"leakage_group":["a","a","b","b"]})
        shifted = pd.DataFrame({"label":[0,1,0,1],"score_ai":[.1,.1,.1,.1],"leakage_group":["c","c","d","d"]})
        changes = independent_changes(familiar,shifted,draws=20).set_index("metric")
        self.assertEqual(changes.loc["accuracy"].tolist(),[-.5,-.5,-.5])
        self.assertEqual(changes.loc["false_negative_rate"].tolist(),[1,1,1])
        with self.assertRaises(ValueError):
            independent_changes(familiar,familiar,draws=2)

    def test_length_boundaries_empty_denominators_and_small_cells(self):
        data = pd.DataFrame({"domain":["news"]*4,"label":[0]*4,"predicted_label":[1,0,1,0],
                             "length_characters":[999,1000,2000,4000]})
        summary = length_error_summary(data)
        human = summary.query("domain=='news' and error_type=='false_positive'")
        self.assertEqual(human.n_class_texts.tolist(),[1,1,1,1])
        self.assertEqual(human.n_errors.tolist(),[1,0,1,0])
        self.assertTrue(human.small_sample.all())
        empty = summary.query("error_type=='false_negative'")
        self.assertTrue(empty.error_rate.isna().all())

    def test_example_rule_is_extreme_score_with_deterministic_ties(self):
        data = pd.DataFrame({"domain":["news"]*5,"label":[0,0,0,1,1],"predicted_label":[1,1,1,0,0],
                             "score_ai":[.8,.9,.9,.2,.1],"row_id":["z","b","a","d","e"]})
        selected = select_examples(data)
        self.assertEqual(selected.row_id.tolist(),["a","e"])


if __name__ == "__main__":
    unittest.main()
