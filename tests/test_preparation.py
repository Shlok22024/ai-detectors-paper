"""Regression tests for the nontrivial leakage-prevention logic."""
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import random
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_data import assign_splits, component_groups, similar_pairs


class LeakageTests(unittest.TestCase):
    def test_candidate_filter_matches_exhaustive_comparison(self):
        rng = random.Random(12)
        docs = [frozenset()]
        for _ in range(25):
            base = set(rng.sample(range(60), rng.randint(5, 35)))
            docs.append(frozenset(map(str, base)))
            for remove in (0, 1, 2, 5):
                variation = base - set(rng.sample(sorted(base), min(remove, len(base))))
                docs.append(frozenset(map(str, variation)))
        # Include exact threshold, below-threshold, and strongly unequal lengths.
        docs += [frozenset("abcde"), frozenset("abcd"), frozenset("abc"), frozenset("a")]
        for threshold in [Fraction(4, 5), Fraction(1, 2), Fraction(1, 1)]:
            expected = set()
            for i, j in combinations(range(len(docs)), 2):
                union = len(docs[i] | docs[j])
                if union and len(docs[i] & docs[j]) * threshold.denominator >= union * threshold.numerator:
                    expected.add((i, j))
            actual, _ = similar_pairs(docs, threshold)
            self.assertEqual({tuple(sorted((i, j))) for i, j, _ in actual}, expected)

    def test_links_merge_transitively_through_source_pairs(self):
        groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
        merged = component_groups(groups, [(1, 2, .9), (3, 4, .9)])
        self.assertEqual(len(set(merged[:6])), 1)
        self.assertNotEqual(merged[0], merged[6])

    def test_split_quarantines_cross_domain_and_preserves_components(self):
        rows = []
        for domain in ["abstracts", "news"]:
            for i in range(30):
                for label in [0, 1]:
                    source = f"{domain}:{i}"
                    component = "crossed" if i == 0 else (f"{domain}:linked" if i in [1, 2] else source)
                    rows.append({"domain": domain, "label": label, "source_group": source,
                                 "leakage_group": component})
        data = pd.DataFrame(rows)
        split, _ = assign_splits(data)
        self.assertTrue(split.loc[split.leakage_group.eq("crossed"), "split"].eq("excluded_cross_domain").all())
        self.assertEqual(split.groupby("leakage_group").split.nunique().max(), 1)
        self.assertEqual(split.groupby("source_group").split.nunique().max(), 1)
        self.assertTrue(split.loc[split.split.isin(["train", "validation"]), "domain"].eq("abstracts").all())
        again, _ = assign_splits(data)
        pd.testing.assert_frame_equal(split, again)


if __name__ == "__main__":
    unittest.main()
