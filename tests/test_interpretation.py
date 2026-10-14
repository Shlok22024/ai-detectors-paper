from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from interpret_model import score_components, select_coefficients


class InterpretationTests(unittest.TestCase):
    def test_signed_contributions_reconstruct_linear_scores_with_intercept(self):
        matrix = csr_matrix([[2,.5,9],[0,3,1]],dtype=float)
        positive,negative,logits = score_components(matrix,np.array([1,-3,0]),-1)
        np.testing.assert_array_equal(positive,[2,0])
        np.testing.assert_array_equal(negative,[-1.5,-9])
        np.testing.assert_array_equal(logits,[-.5,-10])

    def test_feature_selection_uses_sign_and_lexical_tie_breaks(self):
        selected = select_coefficients(["b","a","c","d","zero"],np.array([2,2,-1,-3,0]),count=1)
        self.assertEqual(selected.feature.tolist(),["a","d"])
        self.assertEqual(selected.association.tolist(),["AI","human"])


if __name__ == "__main__":
    unittest.main()
