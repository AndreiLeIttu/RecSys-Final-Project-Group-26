import unittest
from unittest.mock import patch

import numpy as np

from task1.hybrid_selection import crossfit_weights, switch_scores, user_folds
from task1.weighted_hybrid import fit_weights


class HybridSelectionTests(unittest.TestCase):
    def test_crossfit_excludes_held_out_users_from_fit(self):
        train = np.zeros((7, 5), dtype=bool)
        valid = train.copy()
        valid[1:, 1] = True
        # Distinct user values reveal exactly which users entered regression.
        scores = np.ones((1, 7, 5)) * np.arange(7)[None, :, None]
        folds = user_folds(np.arange(1, 7), 7, 2020)
        fitted_users = []

        def inspect_fit(features, targets):
            fitted_users.append(set(features[:, 0].astype(int)))
            return {"weights": np.array([1.0])}

        with patch("task1.hybrid_selection.fit_weights", side_effect=inspect_fit):
            result = crossfit_weights(scores, train, valid, 1, 2020, folds)
        for fold, observed in enumerate(fitted_users):
            self.assertEqual(observed, set(np.flatnonzero((folds >= 0) & (folds != fold))))
        np.testing.assert_allclose(result[1:], scores[0, 1:])

    def test_switch_uses_only_training_activity_and_whole_rows(self):
        first, second = np.ones((3, 4)), np.full((3, 4), 9)
        actual = switch_scores(first, second, [0, 5, 20], 5)
        np.testing.assert_array_equal(actual, [first[0], first[1], second[2]])
        reverse = switch_scores(first, second, [0, 5, 20], 5, reverse=True)
        np.testing.assert_array_equal(reverse, [second[0], second[1], first[2]])

    def test_regression_recovers_complementary_components(self):
        features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        target = 0.25 * features[:, 0] + 0.75 * features[:, 1] + 0.1
        result = fit_weights(features, target)
        np.testing.assert_allclose(result["weights"], [0.25, 0.75], atol=1e-5)
        self.assertAlmostEqual(result["intercept"], 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
