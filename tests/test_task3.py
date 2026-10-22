import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from task3.evaluate import run, select_setting
from task3.metrics import evaluate_lists, gini
from task3.prepare import fit_hybrid, validation_users
from task3.rerankers import (
    METHODS, build_profiles, candidates, miscalibration, popularity_deviation, rerank,
)


class RerankingTests(unittest.TestCase):
    def setUp(self):
        self.train = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 1, 1, 0, 0, 0]], dtype=bool)
        self.genres = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)
        self.profiles = build_profiles(self.train, self.genres)
        self.scores = np.array([0, 1, .9, .8, .7, .6])

    def test_candidate_masks_stable_ties_and_no_mutation(self):
        scores = np.array([[100., 3, 3, 2, 1]])
        before = scores.copy()
        seen = np.array([[0, 0, 0, 1, 0]], dtype=bool)
        np.testing.assert_array_equal(candidates(scores, seen, 10), [[1, 2, 4, -1, -1]])
        np.testing.assert_array_equal(scores, before)
        with self.assertRaises(ValueError):
            candidates(scores * np.nan, seen, 3)

    def test_zero_strength_and_candidate_integrity(self):
        for method in METHODS:
            self.assertEqual(rerank([2, 3, 4, -1], self.scores, method, 0, 2, self.profiles, 1), [2, 3])
            ranked = rerank([2, 3, 4], self.scores, method, .5, 10, self.profiles, 1)
            self.assertEqual(set(ranked), {2, 3, 4})
            self.assertEqual(len(ranked), 3)
            self.assertEqual(rerank([], self.scores, method, 1, 2, self.profiles, 1), [])

    def test_diversity_introduces_a_different_genre(self):
        self.assertEqual(rerank([1, 2, 3], self.scores, "diversity", 1, 2, self.profiles, 1), [1, 3])

    def test_calibration_follows_history_not_highest_score(self):
        # User 1's history is entirely the first genre.
        scores = np.array([0, 0, .5, 1, .9, .4])
        ranked = rerank([3, 4, 2, 5], scores, "calibration", 1, 2, self.profiles, 1)
        self.assertEqual(ranked, [2, 5])

    def test_user_and_item_fairness_have_different_targets(self):
        # User 1 prefers head items; catalog target is 20% head, 80% tail.
        user = rerank([1, 2, 3], self.scores, "user_fairness", 1, 1, self.profiles, 1)
        item = rerank([1, 2, 3], self.scores, "item_fairness", 1, 1, self.profiles, 1)
        self.assertEqual(user, [1])
        self.assertEqual(item, [2])

    def test_metrics_on_hand_calculated_examples(self):
        self.assertAlmostEqual(float(miscalibration([1, 0], [0, 1])), np.log(100))
        self.assertAlmostEqual(float(miscalibration([.5, .5], [.5, .5])), 0)
        self.assertAlmostEqual(float(popularity_deviation([.8, .2], [.2, .8])), .6)
        self.assertAlmostEqual(gini([1, 1, 1]), 0)
        self.assertAlmostEqual(gini([0, 0, 3]), 2/3)
        truth = np.zeros_like(self.train)
        truth[1, 3] = truth[2, 4] = True
        metrics, _ = evaluate_lists([[3], [3]], [1, 2], truth, self.profiles, 1)
        self.assertAlmostEqual(metrics["ActivityGRU@1"], 1)
        self.assertAlmostEqual(metrics["ExposureGap@1"], .2)
        self.assertAlmostEqual(metrics["ExposureGini@1"], .8)

    def test_fractional_genres_and_training_popularity(self):
        genres = self.genres.copy()
        genres[1] = [1, 1]
        profiles = build_profiles(self.train, genres)
        np.testing.assert_allclose(profiles["user_genres"][1], [.5, .5])
        np.testing.assert_array_equal(profiles["groups"][1], [1, 0])
        np.testing.assert_allclose(profiles["target_exposure"], [.2, .8])

    def test_fit_and_tune_users_are_disjoint_and_repeatable(self):
        valid = np.ones((9, 3), dtype=bool)
        fit, tune = validation_users(valid, 2020)
        self.assertFalse(set(fit) & set(tune))
        self.assertEqual(set(fit) | set(tune), set(range(1, 9)))
        np.testing.assert_array_equal(tune, validation_users(valid, 2020)[1])

    def test_selection_respects_accuracy_budget_and_zero_fallback(self):
        rows = [{"strength": 0., "NDCG@10": .5, "UPD@10": .4},
                {"strength": .5, "NDCG@10": .49, "UPD@10": .2},
                {"strength": 1., "NDCG@10": .3, "UPD@10": .1}]
        self.assertEqual(select_setting(rows, "user_fairness", 10, .05)["strength"], .5)
        self.assertEqual(select_setting(rows, "user_fairness", 10, 0)["strength"], 0)

    def test_hybrid_refit_ignores_reranker_tuning_labels(self):
        train = np.zeros((5, 8), dtype=bool)
        train[1:, 1] = True
        valid = np.zeros_like(train)
        valid[1:3, 2] = True
        valid[3:, 3] = True
        scores = {name: np.random.default_rng(seed).random(train.shape)
                  for seed, name in enumerate(("EASE", "SLIMElastic"))}
        first, record = fit_hybrid(scores, list(scores), train, valid, [1, 2], 10, 2020)
        valid[3:] = ~valid[3:]
        second, repeated = fit_hybrid(scores, list(scores), train, valid, [1, 2], 10, 2020)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(record, repeated)
        self.assertEqual(set(record["weights"]), set(scores))
        self.assertAlmostEqual(sum(record["weights"].values()), 1)
        self.assertEqual(record["fit_pairs"], 12)

    def test_pipeline_accepts_new_hybrids(self):
        valid = np.zeros_like(self.train)
        valid[1, 2] = valid[2, 3] = True
        test = np.zeros_like(self.train)
        test[1, 4] = test[2, 5] = True
        names = ["FeatureCombination", "TunedWeightedHybrid", "SwitchingHybrid"]
        scores = {name: np.tile(self.scores, (3, 1)) for name in names}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = (scores, self.train, valid, test, self.genres, np.array([1, 2]), {})
            with patch("task3.evaluate.prepare", return_value=prepared), patch("task3.evaluate.ARTIFACTS", root):
                rows = run(names, [0, .5], 2, 5, .05, 2020, root)
            self.assertEqual(len(rows), 15)
            self.assertEqual({row["model"] for row in rows}, set(names))

    def test_pipeline_keeps_test_labels_out_of_selection(self):
        valid = np.zeros_like(self.train)
        valid[1, 2] = valid[2, 3] = True
        test = np.zeros_like(self.train)
        test[1, 4] = test[2, 5] = True
        scores = np.tile(self.scores, (3, 1))
        selected = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                output = root / str(index)
                prepared = ({"Pop": scores}, self.train, valid, test, self.genres, np.array([1, 2]), {})
                with patch("task3.evaluate.prepare", return_value=prepared), patch("task3.evaluate.ARTIFACTS", root):
                    rows = run(["Pop"], [0, .5, 1], 2, 5, .05, 2020, output)
                self.assertEqual(len(rows), 5)
                selected.append((output / "selected.json").read_text())
                with np.load(root / "Pop-test-rankings.npz") as ranked:
                    for method in ("none", *METHODS):
                        for user, items in zip(ranked["users"], ranked[method]):
                            self.assertTrue(all(i > 0 and not (self.train | valid)[user, i] for i in items if i != -1))
                test = np.roll(test, 1, axis=1)
            self.assertEqual(selected[0], selected[1])


if __name__ == "__main__":
    unittest.main()
