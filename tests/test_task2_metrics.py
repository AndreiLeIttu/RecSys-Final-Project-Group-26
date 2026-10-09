import unittest
from unittest.mock import patch

import numpy as np

from task2.metrics import (
    accuracy_metrics,
    average_recommendation_popularity_at_k,
    beyond_accuracy_metrics,
    catalog_coverage_at_k,
    intra_list_diversity_at_k,
    novelty_at_k,
)
from task2.recommendations import rank_scores


class IndependentMetricTests(unittest.TestCase):
    def test_accuracy_metrics_on_manual_rankings(self):
        recommendations = [[11, 12, 13], [14], []]
        relevant = [{12, 13}, {15}, set()]
        metrics = accuracy_metrics(recommendations, relevant, k=3)
        expected_ndcg_first = (
            1 / np.log2(3) + 1 / np.log2(4)
        ) / (1 + 1 / np.log2(3))
        self.assertAlmostEqual(metrics["Precision@3"], 2 / 9)
        self.assertAlmostEqual(metrics["Recall@3"], 1 / 3)
        self.assertAlmostEqual(metrics["Hit@3"], 1 / 3)
        self.assertAlmostEqual(metrics["MRR@3"], 1 / 6)
        self.assertAlmostEqual(metrics["NDCG@3"], expected_ndcg_first / 3)

    def test_accuracy_metrics_match_recbole_for_identical_rankings(self):
        from recbole.evaluator.metrics import Hit, MRR, NDCG, Precision, Recall

        recommendations = [[1, 2, 3], [3, 1, 2]]
        relevant = [{2, 3}, {3}]
        independent = accuracy_metrics(recommendations, relevant, k=3)
        position_matrix = np.array([
            [item in truth for item in row]
            for row, truth in zip(recommendations, relevant)
        ], dtype=bool)
        relevant_counts = np.array([len(items) for items in relevant])
        config = {"metric_decimal_place": 8, "topk": [3]}
        recbole_metrics = {
            "Precision@3": (Precision(config), position_matrix, None, "precision"),
            "Recall@3": (Recall(config), position_matrix, relevant_counts, "recall"),
            "Hit@3": (Hit(config), position_matrix, None, "hit"),
            "MRR@3": (MRR(config), position_matrix, None, "mrr"),
            "NDCG@3": (NDCG(config), position_matrix, relevant_counts, "ndcg"),
        }
        with patch.object(np, "float", float, create=True):
            for name, (metric, positions, lengths, recbole_name) in recbole_metrics.items():
                values = (
                    metric.metric_info(positions)
                    if lengths is None else metric.metric_info(positions, lengths)
                )
                expected = metric.topk_result(recbole_name, values)[f"{recbole_name}@3"]
                self.assertAlmostEqual(independent[name], expected, places=7, msg=name)

    def test_fewer_than_k_items_and_empty_lists(self):
        metrics = accuracy_metrics([[7], []], [{7}, {8}], k=3)
        self.assertAlmostEqual(metrics["Precision@3"], 1 / 6)
        self.assertEqual(metrics["Recall@3"], 0.5)
        self.assertEqual(metrics["Hit@3"], 0.5)
        self.assertEqual(metrics["MRR@3"], 0.5)
        self.assertEqual(metrics["NDCG@3"], 0.5)
        self.assertEqual(accuracy_metrics([], [], k=3)["NDCG@3"], 0.0)

    def test_beyond_accuracy_definitions(self):
        recommendations = [[1, 2], [2, 3]]
        popularity = np.array([0, 3, 1, 0], dtype=float)
        features = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], dtype=float)
        self.assertAlmostEqual(catalog_coverage_at_k(recommendations, 3, 2), 1.0)
        self.assertAlmostEqual(intra_list_diversity_at_k(recommendations, features, 2), 0.5)
        self.assertAlmostEqual(novelty_at_k([[3]], popularity, 1), np.log2(7))
        self.assertAlmostEqual(average_recommendation_popularity_at_k(recommendations, popularity, 2), 1.25)
        self.assertEqual(
            set(beyond_accuracy_metrics(recommendations, popularity, features, 2)),
            {"Coverage@2", "IntraListDiversity@2", "Novelty@2", "AveragePopularity@2"},
        )

    def test_rank_scores_masks_padding_and_history(self):
        scores = np.array([[10, 0, 3, 3, 1]], dtype=float)
        seen = np.array([[0, 1, 0, 0, 0]], dtype=bool)
        np.testing.assert_array_equal(rank_scores(scores, seen, 4), [[3, 2, 4, -1]])


if __name__ == "__main__":
    unittest.main()