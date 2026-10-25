import unittest

import numpy as np

from task2.group_analysis import define_groups, fit_gate, apply_gate, group_tables, paired_ci


class GroupAnalysisTests(unittest.TestCase):
    def test_groups_exclude_padding_and_partition_catalog(self):
        train = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0],
                          [0, 1, 0, 0, 0], [0, 1, 1, 1, 0]], bool)
        groups, items, cells, _ = define_groups(train)
        self.assertTrue(all(not mask[0] for mask in [*groups.values(), *items.values()]))
        np.testing.assert_array_equal(items['head'] | items['tail'], [0, 1, 1, 1, 1])
        self.assertFalse((items['head'] & items['tail']).any())
        self.assertTrue(items['cold'][4])
        self.assertEqual(cells[0], -1)
        self.assertTrue(np.all((cells[1:] >= 0) & (cells[1:] <= 3)))

    def test_gate_specializes_and_small_cells_fall_back(self):
        valid = np.zeros((5, 4), bool)
        valid[1:, 1] = True
        first = np.array([[-1, -1], [1, 2], [1, 2], [2, 1], [2, 1]])
        second = np.array([[-1, -1], [2, 1], [2, 1], [1, 2], [1, 2]])
        rankings = {'A': first, 'B': second}
        cells = np.array([-1, 0, 0, 1, 1])
        baseline, choices, _ = fit_gate(rankings, valid, np.arange(1, 5), cells, 2, min_users=2)
        self.assertEqual(choices, {0: 'A', 1: 'B'})
        result = apply_gate(rankings, cells, baseline, choices)
        np.testing.assert_array_equal(result[1:, 0], [1, 1, 1, 1])
        _, small_choices, _ = fit_gate(rankings, valid, np.arange(1, 5), cells, 2, min_users=3)
        self.assertEqual(len(set(small_choices.values())), 1)

    def test_item_relevance_keeps_original_rank_and_zero_denominators(self):
        train = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]], bool)
        truth = np.zeros_like(train)
        truth[1, 3] = True
        ranking = np.array([[-1, -1], [2, 3], [3, 2]])
        genres = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        _, items, _, _, _ = group_tables({'A': ranking}, train, truth, genres, 2, 'A')
        cold = next(r for r in items if r['group'] == 'cold')
        self.assertEqual(cold['micro_recall'], 1)
        self.assertAlmostEqual(cold['ndcg_original_ranks_eligible'], 1 / np.log2(3))
        head = next(r for r in items if r['group'] == 'head')
        self.assertIsNone(head['micro_recall'])
        self.assertIsNone(head['slot_precision'])

    def test_paired_bootstrap_preserves_constant_difference(self):
        low, high = paired_ci([.1, .1, .1])
        self.assertAlmostEqual(low, .1)
        self.assertAlmostEqual(high, .1)


if __name__ == '__main__':
    unittest.main()
