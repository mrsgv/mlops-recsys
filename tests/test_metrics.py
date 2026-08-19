import unittest

from src.evaluation.metrics import (
    evaluate_top_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRankingMetrics(unittest.TestCase):

    def test_perfect_single_relevant_item(self):
        recommendations = [10, 20, 30]
        relevant = {10}

        self.assertAlmostEqual(
            precision_at_k(recommendations, relevant, 3),
            1 / 3,
        )
        self.assertEqual(
            recall_at_k(recommendations, relevant, 3),
            1.0,
        )
        self.assertEqual(
            hit_rate_at_k(recommendations, relevant, 3),
            1.0,
        )
        self.assertEqual(
            ndcg_at_k(recommendations, relevant, 3),
            1.0,
        )

    def test_relevant_item_at_rank_three(self):
        recommendations = [10, 20, 30]
        relevant = {30}

        self.assertAlmostEqual(
            precision_at_k(recommendations, relevant, 3),
            1 / 3,
        )
        self.assertEqual(
            recall_at_k(recommendations, relevant, 3),
            1.0,
        )
        self.assertEqual(
            hit_rate_at_k(recommendations, relevant, 3),
            1.0,
        )
        self.assertAlmostEqual(
            ndcg_at_k(recommendations, relevant, 3),
            1 / 2.0,
        )

    def test_no_hit(self):
        recommendations = [10, 20, 30]
        relevant = {40}

        self.assertEqual(
            precision_at_k(recommendations, relevant, 3),
            0.0,
        )
        self.assertEqual(
            recall_at_k(recommendations, relevant, 3),
            0.0,
        )
        self.assertEqual(
            hit_rate_at_k(recommendations, relevant, 3),
            0.0,
        )
        self.assertEqual(
            ndcg_at_k(recommendations, relevant, 3),
            0.0,
        )

    def test_k_smaller_than_recommendation_list(self):
        recommendations = [10, 20, 30]
        relevant = {30}

        self.assertEqual(
            hit_rate_at_k(recommendations, relevant, 2),
            0.0,
        )

        self.assertEqual(
            recall_at_k(recommendations, relevant, 2),
            0.0,
        )

    def test_multiple_relevant_items(self):
        recommendations = [10, 20, 30, 40]
        relevant = {10, 30}

        self.assertEqual(
            recall_at_k(recommendations, relevant, 4),
            1.0,
        )

    def test_empty_recommendations(self):
        recommendations = []
        relevant = {10}

        self.assertEqual(
            precision_at_k(recommendations, relevant, 10),
            0.0,
        )
        self.assertEqual(
            recall_at_k(recommendations, relevant, 10),
            0.0,
        )
        self.assertEqual(
            hit_rate_at_k(recommendations, relevant, 10),
            0.0,
        )
        self.assertEqual(
            ndcg_at_k(recommendations, relevant, 10),
            0.0,
        )

    def test_missing_user_recommendations_are_zero(self):
        recommendations = {
            1: [10, 20],
        }

        ground_truth = {
            1: {10},
            2: {30},
        }

        result = evaluate_top_k(
            recommendations,
            ground_truth,
            2,
        )

        self.assertEqual(result["users_evaluated"], 2)
        self.assertEqual(result["hit_rate_at_2"], 0.5)
        self.assertEqual(result["recall_at_2"], 0.5)


if __name__ == "__main__":
    unittest.main()