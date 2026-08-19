import unittest

import pandas as pd

from src.models.ials_baseline import IALSRecommender


class TestIALS(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "user_idx": [
                    0, 0, 0,
                    1, 1, 1,
                    2, 2,
                    3, 3,
                ],
                "item_idx": [
                    0, 1, 2,
                    0, 2, 3,
                    1, 3,
                    0, 4,
                ],
            }
        )

    def test_fit_and_recommend(self):
        model = IALSRecommender(
            factors=4,
            regularization=0.1,
            iterations=2,
            alpha=1.0,
            random_state=42,
        )

        training_time = model.fit(
            self.df,
            num_users=4,
            num_items=5,
        )

        self.assertGreaterEqual(
            training_time,
            0.0,
        )

        recommendations = (
            model.recommend_users(
                [0, 1, 2],
                k=2,
            )
        )

        self.assertEqual(
            len(recommendations),
            3,
        )

        for user_idx, items in recommendations.items():
            self.assertEqual(
                len(items),
                2,
            )

            self.assertTrue(
                all(
                    isinstance(item, int)
                    for item in items
                )
            )
    def test_factor_shapes(self):
        model = IALSRecommender(
            factors=4,
            regularization=0.1,
            iterations=2,
            alpha=1.0,
            random_state=42,
        )

        model.fit(
            self.df,
            num_users=4,
            num_items=5,
        )

        self.assertEqual(
            model.model.user_factors.shape,
            (4, 4),
        )

        self.assertEqual(
            model.model.item_factors.shape,
            (5, 4),
        )


if __name__ == "__main__":
    unittest.main()