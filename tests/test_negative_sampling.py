import unittest

import numpy as np
import pandas as pd

from src.models.negative_sampling import NegativeSampler


class TestNegativeSampler(unittest.TestCase):

    def setUp(self):
        self.train_df = pd.DataFrame(
            {
                "user_idx": [
                    0, 0, 0,
                    1, 1,
                    2,
                ],
                "item_idx": [
                    0, 1, 2,
                    0, 3,
                    1,
                ],
            }
        )

    def test_shape(self):
        sampler = NegativeSampler(
            num_items=6,
            num_negatives=5,
            seed=42,
        )

        negatives = sampler.sample_for_interactions(
            self.train_df
        )

        self.assertEqual(
            negatives.shape,
            (len(self.train_df), 5),
        )

    def test_negatives_are_valid_items(self):
        sampler = NegativeSampler(
            num_items=6,
            num_negatives=5,
            seed=42,
        )

        negatives = sampler.sample_for_interactions(
            self.train_df
        )

        self.assertTrue(
            np.all(negatives >= 0)
        )

        self.assertTrue(
            np.all(negatives < 6)
        )

    def test_negatives_not_in_user_history(self):
        sampler = NegativeSampler(
            num_items=6,
            num_negatives=5,
            seed=42,
        )

        negatives = sampler.sample_for_interactions(
            self.train_df
        )

        histories = (
            self.train_df
            .groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

        for row_idx, user_idx in enumerate(
            self.train_df["user_idx"]
        ):
            user_history = histories[
                int(user_idx)
            ]

            for item_idx in negatives[row_idx]:
                self.assertNotIn(
                    int(item_idx),
                    user_history,
                )

    def test_deterministic_with_same_seed(self):
        sampler_1 = NegativeSampler(
            num_items=6,
            num_negatives=3,
            seed=42,
        )

        sampler_2 = NegativeSampler(
            num_items=6,
            num_negatives=3,
            seed=42,
        )

        negatives_1 = (
            sampler_1.sample_for_interactions(
                self.train_df
            )
        )

        negatives_2 = (
            sampler_2.sample_for_interactions(
                self.train_df
            )
        )

        np.testing.assert_array_equal(
            negatives_1,
            negatives_2,
        )

    def test_different_seed_can_change_samples(self):
        sampler_1 = NegativeSampler(
            num_items=6,
            num_negatives=3,
            seed=42,
        )

        sampler_2 = NegativeSampler(
            num_items=6,
            num_negatives=3,
            seed=43,
        )

        negatives_1 = (
            sampler_1.sample_for_interactions(
                self.train_df
            )
        )

        negatives_2 = (
            sampler_2.sample_for_interactions(
                self.train_df
            )
        )

        self.assertFalse(
            np.array_equal(
                negatives_1,
                negatives_2,
            )
        )

    def test_user_with_all_items_raises_error(self):
        train_df = pd.DataFrame(
            {
                "user_idx": [0, 0, 0],
                "item_idx": [0, 1, 2],
            }
        )

        sampler = NegativeSampler(
            num_items=3,
            num_negatives=1,
            seed=42,
        )

        with self.assertRaises(ValueError):
            sampler.sample_for_interactions(
                train_df
            )

    def test_required_columns_are_validated(self):
        bad_df = pd.DataFrame(
            {
                "user_idx": [0, 1],
            }
        )

        sampler = NegativeSampler(
            num_items=5,
            num_negatives=2,
            seed=42,
        )

        with self.assertRaises(ValueError):
            sampler.sample_for_interactions(
                bad_df
            )


if __name__ == "__main__":
    unittest.main()