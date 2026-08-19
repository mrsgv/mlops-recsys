import unittest

import pandas as pd

from src.evaluation.split import (
    chronological_train_test_split,
    validate_split,
)


class TestChronologicalSplit(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "user_idx": [1, 1, 1, 2, 2, 3],
                "item_idx": [10, 20, 30, 40, 50, 60],
                "rating": [5, 4, 5, 3, 5, 4],
                "timestamp": [100, 200, 300, 100, 200, 100],
            }
        )

    def test_latest_interaction_is_test(self):
        train, test = chronological_train_test_split(self.df)

        user_1_test = test[test["user_idx"] == 1]
        self.assertEqual(len(user_1_test), 1)
        self.assertEqual(user_1_test.iloc[0]["item_idx"], 30)

    def test_users_with_one_interaction_are_excluded(self):
        train, test = chronological_train_test_split(self.df)

        self.assertNotIn(3, train["user_idx"].values)
        self.assertNotIn(3, test["user_idx"].values)

    def test_each_test_user_has_training_history(self):
        train, test = chronological_train_test_split(self.df)

        validate_split(train, test)

        train_users = set(train["user_idx"])
        test_users = set(test["user_idx"])

        self.assertTrue(test_users.issubset(train_users))

    def test_deterministic_split(self):
        train_1, test_1 = chronological_train_test_split(self.df)
        train_2, test_2 = chronological_train_test_split(self.df)

        pd.testing.assert_frame_equal(train_1, train_2)
        pd.testing.assert_frame_equal(test_1, test_2)

    def test_missing_columns_raise_error(self):
        bad_df = self.df.drop(columns=["rating"])

        with self.assertRaises(ValueError):
            chronological_train_test_split(bad_df)

    def test_empty_dataframe_raises_error(self):
        empty_df = self.df.iloc[0:0]

        with self.assertRaises(ValueError):
            chronological_train_test_split(empty_df)


if __name__ == "__main__":
    unittest.main()