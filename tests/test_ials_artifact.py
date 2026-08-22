import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.ials_baseline import (
    IALSRecommender,
    build_user_item_matrix,
    load_ials_model,
)


class TestBuildUserItemMatrix(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "user_idx": [0, 0, 1, 2],
                "item_idx": [0, 2, 1, 3],
            }
        )

    def test_shape_and_binary_values(self):
        matrix = build_user_item_matrix(
            self.df,
            num_users=3,
            num_items=4,
        )

        self.assertEqual(
            matrix.shape,
            (3, 4),
        )

        self.assertEqual(
            matrix.nnz,
            4,
        )

        # Ratings must not leak in: implicit feedback is binary.
        self.assertTrue(
            np.array_equal(
                np.unique(matrix.data),
                np.array(
                    [1.0],
                    dtype=np.float32,
                ),
            )
        )

    def test_ignores_rating_column(self):
        rated = self.df.copy()
        rated["rating"] = [5.0, 1.0, 3.0, 4.0]

        matrix = build_user_item_matrix(
            rated,
            num_users=3,
            num_items=4,
        )

        self.assertTrue(
            np.array_equal(
                np.unique(matrix.data),
                np.array(
                    [1.0],
                    dtype=np.float32,
                ),
            )
        )

    def test_rejects_empty_frame(self):
        with self.assertRaises(ValueError):
            build_user_item_matrix(
                self.df.head(0),
                num_users=3,
                num_items=4,
            )

    def test_rejects_missing_indices(self):
        broken = self.df.copy()
        broken.loc[0, "item_idx"] = None

        with self.assertRaises(ValueError):
            build_user_item_matrix(
                broken,
                num_users=3,
                num_items=4,
            )


class TestIALSFromArtifact(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.DataFrame(
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

        cls.num_users = 4
        cls.num_items = 5

        cls.trained = IALSRecommender(
            factors=4,
            regularization=0.1,
            iterations=2,
            alpha=1.0,
            random_state=42,
        )

        cls.trained.fit(
            cls.train_df,
            num_users=cls.num_users,
            num_items=cls.num_items,
        )

        cls.directory = (
            tempfile.TemporaryDirectory()
        )

        cls.artifact_path = (
            Path(cls.directory.name)
            / "ials_model.npz"
        )

        cls.trained.model.save(
            str(cls.artifact_path)
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def load(self) -> IALSRecommender:
        return IALSRecommender.from_artifact(
            self.artifact_path,
            train_df=self.train_df,
            num_users=self.num_users,
            num_items=self.num_items,
        )

    def test_hyperparameters_come_from_artifact(self):
        loaded = self.load()

        self.assertEqual(
            loaded.factors,
            4,
        )

        self.assertAlmostEqual(
            loaded.regularization,
            0.1,
        )

        self.assertEqual(
            loaded.iterations,
            2,
        )

        self.assertAlmostEqual(
            loaded.alpha,
            1.0,
        )

    def test_training_matrix_is_rebuilt(self):
        loaded = self.load()

        self.assertEqual(
            loaded.user_item_matrix.shape,
            (
                self.num_users,
                self.num_items,
            ),
        )

    def test_factors_round_trip(self):
        loaded = self.load()

        self.assertTrue(
            np.allclose(
                loaded.model.user_factors,
                self.trained.model.user_factors,
            )
        )

        self.assertTrue(
            np.allclose(
                loaded.model.item_factors,
                self.trained.model.item_factors,
            )
        )

    def test_recommendations_match_trained_model(self):
        loaded = self.load()

        users = [0, 1, 2, 3]

        self.assertEqual(
            loaded.recommend_users(
                users,
                k=2,
            ),
            self.trained.recommend_users(
                users,
                k=2,
            ),
        )

    def test_missing_artifact_raises(self):
        with self.assertRaises(
            FileNotFoundError
        ):
            load_ials_model(
                Path(self.directory.name)
                / "does_not_exist.npz"
            )


if __name__ == "__main__":
    unittest.main()
