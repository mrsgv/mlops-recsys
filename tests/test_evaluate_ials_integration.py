"""
Integration tests for iALS artifact evaluation.

These run the real pipeline path — chronological split, artifact load,
FAISS-free recommendation, ranking metrics — on a small synthetic dataset,
because the Amazon dataset is DVC-tracked and absent in CI.

The central assertion is reproducibility: the model reloaded from disk must
produce exactly the metrics the in-memory model produced. That is the
property the pipeline's evaluate step exists to prove, and the property that
makes promoting an artifact safe.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.evaluate_ials import (
    build_ground_truth,
    compare_with_training_metrics,
    evaluate,
)
from src.evaluation.metrics import evaluate_top_k
from src.evaluation.split import (
    chronological_train_test_split,
)
from src.models.ials_baseline import IALSRecommender


NUM_USERS = 200
NUM_ITEMS = 80
INTERACTIONS_PER_USER = 6
TOP_K = 10


def make_interactions() -> pd.DataFrame:
    """
    Build a synthetic interaction dataset.

    Every user gets enough history to survive leave-one-out, and the first
    rows are pinned so that the item space is contiguous from zero — the
    same contract validate_data enforces on real data.
    """
    generator = np.random.default_rng(7)

    rows = []
    timestamp = 1_600_000_000

    for user_idx in range(NUM_USERS):
        items = generator.choice(
            NUM_ITEMS,
            size=INTERACTIONS_PER_USER,
            replace=False,
        )

        for item_idx in items:
            timestamp += 1

            rows.append(
                {
                    "user_idx": user_idx,
                    "item_idx": int(item_idx),
                    "rating": float(
                        generator.integers(1, 6)
                    ),
                    "timestamp": timestamp,
                }
            )

    df = pd.DataFrame(rows)

    df.loc[
        df.index[:NUM_ITEMS],
        "item_idx",
    ] = list(range(NUM_ITEMS))

    return df


class TestEvaluateArtifact(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = (
            tempfile.TemporaryDirectory()
        )

        root = Path(cls.directory.name)

        cls.df = make_interactions()

        cls.data_path = (
            root / "video_games.parquet"
        )

        cls.df.to_parquet(
            cls.data_path,
            index=False,
        )

        cls.train_df, cls.test_df = (
            chronological_train_test_split(
                cls.df
            )
        )

        cls.trained = IALSRecommender(
            factors=8,
            regularization=0.1,
            iterations=3,
            alpha=1.0,
            random_state=42,
        )

        cls.trained.fit(
            cls.train_df,
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
        )

        cls.artifact_path = (
            root / "ials_model.npz"
        )

        cls.trained.model.save(
            str(cls.artifact_path)
        )

        cls.metrics = evaluate(
            data_path=str(cls.data_path),
            model_path=cls.artifact_path,
            top_k=TOP_K,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_reports_every_ranking_metric(self):
        for name in (
            f"precision_at_{TOP_K}",
            f"recall_at_{TOP_K}",
            f"hit_rate_at_{TOP_K}",
            f"ndcg_at_{TOP_K}",
        ):
            self.assertIn(
                name,
                self.metrics,
            )

            self.assertGreaterEqual(
                self.metrics[name],
                0.0,
            )

            self.assertLessEqual(
                self.metrics[name],
                1.0,
            )

    def test_evaluates_every_held_out_user(self):
        self.assertEqual(
            self.metrics["users_evaluated"],
            self.test_df["user_idx"].nunique(),
        )

    def test_artifact_reproduces_in_memory_model(self):
        # What the pipeline promotes is the file on disk, so the file must
        # score identically to the model that was just trained.
        in_memory = evaluate_top_k(
            recommendations=(
                self.trained.recommend_users(
                    self.test_df["user_idx"]
                    .drop_duplicates()
                    .astype(int)
                    .tolist(),
                    k=TOP_K,
                )
            ),
            ground_truth=build_ground_truth(
                self.test_df
            ),
            k=TOP_K,
        )

        self.assertEqual(
            self.metrics,
            in_memory,
        )

    def test_comparison_reports_no_drift(self):
        training_metrics = {
            name: float(value)
            for name, value in self.metrics.items()
            if name != "users_evaluated"
        }

        self.assertEqual(
            compare_with_training_metrics(
                self.metrics,
                training_metrics,
            ),
            {},
        )

    def test_recommendations_exclude_seen_items(self):
        # Seen-item filtering is what stops the API recommending something
        # the user already interacted with.
        seen = (
            self.train_df
            .groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

        recommendations = (
            self.trained.recommend_users(
                [0, 1, 2, 3, 4],
                k=TOP_K,
            )
        )

        for user_idx, items in recommendations.items():
            self.assertTrue(
                set(items).isdisjoint(
                    seen[user_idx]
                ),
                f"user {user_idx} was recommended a seen item",
            )

    def test_missing_artifact_fails_loudly(self):
        with self.assertRaises(
            FileNotFoundError
        ):
            evaluate(
                data_path=str(self.data_path),
                model_path=(
                    Path(self.directory.name)
                    / "absent.npz"
                ),
                top_k=TOP_K,
            )


if __name__ == "__main__":
    unittest.main()
