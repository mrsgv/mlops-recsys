import unittest
from pathlib import Path

import numpy as np

from src.retrieval.ials_retriever import (
    IALSFaissRetriever,
)


MODEL_PATH = (
    Path("models/ials/ials_model.npz")
)

INDEX_PATH = (
    Path("models/retrieval/faiss.index")
)

INDEX_METADATA_PATH = (
    Path(
        "models/retrieval/index_metadata.json"
    )
)

MAPPING_PATH = (
    Path(
        "data/processed/item_mapping.parquet"
    )
)


class TestIALSFaissRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        required = [
            MODEL_PATH,
            INDEX_PATH,
            INDEX_METADATA_PATH,
            MAPPING_PATH,
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise unittest.SkipTest(
                "Required retrieval artifacts are missing: "
                + ", ".join(missing)
            )

        cls.retriever = (
            IALSFaissRetriever()
        )

    def test_dimensions_are_consistent(self):
        self.assertEqual(
            self.retriever.num_items,
            25612,
        )

        self.assertEqual(
            self.retriever.num_users,
            94762,
        )

        self.assertEqual(
            self.retriever.dimension,
            64,
        )

    def test_recommend_returns_top_k(self):
        result = (
            self.retriever.recommend(
                user_idx=0,
                k=10,
            )
        )

        self.assertEqual(
            len(result),
            10,
        )

        self.assertEqual(
            result["rank"].tolist(),
            list(range(1, 11)),
        )

    def test_recommendation_items_are_unique(self):
        result = (
            self.retriever.recommend(
                user_idx=0,
                k=10,
            )
        )

        self.assertEqual(
            result["item_idx"].nunique(),
            10,
        )

        self.assertEqual(
            result["parent_asin"].nunique(),
            10,
        )

    def test_scores_are_descending(self):
        result = (
            self.retriever.recommend(
                user_idx=0,
                k=10,
            )
        )

        scores = (
            result["score"]
            .to_numpy()
        )

        self.assertTrue(
            np.all(
                scores[:-1]
                >= scores[1:]
            )
        )

    def test_invalid_user_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            self.retriever.recommend(
                user_idx=self.retriever.num_users,
                k=10,
            )

    def test_recommendations_exclude_training_history(self):
        result = self.retriever.recommend(
            user_idx=0,
            k=10,
        )

        seen = self.retriever.user_history[0]

        recommended_items = set(
            result["item_idx"].tolist()
        )

        self.assertTrue(
            recommended_items.isdisjoint(
                seen
            )
        )


if __name__ == "__main__":
    unittest.main()