import unittest
from pathlib import Path

import numpy as np

from src.retrieval.retrieve import (
    RecommendationRetriever,
)


REQUIRED_ARTIFACTS = [
    Path("models/retrieval/faiss.index"),
    Path("models/retrieval/index_metadata.json"),
    Path("data/processed/item_mapping.parquet"),
]


class TestRetrievalIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        missing = [
            str(path)
            for path in REQUIRED_ARTIFACTS
            if not path.exists()
        ]

        if missing:
            raise unittest.SkipTest(
                "Required retrieval artifacts are missing: "
                + ", ".join(missing)
            )

        cls.retriever = (
            RecommendationRetriever()
        )

    def test_real_index_returns_top_k(self):
        # The query vector must match whatever the promoted model's factor
        # dimension is, which depends on the winning hyperparameters — 64 was
        # only ever true of the first iALS champion.
        dimension = (
            self.retriever.retriever.dimension
        )

        result = (
            self.retriever
            .recommend_from_vector(
                np.ones(
                    dimension,
                    dtype=np.float32,
                ),
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

        self.assertEqual(
            result["item_idx"].nunique(),
            10,
        )

        self.assertEqual(
            result["parent_asin"].nunique(),
            10,
        )


if __name__ == "__main__":
    unittest.main()