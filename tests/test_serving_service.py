import unittest
import sys
from pathlib import Path

import pandas as pd

# # Ensure the source directory is available when this test is run directly.
# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.serving.service import (
    RecommendationService,
)


class FakeRetriever:

    def __init__(self):
        self.num_users = 3
        self.num_items = 5
        self.dimension = 2

        class FakeFaiss:
            normalize = False

        self.faiss_index = FakeFaiss()

    def recommend(
        self,
        user_idx: int,
        k: int,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "rank": 1,
                    "item_idx": 3,
                    "parent_asin": "TEST3",
                    "score": 0.9,
                },
                {
                    "rank": 2,
                    "item_idx": 4,
                    "parent_asin": "TEST4",
                    "score": 0.8,
                },
            ][:k]
        )


class TestRecommendationService(
    unittest.TestCase
):

    def test_model_info(self):
        service = RecommendationService(
            retriever=FakeRetriever(),
        )

        # The default family is "als" rather than "ials": the service serves
        # whichever factor model was promoted, and MODEL_TYPE comes from the
        # deployment manifest in a real deployment.
        self.assertEqual(
            service.model_info.model_type,
            "als",
        )

        self.assertEqual(
            service.model_info.num_users,
            3,
        )

        self.assertEqual(
            service.model_info.num_items,
            5,
        )

    def test_recommend(self):
        service = RecommendationService(
            retriever=FakeRetriever(),
        )

        result = service.recommend(
            user_idx=0,
            k=2,
        )

        self.assertEqual(
            len(result),
            2,
        )

        self.assertEqual(
            result.iloc[0]["item_idx"],
            3,
        )


if __name__ == "__main__":
    unittest.main()