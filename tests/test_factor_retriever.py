"""
Tests for the factor + FAISS retrieval path.

These ran against ``models/ials/ials_model.npz`` and asserted a factor
dimension of exactly 64. Both were assumptions the pipeline is no longer
entitled to make: the promoted model may be any matrix-factorisation family,
and its dimension depends on the winning hyperparameters — BPR and LMF also
add bias columns. So the dimension assertions now check internal consistency
between the artifact, the index and the mapping, which is the property that
actually has to hold, rather than a literal that happened to be true of one
model.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.factor_retriever import (
    FactorFaissRetriever,
)


MODEL_PATH = Path(
    "models/promoted/model.npz"
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

INTERACTIONS_PATH = (
    Path(
        "data/processed/video_games.parquet"
    )
)


class TestFactorFaissRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        required = [
            MODEL_PATH,
            INDEX_PATH,
            INDEX_METADATA_PATH,
            MAPPING_PATH,
            INTERACTIONS_PATH,
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
            FactorFaissRetriever()
        )

    def test_dimensions_are_consistent(self):
        # The index, the item factors and the mapping must all agree on how
        # many items exist and how wide a factor row is. A mismatch means
        # the index was built from a different model than the one loaded.
        retriever = self.retriever

        self.assertEqual(
            retriever.num_items,
            retriever.faiss_index.num_items,
        )

        self.assertEqual(
            retriever.num_items,
            len(retriever.item_mapping),
        )

        self.assertEqual(
            retriever.dimension,
            retriever.faiss_index.dimension,
        )

        self.assertEqual(
            retriever.dimension,
            retriever.item_factors.shape[1],
        )

        self.assertGreater(
            retriever.num_users,
            0,
        )

    def test_index_is_not_normalized(self):
        # Every supported family ranks by raw inner product. Normalising
        # would discard factor magnitude and, for BPR and LMF, the bias
        # column — silently changing the ranking.
        self.assertFalse(
            self.retriever.faiss_index.normalize
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


class TestFactorArtifactGuards(unittest.TestCase):
    """
    A model without factors must be rejected at load time.

    Selection marks neighbourhood models undeployable, but that is a policy
    check. This is the mechanical one: if a factorless artifact ever reaches
    the serving path, startup must fail with a clear message rather than the
    service coming up and failing on first request.
    """

    def setUp(self):
        self.directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.directory.name
        )

    def tearDown(self):
        self.directory.cleanup()

    def _write_index(
        self,
        dimension: int,
        num_items: int,
    ) -> tuple[str, str]:
        from src.retrieval.faiss_index import (
            FaissRetriever,
        )

        retriever = FaissRetriever(
            dimension=dimension,
            normalize=False,
        )

        retriever.add(
            np.ones(
                (num_items, dimension),
                dtype=np.float32,
            )
        )

        index_path = (
            self.root / "faiss.index"
        )

        metadata_path = (
            self.root / "metadata.json"
        )

        retriever.save(
            path=str(index_path),
            metadata_path=str(metadata_path),
            metadata={},
        )

        return (
            str(index_path),
            str(metadata_path),
        )

    def test_artifact_without_factors_is_rejected(self):
        model_path = (
            self.root / "knn.npz"
        )

        # A neighbourhood model's artifact holds a similarity matrix, not
        # factor matrices.
        np.savez(
            model_path,
            similarity=np.eye(
                4,
                dtype=np.float32,
            ),
        )

        index_path, metadata_path = (
            self._write_index(
                dimension=4,
                num_items=4,
            )
        )

        with self.assertRaises(
            ValueError
        ) as raised:
            FactorFaissRetriever(
                model_path=str(model_path),
                faiss_index_path=index_path,
                faiss_metadata_path=(
                    metadata_path
                ),
                item_mapping_path=str(
                    self.root / "absent.parquet"
                ),
                interactions_path=str(
                    self.root / "absent.parquet"
                ),
            )

        self.assertIn(
            "user_factors",
            str(raised.exception),
        )

    def test_missing_artifact_is_reported(self):
        index_path, metadata_path = (
            self._write_index(
                dimension=4,
                num_items=4,
            )
        )

        with self.assertRaises(
            FileNotFoundError
        ):
            FactorFaissRetriever(
                model_path=str(
                    self.root / "absent.npz"
                ),
                faiss_index_path=index_path,
                faiss_metadata_path=(
                    metadata_path
                ),
                item_mapping_path=str(
                    self.root / "absent.parquet"
                ),
                interactions_path=str(
                    self.root / "absent.parquet"
                ),
            )

    def test_index_dimension_mismatch_is_reported(self):
        # The index and the model must have been built from the same fit.
        # Promoting a new model without rebuilding the index is exactly the
        # failure this catches.
        import pandas as pd

        model_path = (
            self.root / "model.npz"
        )

        np.savez(
            model_path,
            user_factors=np.ones(
                (3, 8),
                dtype=np.float32,
            ),
            item_factors=np.ones(
                (4, 8),
                dtype=np.float32,
            ),
        )

        index_path, metadata_path = (
            self._write_index(
                dimension=4,
                num_items=4,
            )
        )

        mapping_path = (
            self.root / "mapping.parquet"
        )

        pd.DataFrame(
            {
                "item_idx": range(4),
                "parent_asin": [
                    f"B{index:09d}"
                    for index in range(4)
                ],
            }
        ).to_parquet(mapping_path)

        interactions_path = (
            self.root
            / "interactions.parquet"
        )

        pd.DataFrame(
            {
                "user_idx": [0, 1, 2],
                "item_idx": [0, 1, 2],
            }
        ).to_parquet(interactions_path)

        with self.assertRaises(
            ValueError
        ) as raised:
            FactorFaissRetriever(
                model_path=str(model_path),
                faiss_index_path=index_path,
                faiss_metadata_path=(
                    metadata_path
                ),
                item_mapping_path=str(
                    mapping_path
                ),
                interactions_path=str(
                    interactions_path
                ),
            )

        self.assertIn(
            "Rebuild the index",
            str(raised.exception),
        )


if __name__ == "__main__":
    unittest.main()
