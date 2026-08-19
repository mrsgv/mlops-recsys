import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.faiss_index import (
    FaissRetriever,
)


class TestFaissRetriever(unittest.TestCase):

    def setUp(self):
        self.vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.8, 0.6, 0.0],
            ],
            dtype=np.float32,
        )

    def test_raw_inner_product(self):
        retriever = FaissRetriever(
            dimension=3,
            normalize=False,
        )

        vectors = np.array(
            [
                [3.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        retriever.add(
            vectors
        )

        ids, scores = retriever.search(
            np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            k=2,
        )

        self.assertEqual(
            ids,
            [0, 1],
        )

        self.assertAlmostEqual(
            scores[0],
            3.0,
            places=5,
        )

        self.assertAlmostEqual(
            scores[1],
            1.0,
            places=5,
        )

    def test_normalized_inner_product(self):
        retriever = FaissRetriever(
            dimension=3,
            normalize=True,
        )

        retriever.add(
            self.vectors
        )

        ids, scores = retriever.search(
            np.array(
                [2.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            k=1,
        )

        self.assertEqual(
            ids[0],
            0,
        )

        self.assertAlmostEqual(
            scores[0],
            1.0,
            places=5,
        )

    def test_zero_vectors_are_preserved(self):
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        retriever = FaissRetriever(
            dimension=3,
            normalize=False,
        )

        retriever.add(
            vectors
        )

        self.assertEqual(
            retriever.num_items,
            3,
        )

        self.assertEqual(
            retriever.zero_vector_count,
            1,
        )

    def test_dimension_mismatch(self):
        retriever = FaissRetriever(
            dimension=3,
        )

        with self.assertRaises(
            ValueError
        ):
            retriever.add(
                np.ones(
                    (2, 4),
                    dtype=np.float32,
                )
            )

    def test_empty_vectors_rejected(self):
        retriever = FaissRetriever(
            dimension=3,
        )

        with self.assertRaises(
            ValueError
        ):
            retriever.add(
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                )
            )

    def test_save_and_load_preserves_normalization_mode(
        self
    ):
        retriever = FaissRetriever(
            dimension=3,
            normalize=False,
        )

        retriever.add(
            self.vectors
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            index_path = (
                tmp / "index.faiss"
            )

            metadata_path = (
                tmp / "metadata.json"
            )

            retriever.save(
                str(index_path),
                str(metadata_path),
                {
                    "model_type": "test"
                },
            )

            loaded = (
                FaissRetriever.load(
                    str(index_path),
                    str(metadata_path),
                )
            )

            self.assertFalse(
                loaded.normalize
            )

            self.assertEqual(
                loaded.num_items,
                4,
            )

            ids, scores = loaded.search(
                np.array(
                    [2.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                k=1,
            )

            self.assertEqual(
                ids[0],
                0,
            )

            self.assertAlmostEqual(
                scores[0],
                2.0,
                places=5,
            )


if __name__ == "__main__":
    unittest.main()