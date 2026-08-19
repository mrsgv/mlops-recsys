import unittest
from pathlib import Path
import tempfile

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

    def test_add_and_search(self):
        retriever = FaissRetriever(
            dimension=3
        )

        retriever.add(
            self.vectors
        )

        item_ids, scores = (
            retriever.search(
                np.array(
                    [1.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                k=2,
            )
        )

        self.assertEqual(
            len(item_ids),
            2,
        )

        self.assertEqual(
            item_ids[0],
            0,
        )

        self.assertGreaterEqual(
            scores[0],
            scores[1],
        )

    def test_vectors_are_normalized(self):
        retriever = FaissRetriever(
            dimension=3
        )

        retriever.add(
            self.vectors
        )

        item_ids, scores = (
            retriever.search(
                np.array(
                    [2.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                k=1,
            )
        )

        self.assertEqual(
            item_ids[0],
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
            dimension=3
        )

        retriever.add(vectors)

        self.assertEqual(
            retriever.num_items,
            3,
        )

        self.assertEqual(
            retriever.zero_vector_count,
            1,
        )

    def test_zero_query_vector_rejected(self):
        retriever = FaissRetriever(
            dimension=3
        )

        retriever.add(
            self.vectors
        )

        with self.assertRaises(ValueError):
            retriever.search(
                np.zeros(
                    3,
                    dtype=np.float32,
                ),
                k=1,
            )

    def test_save_and_load(self):
        retriever = FaissRetriever(
            dimension=3
        )

        vectors = np.vstack(
            [
                self.vectors,
                np.zeros(
                    (1, 3),
                    dtype=np.float32,
                ),
            ]
        )

        retriever.add(vectors)

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

            item_ids, scores = (
                loaded.search(
                    np.array(
                        [1.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    k=2,
                )
            )

            self.assertEqual(
                item_ids[0],
                0,
            )

            self.assertEqual(
                loaded.num_items,
                5,
            )

            self.assertEqual(
                loaded.zero_vector_count,
                1,
            )

    def test_dimension_mismatch(self):
        retriever = FaissRetriever(
            dimension=3
        )

        with self.assertRaises(ValueError):
            retriever.add(
                np.ones(
                    (2, 4),
                    dtype=np.float32,
                )
            )

    def test_empty_vectors_rejected(self):
        retriever = FaissRetriever(
            dimension=3
        )

        with self.assertRaises(ValueError):
            retriever.add(
                np.empty(
                    (0, 3),
                    dtype=np.float32,
                )
            )


if __name__ == "__main__":
    unittest.main()