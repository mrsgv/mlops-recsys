from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.retrieval.base import Retriever


class FaissRetriever(Retriever):
    """
    Model-agnostic FAISS retrieval wrapper.

    Current implementation:
        - FAISS IndexFlatIP
        - exact inner-product search
        - item_idx == FAISS internal row ID

    Vectors are L2-normalized before insertion/search, so inner
    product corresponds to cosine similarity.

    Zero vectors are preserved rather than dropped. This keeps the
    invariant that FAISS row ID == item_idx.
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "inner_product",
    ) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        if metric != "inner_product":
            raise ValueError(
                "Only inner_product is currently supported."
            )

        self.dimension = dimension
        self.metric = metric

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.num_items = 0
        self.zero_vector_count = 0

    @staticmethod
    def _validate_vectors(
        vectors: np.ndarray,
        dimension: int | None = None,
    ) -> np.ndarray:
        """Validate and convert vectors to float32."""
        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "vectors must be a 2D array."
            )

        if vectors.shape[0] == 0:
            raise ValueError(
                "vectors must contain at least one vector."
            )

        if dimension is not None and (
            vectors.shape[1] != dimension
        ):
            raise ValueError(
                "Vector dimension mismatch: "
                f"expected {dimension}, "
                f"got {vectors.shape[1]}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "vectors contain NaN or infinite values."
            )

        return vectors

    @staticmethod
    def normalize_vectors(
        vectors: np.ndarray,
    ) -> np.ndarray:
        """
        L2-normalize vectors.

        Non-zero vectors become unit vectors.
        Zero vectors remain zero vectors.
        """
        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        normalized = np.zeros_like(
            vectors,
            dtype=np.float32,
        )

        nonzero_mask = (
            norms[:, 0] > 0.0
        )

        normalized[nonzero_mask] = (
            vectors[nonzero_mask]
            / norms[nonzero_mask]
        )

        return normalized

    def add(
        self,
        vectors: np.ndarray,
        normalize: bool = True,
    ) -> None:
        """
        Add vectors to the FAISS index.

        The row ordering must correspond to item_idx:
            row 0 -> item_idx 0
            row 1 -> item_idx 1
            ...
        """
        vectors = self._validate_vectors(
            vectors,
            self.dimension,
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
        )

        self.zero_vector_count = int(
            np.count_nonzero(
                norms == 0.0
            )
        )

        if normalize:
            vectors = self.normalize_vectors(
                vectors
            )

        self.index.add(vectors)

        self.num_items = int(
            self.index.ntotal
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> tuple[list[int], list[float]]:
        """
        Search the index for Top-K nearest items.
        """
        if self.num_items == 0:
            raise RuntimeError(
                "FAISS index is empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        k = min(
            k,
            self.num_items,
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.ndim == 1:
            query_vector = (
                query_vector.reshape(1, -1)
            )

        query_vector = self._validate_vectors(
            query_vector,
            self.dimension,
        )

        # Query vectors also need normalization.
        query_vector = (
            self.normalize_vectors(
                query_vector
            )
        )

        # A zero query vector has no meaningful direction.
        if np.all(
            np.linalg.norm(
                query_vector,
                axis=1,
            ) == 0.0
        ):
            raise ValueError(
                "query_vector must not be a zero vector."
            )

        scores, indices = (
            self.index.search(
                query_vector,
                k,
            )
        )

        return (
            [
                int(item_id)
                for item_id in indices[0]
            ],
            [
                float(score)
                for score in scores[0]
            ],
        )

    def save(
        self,
        path: str,
        metadata_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save FAISS index and optional metadata."""
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path),
        )

        if metadata_path is None:
            return

        metadata_path = Path(
            metadata_path
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = metadata or {}

        payload = {
            **metadata,
            "dimension": self.dimension,
            "metric": self.metric,
            "num_items": self.num_items,
            "index_type": "IndexFlatIP",
            "normalized_vectors": True,
            "zero_vector_count": (
                self.zero_vector_count
            ),
        }

        metadata_path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str,
        metadata_path: str | None = None,
    ) -> "FaissRetriever":
        """Load an existing FAISS index."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {path}"
            )

        index = faiss.read_index(
            str(path)
        )

        metadata: dict[str, Any] = {}

        if metadata_path is not None:
            metadata_file = Path(
                metadata_path
            )

            if metadata_file.exists():
                metadata = json.loads(
                    metadata_file.read_text(
                        encoding="utf-8"
                    )
                )

        metric = metadata.get(
            "metric",
            "inner_product",
        )

        if metric != "inner_product":
            raise ValueError(
                "Unsupported FAISS metric."
            )

        retriever = cls(
            dimension=index.d,
            metric=metric,
        )

        retriever.index = index

        retriever.num_items = int(
            index.ntotal
        )

        retriever.zero_vector_count = int(
            metadata.get(
                "zero_vector_count",
                0,
            )
        )

        return retriever