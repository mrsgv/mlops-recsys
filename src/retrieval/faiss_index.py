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

    FAISS internal ID == item_idx.

    The similarity semantics are controlled by `normalize`:

    normalize=False:
        raw inner product

    normalize=True:
        cosine similarity implemented as inner product
        over L2-normalized vectors

    This distinction matters because different models may
    define their ranking score differently.
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "inner_product",
        normalize: bool = False,
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
        self.normalize = normalize

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
        """Validate vectors and convert them to float32."""
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

        Zero vectors are preserved as zero vectors.
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

        nonzero = norms[:, 0] > 0.0

        normalized[nonzero] = (
            vectors[nonzero]
            / norms[nonzero]
        )

        return normalized

    def add(
        self,
        vectors: np.ndarray,
    ) -> None:
        """
        Add vectors to the FAISS index.

        The caller is responsible for ensuring row order matches
        the intended item_idx mapping.
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

        if self.normalize:
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

        if self.normalize:
            query_vector = self.normalize_vectors(
                query_vector
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
        """Save the FAISS index and optional metadata."""
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

        payload = {
            **(metadata or {}),
            "dimension": self.dimension,
            "metric": self.metric,
            "num_items": self.num_items,
            "index_type": "IndexFlatIP",
            "normalized_vectors": self.normalize,
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

        normalize = bool(
            metadata.get(
                "normalized_vectors",
                False,
            )
        )

        retriever = cls(
            dimension=index.d,
            metric=metric,
            normalize=normalize,
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