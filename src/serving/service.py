from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.retrieval.factor_retriever import (
    FactorFaissRetriever,
)


@dataclass(frozen=True)
class ModelInfo:
    model_type: str
    model_version: str
    retriever: str
    num_users: int
    num_items: int
    embedding_dimension: int
    faiss_index_type: str
    normalization: bool


class RecommendationService:
    """
    Application-level recommendation service.

    HTTP/API code should interact with this service rather than
    directly calling the underlying retriever.
    """

    def __init__(
        self,
        retriever: FactorFaissRetriever,
        model_type: str = "als",
        model_version: str = "1",
    ) -> None:
        self.retriever = retriever

        faiss_metadata = (
            retriever.faiss_index
        )

        self.model_info = ModelInfo(
            model_type=model_type,
            model_version=model_version,
            retriever="faiss",
            num_users=retriever.num_users,
            num_items=retriever.num_items,
            embedding_dimension=retriever.dimension,
            faiss_index_type="IndexFlatIP",
            normalization=(
                faiss_metadata.normalize
            ),
        )

    def recommend(
        self,
        user_idx: int,
        k: int,
    ) -> pd.DataFrame:
        """Generate Top-K recommendations."""
        return self.retriever.recommend(
            user_idx=user_idx,
            k=k,
        )