from __future__ import annotations

import numpy as np
import pandas as pd

from src.retrieval.faiss_index import (
    FaissRetriever,
)


INDEX_PATH = (
    "models/retrieval/faiss.index"
)

METADATA_PATH = (
    "models/retrieval/index_metadata.json"
)

ITEM_MAPPING_PATH = (
    "data/processed/item_mapping.parquet"
)


class RecommendationRetriever:
    """
    High-level retrieval service.

    Converts FAISS item_idx results into the original
    parent_asin identifiers.
    """

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        metadata_path: str = METADATA_PATH,
        item_mapping_path: str = ITEM_MAPPING_PATH,
    ) -> None:
        self.retriever = (
            FaissRetriever.load(
                index_path,
                metadata_path,
            )
        )

        self.item_mapping = (
            pd.read_parquet(
                item_mapping_path
            )
            .sort_values("item_idx")
            .reset_index(drop=True)
        )

        if len(
            self.item_mapping
        ) != self.retriever.num_items:
            raise ValueError(
                "Item mapping and FAISS index contain "
                "different numbers of items."
            )

        expected_item_ids = list(
            range(
                len(self.item_mapping)
            )
        )

        actual_item_ids = (
            self.item_mapping["item_idx"]
            .tolist()
        )

        if actual_item_ids != expected_item_ids:
            raise ValueError(
                "Item mapping must contain contiguous "
                "item_idx values from 0."
            )

    def recommend_from_vector(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Retrieve Top-K recommendations from a query vector.
        """
        item_ids, scores = (
            self.retriever.search(
                query_vector,
                k,
            )
        )

        result = (
            self.item_mapping
            .iloc[item_ids]
            [
                [
                    "item_idx",
                    "parent_asin",
                ]
            ]
            .copy()
        )

        result["score"] = scores

        result["rank"] = range(
            1,
            len(result) + 1,
        )

        return result[
            [
                "rank",
                "item_idx",
                "parent_asin",
                "score",
            ]
        ].reset_index(drop=True)