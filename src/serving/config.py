from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """
    Runtime configuration for the recommendation API.

    Environment variables can override the defaults, which will
    be useful later for Docker and Cloud Run.
    """

    model_type: str = os.getenv(
        "MODEL_TYPE",
        "ials",
    )

    model_version: str = os.getenv(
        "MODEL_VERSION",
        "1",
    )

    ials_model_path: str = os.getenv(
        "IALS_MODEL_PATH",
        "models/ials/ials_model.npz",
    )

    faiss_index_path: str = os.getenv(
        "FAISS_INDEX_PATH",
        "models/retrieval/faiss.index",
    )

    faiss_metadata_path: str = os.getenv(
        "FAISS_METADATA_PATH",
        "models/retrieval/index_metadata.json",
    )

    item_mapping_path: str = os.getenv(
        "ITEM_MAPPING_PATH",
        "data/processed/item_mapping.parquet",
    )

    interactions_path: str = os.getenv(
        "INTERACTIONS_PATH",
        "data/processed/video_games.parquet",
    )

    max_recommendations: int = int(
        os.getenv(
            "MAX_RECOMMENDATIONS",
            "100",
        )
    )

    @property
    def artifact_paths(self) -> dict[str, Path]:
        return {
            "ials_model": Path(
                self.ials_model_path
            ),
            "faiss_index": Path(
                self.faiss_index_path
            ),
            "faiss_metadata": Path(
                self.faiss_metadata_path
            ),
            "item_mapping": Path(
                self.item_mapping_path
            ),
            "interactions": Path(
                self.interactions_path
            ),
        }


settings = Settings()