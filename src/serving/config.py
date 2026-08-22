from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """
    Runtime configuration for the recommendation API.

    Environment variables override the defaults, which is how the
    deployment manifest configures a Cloud Run instance: the manifest's
    ``serving_env`` block is exactly this set of variables.

    ``MODEL_PATH`` points at the canonical promoted artifact rather than at
    a family-specific path. It replaced ``IALS_MODEL_PATH``, which asserted
    a model family the pipeline no longer fixes — selection may promote ALS,
    BPR or LMF, and all three are served by the same code. The old name is
    still honoured so an in-flight container configured with it keeps
    working.
    """

    model_type: str = os.getenv(
        "MODEL_TYPE",
        "als",
    )

    model_version: str = os.getenv(
        "MODEL_VERSION",
        "1",
    )

    model_path: str = os.getenv(
        "MODEL_PATH",
        os.getenv(
            "IALS_MODEL_PATH",
            "models/promoted/model.npz",
        ),
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
            "promoted_model": Path(
                self.model_path
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
