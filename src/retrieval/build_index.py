from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.faiss_index import (
    FaissRetriever,
)


IALS_MODEL_PATH = (
    "models/ials/ials_model.npz"
)

ITEM_MAPPING_PATH = (
    "data/processed/item_mapping.parquet"
)

INDEX_DIR = Path(
    "models/retrieval"
)

INDEX_PATH = (
    INDEX_DIR / "faiss.index"
)

METADATA_PATH = (
    INDEX_DIR / "index_metadata.json"
)


def load_ials_item_factors(
    path: str,
) -> np.ndarray:
    """
    Load item factors from the iALS artifact.

    Expected shape:
        (num_items, factors)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"iALS model not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if "item_factors" not in data:
            raise ValueError(
                "iALS artifact does not contain "
                "'item_factors'."
            )

        item_factors = data[
            "item_factors"
        ]

    item_factors = np.asarray(
        item_factors,
        dtype=np.float32,
    )

    if item_factors.ndim != 2:
        raise ValueError(
            "item_factors must be a 2D matrix."
        )

    if len(item_factors) == 0:
        raise ValueError(
            "item_factors is empty."
        )

    if not np.isfinite(
        item_factors
    ).all():
        raise ValueError(
            "item_factors contain NaN or infinite values."
        )

    return item_factors


def load_and_validate_mapping(
    path: str,
    num_items: int,
) -> pd.DataFrame:
    """Load and validate item_idx -> parent_asin mapping."""
    mapping = (
        pd.read_parquet(path)
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    required_columns = {
        "item_idx",
        "parent_asin",
    }

    missing = (
        required_columns
        - set(mapping.columns)
    )

    if missing:
        raise ValueError(
            "Item mapping is missing required columns: "
            f"{sorted(missing)}"
        )

    if len(mapping) != num_items:
        raise ValueError(
            "Number of item factors does not match "
            "number of item mappings: "
            f"{num_items} vs {len(mapping)}."
        )

    expected_item_ids = list(
        range(num_items)
    )

    actual_item_ids = (
        mapping["item_idx"]
        .tolist()
    )

    if actual_item_ids != expected_item_ids:
        raise ValueError(
            "item_idx values must be contiguous from "
            "0 to num_items - 1."
        )

    if mapping["parent_asin"].duplicated().any():
        raise ValueError(
            "Duplicate parent_asin values found."
        )

    return mapping


def main() -> None:
    print("=" * 60)
    print(
        "Building FAISS Index from iALS Item Factors"
    )
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load item factors
    # ---------------------------------------------------------

    item_factors = (
        load_ials_item_factors(
            IALS_MODEL_PATH
        )
    )

    num_items, dimension = (
        item_factors.shape
    )

    print(
        f"Item factors shape: "
        f"({num_items}, {dimension})"
    )

    # ---------------------------------------------------------
    # 2. Load item mapping
    # ---------------------------------------------------------

    mapping = (
        load_and_validate_mapping(
            ITEM_MAPPING_PATH,
            num_items,
        )
    )

    # ---------------------------------------------------------
    # 3. Build FAISS index
    # ---------------------------------------------------------

    retriever = FaissRetriever(
        dimension=dimension,
    )

    retriever.add(
        item_factors,
        normalize=True,
    )

    print(
        f"Zero-vector items: "
        f"{retriever.zero_vector_count:,}"
    )

    print(
        f"Non-zero item vectors: "
        f"{retriever.num_items - retriever.zero_vector_count:,}"
    )

    print(
        f"FAISS vectors: "
        f"{retriever.num_items:,}"
    )

    # ---------------------------------------------------------
    # 4. Save index metadata
    # ---------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "model_type": "ials",
        "model_artifact": IALS_MODEL_PATH,
        "item_mapping": ITEM_MAPPING_PATH,
        "embedding_dimension": dimension,
        "num_items": num_items,
        "metric": "inner_product",
        "index_type": "IndexFlatIP",
        "normalized_vectors": True,
        "zero_vector_count": (
            retriever.zero_vector_count
        ),
    }

    # ---------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------

    retriever.save(
        path=str(INDEX_PATH),
        metadata_path=str(
            METADATA_PATH
        ),
        metadata=metadata,
    )

    print(
        f"Saved index: "
        f"{INDEX_PATH}"
    )

    print(
        f"Saved metadata: "
        f"{METADATA_PATH}"
    )

    print("\n=== Index Metadata ===")

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()