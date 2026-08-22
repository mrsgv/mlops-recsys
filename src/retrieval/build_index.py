"""
Build the FAISS index from the promoted model's item factors.

Previously this step read ``models/ials/ials_model.npz`` from a module
constant, which meant the index was always built from iALS no matter which
model the selection step had chosen. Selection and indexing could therefore
disagree silently: the pipeline would report promoting one model while
serving the item factors of another.

The index is now built from the canonical promoted artifact that the
selection step writes, and the metadata records which family and
hyperparameters produced it — so a mismatch becomes visible rather than
invisible.

IMPORTANT
---------
The index must NOT normalize vectors. Every supported factor model ranks by
raw inner product between a user factor row and an item factor row, and
normalizing item vectors would discard the magnitude that encodes item
popularity and, for BPR and LMF, the bias column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.faiss_index import (
    FaissRetriever,
)


PROMOTED_MODEL_PATH = (
    "models/promoted/model.npz"
)

SELECTION_PATH = (
    "models/deployment/selected_model.json"
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


def load_item_factors(
    path: str,
) -> np.ndarray:
    """
    Load item factors from a model's ``.npz`` artifact.

    Expected shape:
        (num_items, factors)

    A neighbourhood model's artifact has no ``item_factors`` array, so the
    absence of the key is reported as "this model cannot be served" rather
    than as a generic missing-key error.
    """
    resolved = Path(path)

    if not resolved.exists():
        raise FileNotFoundError(
            f"Promoted model not found: {resolved}. "
            "Run the selection step first."
        )

    with np.load(
        resolved,
        allow_pickle=False,
    ) as data:

        if "item_factors" not in data:
            raise ValueError(
                f"Model artifact {resolved} contains no "
                "'item_factors', so no inner-product index can be "
                "built from it. Only factor-based models are "
                "servable."
            )

        item_factors = np.asarray(
            data["item_factors"],
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


def load_selection(
    path: str = SELECTION_PATH,
) -> dict[str, object]:
    """
    Read the promotion decision so the index can record its provenance.

    Returns an empty dict when absent: the index is still buildable from a
    promoted artifact alone, it just carries less provenance.
    """
    resolved = Path(path)

    if not resolved.exists():
        return {}

    return json.loads(
        resolved.read_text()
    )


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

    if mapping[
        "parent_asin"
    ].duplicated().any():
        raise ValueError(
            "Duplicate parent_asin values found."
        )

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the FAISS index from the promoted model."
        )
    )

    parser.add_argument(
        "--model",
        default=PROMOTED_MODEL_PATH,
        help=(
            "Model artifact to index. Defaults to the canonical "
            "promoted artifact."
        ),
    )

    parser.add_argument(
        "--selection",
        default=SELECTION_PATH,
        help=(
            "Selection record, read for index provenance."
        ),
    )

    parser.add_argument(
        "--item-mapping",
        default=ITEM_MAPPING_PATH,
        help="item_idx -> parent_asin mapping.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(
        "Building FAISS Index from Promoted Item Factors"
    )
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Provenance
    # ---------------------------------------------------------

    selection = load_selection(
        args.selection
    )

    selected = selection.get(
        "selected",
        {},
    )

    model_name = selected.get(
        "name",
        "unknown",
    )

    model_family = selected.get(
        "family",
        "unknown",
    )

    print(
        f"\nPromoted model: {model_name} "
        f"(family={model_family})"
    )

    # ---------------------------------------------------------
    # 2. Load item factors
    # ---------------------------------------------------------

    item_factors = load_item_factors(
        args.model
    )

    num_items, dimension = (
        item_factors.shape
    )

    print(
        f"Item factors shape: "
        f"({num_items}, {dimension})"
    )

    # ---------------------------------------------------------
    # 3. Load item mapping
    # ---------------------------------------------------------

    load_and_validate_mapping(
        args.item_mapping,
        num_items,
    )

    # ---------------------------------------------------------
    # 4. Build FAISS index
    # ---------------------------------------------------------

    retriever = FaissRetriever(
        dimension=dimension,
        metric="inner_product",
        normalize=False,
    )

    retriever.add(
        item_factors
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
    # 5. Build metadata
    # ---------------------------------------------------------

    metadata = {
        "model_name": model_name,
        "model_type": model_family,
        "model_family": model_family,
        "model_params": selected.get(
            "params",
            {},
        ),
        "model_artifact": args.model,
        "item_mapping": args.item_mapping,
        "embedding_dimension": dimension,
        "num_items": num_items,
        "metric": "inner_product",
        "index_type": "IndexFlatIP",
        "normalized_vectors": False,
        "zero_vector_count": (
            retriever.zero_vector_count
        ),
        "mlflow": selected.get(
            "mlflow",
            {},
        ),
    }

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
