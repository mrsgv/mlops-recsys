from __future__ import annotations

import gzip
import json
import os
import re
from typing import Any

import pandas as pd


METADATA_PATH = "data/raw/meta_Video_Games.jsonl.gz"
MAPPING_PATH = "data/processed/item_mapping.parquet"
OUTPUT_PATH = "data/processed/video_games_items.parquet"


def normalize_text(value: Any) -> str:
    """Convert a value into a normalized whitespace-separated string."""
    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(
            str(item)
            for item in value
            if item is not None
        )

    elif isinstance(value, dict):
        value = " ".join(
            f"{key} {val}"
            for key, val in value.items()
            if val is not None
        )

    value = str(value)

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def extract_brand(details: Any) -> str:
    """Extract brand from the metadata details dictionary."""
    if not isinstance(details, dict):
        return ""

    for key, value in details.items():
        if str(key).strip().lower() == "brand":
            return normalize_text(value)

    return ""


def parse_price(value: Any) -> float | None:
    """Extract a numeric price when one is available."""
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)",
        text,
    )

    if match is None:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def price_bucket(price: float | None) -> str:
    """Map price into deterministic static buckets."""
    if price is None:
        return "unknown"

    if price < 10:
        return "under_10"

    if price < 25:
        return "10_to_25"

    if price < 50:
        return "25_to_50"

    if price < 100:
        return "50_to_100"

    return "100_plus"


def load_metadata() -> pd.DataFrame:
    """Read the Video Games metadata JSONL file."""
    rows = []

    with gzip.open(
        METADATA_PATH,
        "rt",
        encoding="utf-8",
    ) as fh:
        for line_number, line in enumerate(
            fh,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on metadata line "
                    f"{line_number}."
                ) from exc

            parent_asin = record.get(
                "parent_asin"
            )

            if not parent_asin:
                continue

            title = normalize_text(
                record.get("title")
            )

            categories = normalize_text(
                record.get("categories")
            )

            features = normalize_text(
                record.get("features")
            )

            description = normalize_text(
                record.get("description")
            )

            details = record.get("details")

            brand = extract_brand(details)

            price = parse_price(
                record.get("price")
            )

            rows.append(
                {
                    "parent_asin": parent_asin,
                    "title": title,
                    "main_category": normalize_text(
                        record.get("main_category")
                    ),
                    "categories": categories,
                    "store": normalize_text(
                        record.get("store")
                    ),
                    "brand": brand,
                    "price": price,
                    "price_bucket": price_bucket(
                        price
                    ),
                    "title_length": len(title),
                    "feature_count": (
                        len(record.get("features", []))
                        if isinstance(
                            record.get("features"),
                            list,
                        )
                        else 0
                    ),
                    "features_text": features,
                    "description_text": description,
                }
            )

    return pd.DataFrame(rows)


def validate_metadata(
    metadata: pd.DataFrame,
) -> None:
    """Validate metadata before joining."""
    required_columns = {
        "parent_asin",
        "title",
        "main_category",
        "categories",
        "store",
        "brand",
        "price",
        "price_bucket",
        "title_length",
        "feature_count",
        "features_text",
        "description_text",
    }

    missing = (
        required_columns - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{sorted(missing)}"
        )

    if metadata["parent_asin"].duplicated().any():
        duplicates = int(
            metadata["parent_asin"].duplicated().sum()
        )

        raise ValueError(
            f"Metadata contains {duplicates} "
            "duplicate parent_asin values."
        )


def main() -> None:
    print("\n=== Loading Item Mapping ===")

    mapping = pd.read_parquet(
        MAPPING_PATH
    )

    print(
        f"Mapping rows: {len(mapping):,}"
    )

    if mapping["item_idx"].duplicated().any():
        raise ValueError(
            "Duplicate item_idx values found."
        )

    if mapping["parent_asin"].duplicated().any():
        raise ValueError(
            "Duplicate parent_asin values found."
        )

    print("\n=== Loading Metadata ===")

    metadata = load_metadata()

    print(
        f"Metadata rows: {len(metadata):,}"
    )

    validate_metadata(metadata)

    print("\n=== Joining Mapping + Metadata ===")

    items = mapping.merge(
        metadata,
        on="parent_asin",
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    items["metadata_found"] = (
        items["_merge"] == "both"
    )

    items = items.drop(
        columns=["_merge"]
    )

    missing_metadata = int(
        (~items["metadata_found"]).sum()
    )

    metadata_coverage = (
        items["metadata_found"].mean()
    )

    print(
        f"Items: {len(items):,}"
    )

    print(
        f"Items with metadata: "
        f"{items['metadata_found'].sum():,}"
    )

    print(
        f"Items without metadata: "
        f"{missing_metadata:,}"
    )

    print(
        f"Metadata coverage: "
        f"{metadata_coverage:.2%}"
    )

    # ---------------------------------------------------------
    # Missing-value handling
    # ---------------------------------------------------------

    text_columns = [
        "title",
        "main_category",
        "categories",
        "store",
        "brand",
        "features_text",
        "description_text",
    ]

    for column in text_columns:
        items[column] = (
            items[column]
            .fillna("")
            .astype(str)
        )

    items["price_bucket"] = (
        items["price_bucket"]
        .fillna("unknown")
        .astype(str)
    )

    items["title_length"] = (
        items["title_length"]
        .fillna(0)
        .astype(int)
    )

    items["feature_count"] = (
        items["feature_count"]
        .fillna(0)
        .astype(int)
    )

    # ---------------------------------------------------------
    # Final column order
    # ---------------------------------------------------------

    items = items[
        [
            "item_idx",
            "parent_asin",
            "metadata_found",
            "title",
            "main_category",
            "categories",
            "store",
            "brand",
            "price",
            "price_bucket",
            "title_length",
            "feature_count",
            "features_text",
            "description_text",
        ]
    ].sort_values(
        "item_idx"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Validate final table
    # ---------------------------------------------------------

    if len(items) != len(mapping):
        raise ValueError(
            "Item feature table size does not match "
            "the item mapping."
        )

    if items["item_idx"].tolist() != mapping[
        "item_idx"
    ].tolist():
        raise ValueError(
            "Item ordering changed during metadata join."
        )

    if items["item_idx"].duplicated().any():
        raise ValueError(
            "Duplicate item_idx values in final "
            "feature table."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    items.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== Final Item Feature Table ===")

    print(
        f"Rows: {len(items):,}"
    )

    print(
        f"Columns: {items.columns.tolist()}"
    )

    print("\n=== Sample ===")

    print(
        items.head(5).to_string(
            index=False
        )
    )

    print("\n=== Saved ===")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()