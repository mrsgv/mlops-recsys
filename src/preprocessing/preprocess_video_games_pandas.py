from __future__ import annotations

import os

import pandas as pd

"""
Pandas implementation - kept for reference.

It is not used in the pipeline.

It is used to validate the Spark implementation.
"""


INPUT_PATH = "data/raw/Video_Games.csv.gz"

INTERACTIONS_OUTPUT_PATH = (
    "data/processed/video_games.parquet"
)

ITEM_MAPPING_OUTPUT_PATH = (
    "data/processed/item_mapping.parquet"
)


def main() -> None:
    print("\n=== Reading Input ===")

    df = pd.read_csv(
        INPUT_PATH,
        compression="gzip",
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # ---------------------------------------------------------
    # 1. Validate input schema
    # ---------------------------------------------------------

    required_columns = {
        "user_id",
        "parent_asin",
        "rating",
        "timestamp",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 2. Validate values
    # ---------------------------------------------------------

    print("\n=== Null Counts ===")
    print(df.isnull().sum())

    invalid_ratings = (
        (df["rating"] < 1)
        | (df["rating"] > 5)
    ).sum()

    print("\n=== Invalid Ratings ===")
    print(invalid_ratings)

    if invalid_ratings > 0:
        raise ValueError(
            f"Found {invalid_ratings} invalid ratings."
        )

    # ---------------------------------------------------------
    # 3. Remove duplicate user-product interactions
    # ---------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=[
            "user_id",
            "parent_asin",
        ]
    ).sum()

    print("\n=== Duplicate User-Product Pairs ===")
    print(duplicate_count)

    df = df.drop_duplicates(
        subset=[
            "user_id",
            "parent_asin",
        ]
    ).copy()

    # ---------------------------------------------------------
    # 4. Create deterministic user IDs
    # ---------------------------------------------------------

    user_ids = {
        user_id: idx
        for idx, user_id in enumerate(
            sorted(df["user_id"].unique())
        )
    }

    # ---------------------------------------------------------
    # 5. Create deterministic item IDs
    # ---------------------------------------------------------

    item_ids = {
        item_id: idx
        for idx, item_id in enumerate(
            sorted(df["parent_asin"].unique())
        )
    }

    # ---------------------------------------------------------
    # 6. Apply mappings
    # ---------------------------------------------------------

    df["user_idx"] = df["user_id"].map(
        user_ids
    )

    df["item_idx"] = df["parent_asin"].map(
        item_ids
    )

    # ---------------------------------------------------------
    # 7. Validate mappings
    # ---------------------------------------------------------

    if df["user_idx"].isna().any():
        raise ValueError(
            "Some user IDs could not be mapped."
        )

    if df["item_idx"].isna().any():
        raise ValueError(
            "Some item IDs could not be mapped."
        )

    # ---------------------------------------------------------
    # 8. Build interaction dataset
    # ---------------------------------------------------------

    processed = df[
        [
            "user_idx",
            "item_idx",
            "rating",
            "timestamp",
        ]
    ].copy()

    # ---------------------------------------------------------
    # 9. Build item mapping dataset
    # ---------------------------------------------------------

    item_mapping = pd.DataFrame(
        [
            {
                "item_idx": item_idx,
                "parent_asin": parent_asin,
            }
            for parent_asin, item_idx in item_ids.items()
        ]
    ).sort_values(
        "item_idx"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 10. Validate item mapping
    # ---------------------------------------------------------

    if len(item_mapping) != len(item_ids):
        raise ValueError(
            "Unexpected item mapping size."
        )

    if item_mapping["item_idx"].duplicated().any():
        raise ValueError(
            "Duplicate item_idx values found."
        )

    if item_mapping["parent_asin"].duplicated().any():
        raise ValueError(
            "Duplicate parent_asin values found."
        )

    expected_item_indices = list(
        range(len(item_mapping))
    )

    actual_item_indices = (
        item_mapping["item_idx"].tolist()
    )

    if actual_item_indices != expected_item_indices:
        raise ValueError(
            "item_idx values are not contiguous "
            "starting from zero."
        )

    # ---------------------------------------------------------
    # 11. Save outputs
    # ---------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    processed.to_parquet(
        INTERACTIONS_OUTPUT_PATH,
        index=False,
    )

    item_mapping.to_parquet(
        ITEM_MAPPING_OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # 12. Summary
    # ---------------------------------------------------------

    print("\n=== Processed Dataset ===")
    print(
        f"Interactions: "
        f"{len(processed):,}"
    )
    print(
        f"Users: "
        f"{processed['user_idx'].nunique():,}"
    )
    print(
        f"Products: "
        f"{processed['item_idx'].nunique():,}"
    )

    print("\n=== Item Mapping ===")
    print(
        f"Items: "
        f"{len(item_mapping):,}"
    )

    print("\n=== Interaction Schema ===")
    print(processed.dtypes)

    print("\n=== Item Mapping Schema ===")
    print(item_mapping.dtypes)

    print("\n=== Sample Item Mapping ===")
    print(
        item_mapping
        .head(10)
        .to_string(index=False)
    )

    print("\n=== Saved Outputs ===")
    print(
        f"Interactions: "
        f"{INTERACTIONS_OUTPUT_PATH}"
    )
    print(
        f"Item mapping: "
        f"{ITEM_MAPPING_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()