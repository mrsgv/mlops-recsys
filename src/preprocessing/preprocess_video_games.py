import gzip
import os

import pandas as pd


def main():
    input_path = "data/raw/Video_Games.csv.gz"
    output_path = "data/processed/video_games.parquet"

    print("\n=== Reading Input ===")

    df = pd.read_csv(
        input_path,
        compression="gzip"
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # ---------------------------------------------------------
    # 1. Validate
    # ---------------------------------------------------------
    print("\n=== Null Counts ===")
    print(df.isnull().sum())

    print("\n=== Invalid Ratings ===")
    invalid_ratings = ((df["rating"] < 1) | (df["rating"] > 5)).sum()
    print(invalid_ratings)

    # ---------------------------------------------------------
    # 2. Remove duplicate user-product pairs
    # ---------------------------------------------------------
    duplicate_count = df.duplicated(
        subset=["user_id", "parent_asin"]
    ).sum()

    print("\n=== Duplicate User-Product Pairs ===")
    print(duplicate_count)

    df = df.drop_duplicates(
        subset=["user_id", "parent_asin"]
    ).copy()

    # ---------------------------------------------------------
    # 3. Create integer user IDs
    # ---------------------------------------------------------
    user_ids = {
        user_id: idx
        for idx, user_id in enumerate(
            sorted(df["user_id"].unique())
        )
    }

    # ---------------------------------------------------------
    # 4. Create integer item IDs
    # ---------------------------------------------------------
    item_ids = {
        item_id: idx
        for idx, item_id in enumerate(
            sorted(df["parent_asin"].unique())
        )
    }

    df["user_idx"] = df["user_id"].map(user_ids)
    df["item_idx"] = df["parent_asin"].map(item_ids)

    # ---------------------------------------------------------
    # 5. Select model-ready columns
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
    # 6. Save
    # ---------------------------------------------------------
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    processed.to_parquet(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Final summary
    # ---------------------------------------------------------
    print("\n=== Processed Dataset ===")
    print(f"Interactions: {len(processed):,}")
    print(f"Users: {processed['user_idx'].nunique():,}")
    print(f"Products: {processed['item_idx'].nunique():,}")

    print("\n=== Processed Schema ===")
    print(processed.dtypes)

    print("\n=== First 10 Rows ===")
    print(processed.head(10).to_string(index=False))

    print("\n=== Saved To ===")
    print(output_path)


if __name__ == "__main__":
    main()