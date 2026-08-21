from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = str(PROJECT_ROOT / "data/raw/Video_Games.csv.gz")

INTERACTIONS_OUTPUT_PATH = str(
    PROJECT_ROOT / "data/processed/video_games.parquet"
)

ITEM_MAPPING_OUTPUT_PATH = str(
    PROJECT_ROOT / "data/processed/item_mapping.parquet"
)


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("VideoGamesPreprocessing")
        .master("local[*]")
        .getOrCreate()
    )


def validate_schema(df) -> None:
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


def main() -> None:
    spark = create_spark_session()

    try:
        print("\n=== Reading Input ===")

        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(INPUT_PATH)
        )

        print(f"Rows: {df.count():,}")
        print(f"Columns: {df.columns}")

        # -----------------------------------------------------
        # 1. Validate input schema
        # -----------------------------------------------------

        validate_schema(df)

        # -----------------------------------------------------
        # 2. Validate nulls
        # -----------------------------------------------------

        print("\n=== Null Counts ===")

        null_counts = df.select(
            *[
                F.sum(
                    F.col(column).isNull().cast("long")
                ).alias(column)
                for column in df.columns
            ]
        )

        null_counts.show()

        # -----------------------------------------------------
        # 3. Validate ratings
        # -----------------------------------------------------

        invalid_ratings = (
            df.filter(
                (F.col("rating") < 1)
                | (F.col("rating") > 5)
            )
            .count()
        )

        print("\n=== Invalid Ratings ===")
        print(invalid_ratings)

        if invalid_ratings > 0:
            raise ValueError(
                f"Found {invalid_ratings} invalid ratings."
            )

        # -----------------------------------------------------
        # 4. Remove duplicate user-product interactions
        # -----------------------------------------------------

        duplicate_pairs = (
            df.groupBy(
                "user_id",
                "parent_asin",
            )
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        print("\n=== Duplicate User-Product Pairs ===")
        print(duplicate_pairs)

        df = df.dropDuplicates(
            ["user_id", "parent_asin"]
        )

        # -----------------------------------------------------
        # 5. Create deterministic user IDs
        # -----------------------------------------------------

        print("\n=== Creating User Mapping ===")

        user_window = Window.orderBy("user_id")

        user_mapping = (
            df.select("user_id")
            .distinct()
            .withColumn(
                "user_idx",
                F.row_number().over(user_window) - 1,
            )
        )

        # -----------------------------------------------------
        # 6. Create deterministic item IDs
        # -----------------------------------------------------

        print("\n=== Creating Item Mapping ===")

        item_window = Window.orderBy("parent_asin")

        item_mapping = (
            df.select("parent_asin")
            .distinct()
            .withColumn(
                "item_idx",
                F.row_number().over(item_window) - 1,
            )
            .select(
                "item_idx",
                "parent_asin",
            )
        )

        # -----------------------------------------------------
        # 7. Apply mappings
        # -----------------------------------------------------

        processed = (
            df
            .join(
                user_mapping,
                on="user_id",
                how="inner",
            )
            .join(
                item_mapping,
                on="parent_asin",
                how="inner",
            )
            .select(
                "user_idx",
                "item_idx",
                "rating",
                "timestamp",
            )
        )

        # -----------------------------------------------------
        # 8. Validate mappings
        # -----------------------------------------------------

        missing_user_mapping = (
            processed
            .filter(F.col("user_idx").isNull())
            .count()
        )

        missing_item_mapping = (
            processed
            .filter(F.col("item_idx").isNull())
            .count()
        )

        if missing_user_mapping > 0:
            raise ValueError(
                "Some user IDs could not be mapped."
            )

        if missing_item_mapping > 0:
            raise ValueError(
                "Some item IDs could not be mapped."
            )

        # -----------------------------------------------------
        # 9. Validate item mapping
        # -----------------------------------------------------

        item_count = item_mapping.count()

        distinct_item_indices = (
            item_mapping
            .select("item_idx")
            .distinct()
            .count()
        )

        if distinct_item_indices != item_count:
            raise ValueError(
                "Duplicate item_idx values found."
            )

        expected_max_item_idx = item_count - 1

        actual_max_item_idx = (
            item_mapping
            .agg(F.max("item_idx"))
            .first()[0]
        )

        if actual_max_item_idx != expected_max_item_idx:
            raise ValueError(
                "item_idx values are not contiguous "
                "starting from zero."
            )

        # -----------------------------------------------------
        # 10. Summary
        # -----------------------------------------------------

        print("\n=== Processed Dataset ===")

        print(
            f"Interactions: "
            f"{processed.count():,}"
        )

        print(
            f"Users: "
            f"{processed.select('user_idx').distinct().count():,}"
        )

        print(
            f"Products: "
            f"{processed.select('item_idx').distinct().count():,}"
        )

        print("\n=== Item Mapping ===")

        print(
            f"Items: "
            f"{item_count:,}"
        )

        print("\n=== Interaction Schema ===")
        processed.printSchema()

        print("\n=== Item Mapping Schema ===")
        item_mapping.printSchema()

        print("\n=== Sample Item Mapping ===")

        (
            item_mapping
            .orderBy("item_idx")
            .show(10, truncate=False)
        )

        # -----------------------------------------------------
        # 11. Save outputs
        # -----------------------------------------------------

        os.makedirs(
            PROJECT_ROOT / "data/processed",
            exist_ok=True,
        )

        (
            processed
            .write
            .mode("overwrite")
            .parquet(
                INTERACTIONS_OUTPUT_PATH
            )
        )

        (
            item_mapping
            .write
            .mode("overwrite")
            .parquet(
                ITEM_MAPPING_OUTPUT_PATH
            )
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

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
