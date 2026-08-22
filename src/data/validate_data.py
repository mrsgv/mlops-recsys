"""
Data validation for the recommendation pipeline.

This module is the pipeline's ``validate_data`` gate. It is deliberately
independent of Spark so that it can run as a cheap Airflow task before the
expensive preprocessing job starts, and again afterwards to prove that the
preprocessing output is fit for training.

Two stages are supported:

raw
    Validates ``data/raw/Video_Games.csv.gz`` — the schema the Spark
    preprocessing job expects. Catching a bad download here saves a full
    Spark run.

processed
    Validates ``data/processed/video_games.parquet`` and
    ``data/processed/item_mapping.parquet`` — the contract that iALS
    training, FAISS index building and the serving retriever all rely on:
    contiguous integer indices starting at zero, no nulls, no duplicate
    interactions, and an item mapping that lines up one-to-one with the
    interaction item space.

Validation failure raises DataValidationError, which exits non-zero and
fails the Airflow task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RAW_INTERACTIONS_PATH = "data/raw/Video_Games.csv.gz"

PROCESSED_INTERACTIONS_PATH = (
    "data/processed/video_games.parquet"
)

PROCESSED_ITEM_MAPPING_PATH = (
    "data/processed/item_mapping.parquet"
)

RAW_REQUIRED_COLUMNS = {
    "user_id",
    "parent_asin",
    "rating",
    "timestamp",
}

PROCESSED_REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "rating",
    "timestamp",
}

ITEM_MAPPING_REQUIRED_COLUMNS = {
    "item_idx",
    "parent_asin",
}

MIN_RATING = 1.0
MAX_RATING = 5.0

# A successful run of this dataset produces ~814k interactions. Anything
# far below that means a truncated download or a partial Spark write.
MIN_INTERACTIONS = 10_000


class DataValidationError(Exception):
    """Raised when a dataset violates the pipeline's data contract."""


def check_required_columns(
    df: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    """Fail if any required column is absent."""
    missing = required - set(df.columns)

    if missing:
        raise DataValidationError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def check_no_nulls(
    df: pd.DataFrame,
    columns: set[str],
    dataset_name: str,
) -> None:
    """Fail if any required column contains nulls."""
    null_counts = {
        column: int(df[column].isna().sum())
        for column in sorted(columns)
    }

    offenders = {
        column: count
        for column, count in null_counts.items()
        if count > 0
    }

    if offenders:
        raise DataValidationError(
            f"{dataset_name} contains null values: "
            f"{offenders}"
        )


def check_rating_range(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Fail if ratings fall outside the 1-5 star range."""
    ratings = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    if ratings.isna().any():
        raise DataValidationError(
            f"{dataset_name} contains non-numeric ratings."
        )

    out_of_range = int(
        (
            (ratings < MIN_RATING)
            | (ratings > MAX_RATING)
        ).sum()
    )

    if out_of_range:
        raise DataValidationError(
            f"{dataset_name} contains {out_of_range} ratings "
            f"outside [{MIN_RATING}, {MAX_RATING}]."
        )


def check_contiguous_index(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> int:
    """
    Fail unless ``column`` holds every integer in ``0..n-1`` exactly once
    across the dataset.

    Every downstream component — the SVD/iALS matrices, the FAISS index and
    the serving item mapping — treats these indices as row positions, so a
    gap silently misaligns embeddings with items.

    Returns
    -------
    int
        The index cardinality (``n``).
    """
    values = df[column]

    if not pd.api.types.is_integer_dtype(values):
        coerced = pd.to_numeric(
            values,
            errors="coerce",
        )

        if (
            coerced.isna().any()
            or not (coerced % 1 == 0).all()
        ):
            raise DataValidationError(
                f"{dataset_name}.{column} must contain integers."
            )

        values = coerced.astype("int64")

    cardinality = int(values.nunique())

    minimum = int(values.min())
    maximum = int(values.max())

    if minimum != 0 or maximum != cardinality - 1:
        raise DataValidationError(
            f"{dataset_name}.{column} must be contiguous from 0 to "
            f"{cardinality - 1}, but spans [{minimum}, {maximum}]."
        )

    return cardinality


def validate_raw_frame(
    df: pd.DataFrame,
) -> dict[str, object]:
    """
    Validate the raw interaction dataset and return summary statistics.
    """
    check_required_columns(
        df,
        RAW_REQUIRED_COLUMNS,
        "Raw interactions",
    )

    if len(df) < MIN_INTERACTIONS:
        raise DataValidationError(
            f"Raw interactions has only {len(df):,} rows; at least "
            f"{MIN_INTERACTIONS:,} were expected."
        )

    check_no_nulls(
        df,
        RAW_REQUIRED_COLUMNS,
        "Raw interactions",
    )

    check_rating_range(
        df,
        "Raw interactions",
    )

    timestamps = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise DataValidationError(
            "Raw interactions contains non-numeric timestamps."
        )

    if (timestamps <= 0).any():
        raise DataValidationError(
            "Raw interactions contains non-positive timestamps."
        )

    return {
        "stage": "raw",
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "items": int(df["parent_asin"].nunique()),
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
        "timestamp_min": int(timestamps.min()),
        "timestamp_max": int(timestamps.max()),
    }


def validate_processed_frame(
    df: pd.DataFrame,
) -> dict[str, object]:
    """
    Validate the processed interaction dataset and return summary
    statistics.
    """
    check_required_columns(
        df,
        PROCESSED_REQUIRED_COLUMNS,
        "Processed interactions",
    )

    if len(df) < MIN_INTERACTIONS:
        raise DataValidationError(
            f"Processed interactions has only {len(df):,} rows; at least "
            f"{MIN_INTERACTIONS:,} were expected."
        )

    check_no_nulls(
        df,
        PROCESSED_REQUIRED_COLUMNS,
        "Processed interactions",
    )

    check_rating_range(
        df,
        "Processed interactions",
    )

    num_users = check_contiguous_index(
        df,
        "user_idx",
        "Processed interactions",
    )

    num_items = check_contiguous_index(
        df,
        "item_idx",
        "Processed interactions",
    )

    duplicates = int(
        df.duplicated(
            subset=[
                "user_idx",
                "item_idx",
            ]
        ).sum()
    )

    if duplicates:
        raise DataValidationError(
            f"Processed interactions contains {duplicates:,} duplicate "
            "user-item pairs."
        )

    # Leave-one-out evaluation needs at least two interactions per user,
    # so a dataset where no user qualifies cannot be trained on.
    interactions_per_user = (
        df.groupby("user_idx").size()
    )

    eligible_users = int(
        (interactions_per_user >= 2).sum()
    )

    if eligible_users == 0:
        raise DataValidationError(
            "No user has at least two interactions; chronological "
            "leave-one-out evaluation is impossible."
        )

    return {
        "stage": "processed",
        "interactions": int(len(df)),
        "users": num_users,
        "items": num_items,
        "users_eligible_for_evaluation": eligible_users,
        "min_interactions_per_user": int(
            interactions_per_user.min()
        ),
    }


def validate_item_mapping_frame(
    mapping: pd.DataFrame,
    num_items: int,
) -> dict[str, object]:
    """
    Validate the item mapping against the interaction item space.

    The mapping is what turns an internal ``item_idx`` back into a real
    product, so it must cover exactly the same items as the interactions.
    """
    check_required_columns(
        mapping,
        ITEM_MAPPING_REQUIRED_COLUMNS,
        "Item mapping",
    )

    check_no_nulls(
        mapping,
        ITEM_MAPPING_REQUIRED_COLUMNS,
        "Item mapping",
    )

    if len(mapping) != num_items:
        raise DataValidationError(
            f"Item mapping has {len(mapping):,} rows but the "
            f"interactions contain {num_items:,} items."
        )

    check_contiguous_index(
        mapping,
        "item_idx",
        "Item mapping",
    )

    duplicate_asins = int(
        mapping["parent_asin"].duplicated().sum()
    )

    if duplicate_asins:
        raise DataValidationError(
            f"Item mapping contains {duplicate_asins:,} duplicate "
            "parent_asin values."
        )

    return {
        "rows": int(len(mapping)),
        "unique_parent_asin": int(
            mapping["parent_asin"].nunique()
        ),
    }


def _require_path(
    path: str,
    description: str,
) -> Path:
    resolved = Path(path)

    if not resolved.exists():
        raise DataValidationError(
            f"{description} not found: {path}. "
            "Run 'dvc pull' to fetch versioned data."
        )

    return resolved


def validate_raw(
    interactions_path: str = RAW_INTERACTIONS_PATH,
    sample_rows: int | None = None,
) -> dict[str, object]:
    """Validate the raw dataset on disk."""
    print("\n=== Validating Raw Interactions ===")

    resolved = _require_path(
        interactions_path,
        "Raw interaction dataset",
    )

    df = pd.read_csv(
        resolved,
        nrows=sample_rows,
    )

    print(f"Path: {interactions_path}")
    print(f"Rows read: {len(df):,}")

    report = validate_raw_frame(df)

    report["path"] = interactions_path
    report["sampled"] = sample_rows is not None

    return report


def validate_processed(
    interactions_path: str = PROCESSED_INTERACTIONS_PATH,
    item_mapping_path: str = PROCESSED_ITEM_MAPPING_PATH,
) -> dict[str, object]:
    """Validate the processed dataset and item mapping on disk."""
    print("\n=== Validating Processed Interactions ===")

    interactions = _require_path(
        interactions_path,
        "Processed interaction dataset",
    )

    mapping_path = _require_path(
        item_mapping_path,
        "Item mapping",
    )

    df = pd.read_parquet(interactions)

    print(f"Path: {interactions_path}")
    print(f"Interactions: {len(df):,}")

    report = validate_processed_frame(df)

    print("\n=== Validating Item Mapping ===")

    mapping = pd.read_parquet(mapping_path)

    print(f"Path: {item_mapping_path}")
    print(f"Rows: {len(mapping):,}")

    report["item_mapping"] = (
        validate_item_mapping_frame(
            mapping,
            num_items=int(report["items"]),
        )
    )

    report["paths"] = {
        "interactions": interactions_path,
        "item_mapping": item_mapping_path,
    }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the recommendation pipeline's datasets."
        )
    )

    parser.add_argument(
        "--stage",
        choices=[
            "raw",
            "processed",
        ],
        default="processed",
        help=(
            "Which dataset to validate: the raw download that "
            "preprocessing consumes, or the processed output that "
            "training consumes."
        ),
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help=(
            "Validate only the first N rows of the raw dataset. "
            "Useful for a fast smoke check."
        ),
    )

    parser.add_argument(
        "--report",
        default=None,
        help=(
            "Optional path to write the validation report as JSON."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(
        f"Data Validation — stage: {args.stage}"
    )
    print("=" * 60)

    if args.stage == "raw":
        report = validate_raw(
            sample_rows=args.sample_rows,
        )
    else:
        report = validate_processed()

    print("\n=== Validation Report ===")

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    if args.report:
        report_path = Path(args.report)

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
            + "\n"
        )

        print(f"\nReport written to: {args.report}")

    print("\n=== Validation Passed ===")


if __name__ == "__main__":
    main()
