from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import evaluate_top_k
from src.evaluation.split import (
    chronological_train_test_split,
    validate_split,
)


DATA_PATH = "data/processed/video_games.parquet"
RECOMMENDATIONS_PATH = "data/predictions/svd_top10.parquet"
OUTPUT_PATH = "data/predictions/svd_evaluation.csv"

TOP_K = 10
SVD_FACTORS = 50


def load_data() -> pd.DataFrame:
    """Load the processed Video Games interaction dataset."""
    print("\n=== Loading Data ===")

    df = pd.read_parquet(DATA_PATH)

    print(f"Interactions: {len(df):,}")
    print(f"Users: {df['user_idx'].nunique():,}")
    print(f"Products: {df['item_idx'].nunique():,}")

    return df


def create_evaluation_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create and validate the common chronological evaluation split."""
    print("\n=== Creating Common Evaluation Split ===")

    train, test = chronological_train_test_split(df)
    validate_split(train, test)

    print(f"Training interactions: {len(train):,}")
    print(f"Test interactions: {len(test):,}")
    print(f"Test users: {test['user_idx'].nunique():,}")

    return train, test


def load_recommendations() -> pd.DataFrame:
    """Load precomputed SVD recommendations."""
    print("\n=== Loading SVD Recommendations ===")

    recommendations = pd.read_parquet(
        RECOMMENDATIONS_PATH
    )

    required_columns = {
        "user_idx",
        "item_idx",
        "rank",
    }

    missing = required_columns - set(recommendations.columns)

    if missing:
        raise ValueError(
            "Recommendation file is missing required columns: "
            f"{sorted(missing)}"
        )

    print(f"Recommendation rows: {len(recommendations):,}")
    print(
        "Users with recommendations: "
        f"{recommendations['user_idx'].nunique():,}"
    )

    return recommendations


def build_recommendation_map(
    recommendations: pd.DataFrame,
) -> dict[int, list[int]]:
    """Convert recommendation rows into ranked item lists per user."""
    recommendations = recommendations.sort_values(
        ["user_idx", "rank"]
    )

    return (
        recommendations
        .groupby("user_idx")["item_idx"]
        .apply(list)
        .to_dict()
    )


def build_ground_truth(
    test: pd.DataFrame,
) -> dict[int, set[int]]:
    """
    Build the evaluation ground-truth mapping.

    Our chronological leave-one-out protocol produces exactly
    one held-out item per eligible user.
    """
    return (
        test.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def main() -> None:
    df = load_data()

    _, test = create_evaluation_split(df)

    recommendations = load_recommendations()

    recommendation_map = build_recommendation_map(
        recommendations
    )

    ground_truth = build_ground_truth(test)

    print("\n=== Evaluating SVD ===")

    results = evaluate_top_k(
        recommendations=recommendation_map,
        ground_truth=ground_truth,
        k=TOP_K,
    )

    print(
        f"Users evaluated: "
        f"{results['users_evaluated']:,}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{results[f'precision_at_{TOP_K}']:.6f}"
    )

    print(
        f"Recall@{TOP_K}:    "
        f"{results[f'recall_at_{TOP_K}']:.6f}"
    )

    print(
        f"Hit Rate@{TOP_K}:  "
        f"{results[f'hit_rate_at_{TOP_K}']:.6f}"
    )

    print(
        f"NDCG@{TOP_K}:      "
        f"{results[f'ndcg_at_{TOP_K}']:.6f}"
    )

    output = {
        "model": "SVD",
        "factors": SVD_FACTORS,
        **results,
    }

    results_df = pd.DataFrame([output])

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== Saved ===")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()