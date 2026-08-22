"""
Evaluate the SVD baseline and record it as an MLflow run.

Tracking lives here rather than in ``src/models/svd_baseline.py`` for a
practical reason: training materialises a dense
``num_users x num_items`` prediction matrix — roughly 18 GB for this
dataset — so re-running it merely to attach tracking is expensive and
risky. Evaluation reads the stored Top-K predictions instead and produces
the same numbers in seconds.

The model artifacts themselves stay versioned by DVC (``models/svd``);
MLflow records the experiment, parameters and metrics, which is what makes
the baseline comparable to iALS and the Two-Tower runs in one place.
"""

from __future__ import annotations

import argparse
import os

import mlflow
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

EXPERIMENT_NAME = "svd"
RUN_NAME = "svd-baseline-50"


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


def build_run_payload(
    results: dict[str, float | int],
    num_users: int,
    num_items: int,
    train_interactions: int,
    test_interactions: int,
) -> tuple[
    dict[str, object],
    dict[str, float],
    dict[str, str],
]:
    """
    Assemble the params, metrics and tags for the SVD run.

    Kept separate from the logging call so the payload can be asserted in
    tests without a tracking server.

    ``metrics_source`` records that the numbers come from the stored Top-K
    predictions rather than a fresh training pass, so nobody later mistakes
    this for a run that retrained the model.
    """
    params = {
        "model": "SVD",
        "factors": SVD_FACTORS,
        "top_k": TOP_K,
        "num_users": num_users,
        "num_items": num_items,
        "train_interactions": train_interactions,
        "test_interactions": test_interactions,
        "feedback_type": "explicit",
        "dataset_path": DATA_PATH,
        "recommendations_path": (
            RECOMMENDATIONS_PATH
        ),
    }

    metrics = {
        name: float(value)
        for name, value in results.items()
        if name != "users_evaluated"
    }

    metrics["users_evaluated"] = float(
        results["users_evaluated"]
    )

    tags = {
        "model_type": "svd",
        # No FAISS path: the index is built from iALS item factors, so the
        # SVD baseline is comparable but not servable.
        "retrieval": "none",
        "stage": "baseline",
        "dataset": DATA_PATH,
        "metrics_source": "stored_predictions",
    }

    return params, metrics, tags


def log_to_mlflow(
    params: dict[str, object],
    metrics: dict[str, float],
    tags: dict[str, str],
    tracking_uri: str | None = None,
) -> str:
    """
    Record the baseline as an MLflow run and return its run ID.

    Model files are deliberately not uploaded: DVC already versions
    ``models/svd``, so an MLflow copy would duplicate it.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(
            tracking_uri
        )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name=RUN_NAME
    ) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        return run.info.run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the SVD baseline and log it to MLflow."
        )
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help=(
            "Skip logging to MLflow. Useful offline or in CI."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_data()

    train, test = create_evaluation_split(df)

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

    # ---------------------------------------------------------
    # MLflow
    # ---------------------------------------------------------

    if args.no_mlflow:
        print(
            "\nMLflow logging skipped (--no-mlflow)."
        )

        return

    params, metrics, tags = build_run_payload(
        results=results,
        num_users=int(
            df["user_idx"].nunique()
        ),
        num_items=int(
            df["item_idx"].nunique()
        ),
        train_interactions=len(train),
        test_interactions=len(test),
    )

    run_id = log_to_mlflow(
        params=params,
        metrics=metrics,
        tags=tags,
        tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI"
        ),
    )

    print("\n=== MLflow ===")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run name:   {RUN_NAME}")
    print(f"Run ID:     {run_id}")


if __name__ == "__main__":
    main()