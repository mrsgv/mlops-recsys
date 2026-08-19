from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd

from src.evaluation.metrics import evaluate_top_k
from src.evaluation.split import (
    chronological_train_test_split,
    validate_split,
)
from src.models.ials_baseline import IALSRecommender


DATA_PATH = "data/processed/video_games.parquet"

MODEL_DIR = Path("models/ials")

MODEL_PATH = (
    MODEL_DIR / "ials_model.npz"
)

TOP_K = 10

FACTORS = 64
REGULARIZATION = 0.1
ITERATIONS = 20
ALPHA = 1.0
RANDOM_SEED = 42


def build_ground_truth(
    test_df: pd.DataFrame,
) -> dict[int, set[int]]:
    """
    Build user -> relevant item mapping.

    Chronological leave-one-out gives exactly one
    held-out item for every eligible user.
    """
    return (
        test_df
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def main() -> None:
    print("=" * 60)
    print("iALS Benchmark — Implicit Collaborative Filtering")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\n=== Loading Data ===")

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Interactions: {len(df):,}"
    )

    print(
        f"Users: "
        f"{df['user_idx'].nunique():,}"
    )

    print(
        f"Items: "
        f"{df['item_idx'].nunique():,}"
    )

    # ---------------------------------------------------------
    # 2. Common chronological split
    # ---------------------------------------------------------

    print(
        "\n=== Creating Common Chronological Split ==="
    )

    train_df, test_df = (
        chronological_train_test_split(df)
    )

    validate_split(
        train_df,
        test_df,
    )

    print(
        f"Train interactions: "
        f"{len(train_df):,}"
    )

    print(
        f"Test interactions: "
        f"{len(test_df):,}"
    )

    print(
        f"Test users: "
        f"{test_df['user_idx'].nunique():,}"
    )

    # ---------------------------------------------------------
    # 3. Dimensions
    # ---------------------------------------------------------

    num_users = (
        df["user_idx"].nunique()
    )

    num_items = (
        df["item_idx"].nunique()
    )

    # ---------------------------------------------------------
    # 4. Create model
    # ---------------------------------------------------------

    model = IALSRecommender(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        alpha=ALPHA,
        random_state=RANDOM_SEED,
    )

    # ---------------------------------------------------------
    # 5. MLflow
    # ---------------------------------------------------------

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )

    if tracking_uri:
        mlflow.set_tracking_uri(
            tracking_uri
        )

    mlflow.set_experiment(
        "ials"
    )

    with mlflow.start_run(
        run_name="ials-v1-binary"
    ):

        mlflow.log_params(
            {
                "model": "iALS",
                "factors": FACTORS,
                "regularization": REGULARIZATION,
                "iterations": ITERATIONS,
                "alpha": ALPHA,
                "random_seed": RANDOM_SEED,
                "num_users": num_users,
                "num_items": num_items,
                "train_interactions": len(
                    train_df
                ),
                "test_interactions": len(
                    test_df
                ),
                "feedback_type": "binary",
                "top_k": TOP_K,
            }
        )

        # -----------------------------------------------------
        # 6. Train
        # -----------------------------------------------------

        print(
            "\n=== Training iALS ==="
        )

        training_time = model.fit(
            train_df=train_df,
            num_users=num_users,
            num_items=num_items,
        )

        print(
            f"Training time: "
            f"{training_time:.2f} seconds"
        )

        mlflow.log_metric(
            "training_time_seconds",
            training_time,
        )

        # -----------------------------------------------------
        # 7. Generate recommendations
        # -----------------------------------------------------

        print(
            "\n=== Generating Recommendations ==="
        )

        test_users = (
            test_df["user_idx"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )

        recommendations = (
            model.recommend_users(
                user_ids=test_users,
                k=TOP_K,
            )
        )

        print(
            f"Users with recommendations: "
            f"{len(recommendations):,}"
        )

        # -----------------------------------------------------
        # 8. Evaluate
        # -----------------------------------------------------

        print(
            "\n=== Evaluating iALS ==="
        )

        ground_truth = build_ground_truth(
            test_df
        )

        metrics = evaluate_top_k(
            recommendations=recommendations,
            ground_truth=ground_truth,
            k=TOP_K,
        )

        print(
            f"Users evaluated: "
            f"{metrics['users_evaluated']:,}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{metrics[f'precision_at_{TOP_K}']:.6f}"
        )

        print(
            f"Recall@{TOP_K}: "
            f"{metrics[f'recall_at_{TOP_K}']:.6f}"
        )

        print(
            f"Hit Rate@{TOP_K}: "
            f"{metrics[f'hit_rate_at_{TOP_K}']:.6f}"
        )

        print(
            f"NDCG@{TOP_K}: "
            f"{metrics[f'ndcg_at_{TOP_K}']:.6f}"
        )

        mlflow.log_metrics(
            {
                f"precision_at_{TOP_K}":
                    float(
                        metrics[
                            f"precision_at_{TOP_K}"
                        ]
                    ),
                f"recall_at_{TOP_K}":
                    float(
                        metrics[
                            f"recall_at_{TOP_K}"
                        ]
                    ),
                f"hit_rate_at_{TOP_K}":
                    float(
                        metrics[
                            f"hit_rate_at_{TOP_K}"
                        ]
                    ),
                f"ndcg_at_{TOP_K}":
                    float(
                        metrics[
                            f"ndcg_at_{TOP_K}"
                        ]
                    ),
            }
        )

        # -----------------------------------------------------
        # 9. Save model
        # -----------------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.model.save(
            str(MODEL_PATH)
        )

        mlflow.log_artifact(
            str(MODEL_PATH)
        )

        print(
            f"\nModel saved to: "
            f"{MODEL_PATH}"
        )

    print("\n=== iALS Benchmark Complete ===")


if __name__ == "__main__":
    main()