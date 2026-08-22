"""
Evaluate a trained iALS artifact.

Training already reports metrics for the model it holds in memory. This
step re-evaluates the model *as saved on disk*, which is what the pipeline
actually promotes and serves. Running it as its own Airflow task means a
corrupt or mismatched artifact fails the pipeline before a FAISS index is
built on top of it.

Metrics are written next to the artifact for the model-selection step and,
when a training run is known, logged back to the same MLflow run under
``eval_`` names so the experiment shows both the in-training numbers and
the artifact numbers side by side.
"""

from __future__ import annotations

import argparse
import json
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

TRAINING_RUN_PATH = (
    MODEL_DIR / "training_run.json"
)

EVALUATION_PATH = (
    MODEL_DIR / "evaluation.json"
)

TOP_K = 10

# The artifact runs the same code path as training, so metrics should match
# to well within this tolerance. A larger gap means the artifact does not
# reproduce the run that produced it.
METRIC_TOLERANCE = 1e-6


def build_ground_truth(
    test_df: pd.DataFrame,
) -> dict[int, set[int]]:
    """
    Build user -> relevant item mapping.

    Chronological leave-one-out gives exactly one held-out item for every
    eligible user.
    """
    return (
        test_df
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def load_training_run(
    path: Path = TRAINING_RUN_PATH,
) -> dict[str, object] | None:
    """
    Load the training-run record written next to the artifact.

    Returns None when the record is absent, so a manually trained model can
    still be evaluated.
    """
    if not path.exists():
        return None

    return json.loads(
        path.read_text()
    )


def compare_with_training_metrics(
    evaluation_metrics: dict[str, float | int],
    training_metrics: dict[str, float],
    tolerance: float = METRIC_TOLERANCE,
) -> dict[str, dict[str, float]]:
    """
    Compare artifact metrics against the metrics training reported.

    Returns
    -------
    dict
        Only the metrics that differ by more than ``tolerance``, mapped to
        their training value, evaluation value and absolute difference.
    """
    mismatches: dict[str, dict[str, float]] = {}

    for name, training_value in training_metrics.items():
        if name not in evaluation_metrics:
            continue

        evaluation_value = float(
            evaluation_metrics[name]
        )

        difference = abs(
            evaluation_value
            - float(training_value)
        )

        if difference > tolerance:
            mismatches[name] = {
                "training": float(training_value),
                "evaluation": evaluation_value,
                "difference": difference,
            }

    return mismatches


def build_evaluation_record(
    metrics: dict[str, float | int],
    training_run: dict[str, object] | None,
    model_path: Path = MODEL_PATH,
    data_path: str = DATA_PATH,
    top_k: int = TOP_K,
) -> dict[str, object]:
    """
    Assemble the evaluation record consumed by model selection.

    ``deployable`` marks that this candidate has a working retrieval path
    (iALS factors are what the FAISS index is built from), which is what
    makes it eligible for promotion rather than merely comparable.
    """
    mlflow_metadata = {}

    if training_run:
        mlflow_metadata = dict(
            training_run.get("mlflow", {})
        )

    return {
        "model_type": "ials",
        "deployable": True,
        "retrieval": "faiss",
        "top_k": top_k,
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if key != "users_evaluated"
        },
        "users_evaluated": int(
            metrics["users_evaluated"]
        ),
        "artifacts": {
            "ials_model": str(model_path),
        },
        "dataset": {
            "interactions": data_path,
        },
        "mlflow": mlflow_metadata,
    }


def log_to_mlflow(
    run_id: str,
    metrics: dict[str, float | int],
    tracking_uri: str | None = None,
) -> None:
    """
    Attach artifact-evaluation metrics to the original training run.

    Names are prefixed with ``eval_`` so they never collide with the
    metrics training logged under the same run.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(
            tracking_uri
        )

    with mlflow.start_run(
        run_id=run_id,
    ):
        mlflow.log_metrics(
            {
                f"eval_{name}": float(value)
                for name, value in metrics.items()
                if name != "users_evaluated"
            }
        )

        mlflow.set_tag(
            "artifact_evaluated",
            "true",
        )


def evaluate(
    data_path: str = DATA_PATH,
    model_path: Path = MODEL_PATH,
    top_k: int = TOP_K,
) -> dict[str, float | int]:
    """Score the saved iALS artifact on the common evaluation split."""
    print("\n=== Loading Data ===")

    df = pd.read_parquet(data_path)

    print(f"Interactions: {len(df):,}")

    print("\n=== Creating Common Chronological Split ===")

    train_df, test_df = (
        chronological_train_test_split(df)
    )

    validate_split(
        train_df,
        test_df,
    )

    print(f"Train interactions: {len(train_df):,}")
    print(f"Test interactions: {len(test_df):,}")

    print("\n=== Loading iALS Artifact ===")

    model = IALSRecommender.from_artifact(
        model_path,
        train_df=train_df,
        num_users=int(
            df["user_idx"].nunique()
        ),
        num_items=int(
            df["item_idx"].nunique()
        ),
    )

    print(f"Artifact: {model_path}")
    print(f"Factors: {model.factors}")
    print(f"Regularization: {model.regularization}")
    print(f"Iterations: {model.iterations}")
    print(f"Alpha: {model.alpha}")

    print("\n=== Generating Recommendations ===")

    test_users = (
        test_df["user_idx"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    recommendations = model.recommend_users(
        user_ids=test_users,
        k=top_k,
    )

    print(
        f"Users with recommendations: "
        f"{len(recommendations):,}"
    )

    print("\n=== Evaluating Artifact ===")

    metrics = evaluate_top_k(
        recommendations=recommendations,
        ground_truth=build_ground_truth(
            test_df
        ),
        k=top_k,
    )

    print(
        f"Users evaluated: "
        f"{metrics['users_evaluated']:,}"
    )

    for name in (
        f"precision_at_{top_k}",
        f"recall_at_{top_k}",
        f"hit_rate_at_{top_k}",
        f"ndcg_at_{top_k}",
    ):
        print(f"{name}: {metrics[name]:.6f}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the saved iALS artifact and record its metrics."
        )
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help=(
            "Skip logging to MLflow. Useful offline or in CI."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if the artifact's metrics disagree with the metrics "
            "recorded during training."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("iALS Artifact Evaluation")
    print("=" * 60)

    metrics = evaluate()

    training_run = load_training_run()

    if training_run is None:
        print(
            "\nNo training run record found at "
            f"{TRAINING_RUN_PATH}; skipping run comparison."
        )
    else:
        mismatches = compare_with_training_metrics(
            metrics,
            training_run.get("metrics", {}),
        )

        if mismatches:
            message = (
                "Artifact metrics disagree with the training run: "
                f"{json.dumps(mismatches, indent=2)}"
            )

            if args.strict:
                raise SystemExit(
                    f"ERROR: {message}"
                )

            print(f"\nWARNING: {message}")
        else:
            print(
                "\nArtifact reproduces the training run's metrics."
            )

    record = build_evaluation_record(
        metrics=metrics,
        training_run=training_run,
    )

    EVALUATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_PATH.write_text(
        json.dumps(
            record,
            indent=2,
        )
        + "\n"
    )

    print(f"\nEvaluation written to: {EVALUATION_PATH}")

    run_id = (
        record.get("mlflow", {}).get("run_id")
        if record.get("mlflow")
        else None
    )

    if args.no_mlflow:
        print("MLflow logging skipped (--no-mlflow).")
    elif not run_id:
        print(
            "No MLflow run ID available; metrics were not logged to "
            "the tracking server."
        )
    else:
        log_to_mlflow(
            run_id=run_id,
            metrics=metrics,
            tracking_uri=os.environ.get(
                "MLFLOW_TRACKING_URI"
            ),
        )

        print(
            f"Metrics logged to MLflow run: {run_id}"
        )

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
