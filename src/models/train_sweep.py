"""
Train and evaluate a declared grid of model variants.

This replaces the single-model ``train_ials`` step in the pipeline. That
script trained one model with hyperparameters frozen as module constants,
which is why the deployed champion ran for months with ``alpha=1.0`` — a
value no experiment ever challenged. A sweep makes the comparison the
pipeline's job rather than a thing someone remembered to do by hand.

Every variant produces a self-describing candidate directory:

    models/candidates/<slug>/
        model.npz           the fitted artifact
        training_run.json    params, metrics, MLflow run, timing
        evaluation.json      the record model selection reads

``select_model`` discovers candidates by globbing those directories, so a
variant becomes promotable simply by existing. Nothing here decides what to
deploy; that stays in the selection step.

MLflow layout
-------------
One parent run per sweep, one nested child run per variant. Flat runs would
give N identically-named entries with no grouping, which makes the MLflow
comparison view useless exactly when it matters most. Nesting means the
whole sweep is one collapsible unit, and the parent carries the winner so
the experiment page answers "which run won" without sorting by hand.

``--no-mlflow`` skips tracking entirely so the step stays runnable offline
and in CI, where no tracking server is reachable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_top_k
from src.evaluation.split import (
    chronological_train_test_split,
    validate_split,
)
from src.models.ials_baseline import (
    build_user_item_matrix,
)
from src.models.registry import (
    extract_factors,
    fit_model,
    get_family,
    recommend_batch,
    save_model,
    variant_slug,
)
from src.models.tracking import (
    log_artifacts_safely,
    mark_artifacts_uploaded,
    report_upload_outcome,
    run_identity,
)


DATA_PATH = "data/processed/video_games.parquet"

CANDIDATES_DIR = Path(
    "models/candidates"
)

LEADERBOARD_PATH = (
    CANDIDATES_DIR / "leaderboard.json"
)

EXPERIMENT_NAME = "recsys-sweep"

PRIMARY_METRIC = "recall_at_10"

TOP_K = 10

RANDOM_SEED = 42

# The declared sweep.
#
# ALS dominates the grid on purpose. The scouting run showed that confidence
# weighting, not capacity, was the binding constraint: at alpha=1 going from
# 64 to 256 factors gained about 5%, while raising alpha from 1 to 40 at a
# fixed 64 factors gained about 91%. So alpha is swept until it stops
# paying, and capacity and regularization are swept around the best shape.
#
# BPR and LMF get a genuine tuning pass rather than one token entry, because
# a comparison table is only defensible if every family had a fair chance.
#
# BM25 and cosine are neighbourhood models and can never be promoted, but
# they fit in under a second and beat several factor models, so they belong
# in the comparison as an honest cheap baseline.
DEFAULT_SWEEP: list[tuple[str, dict[str, Any]]] = [
    # --- ALS: control, reproducing the previously deployed champion ------
    (
        "als",
        {
            "factors": 64,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 1.0,
        },
    ),
    # --- ALS: alpha ceiling ---------------------------------------------
    (
        "als",
        {
            "factors": 64,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 40.0,
        },
    ),
    (
        "als",
        {
            "factors": 128,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 40.0,
        },
    ),
    (
        "als",
        {
            "factors": 128,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 60.0,
        },
    ),
    (
        "als",
        {
            "factors": 128,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 100.0,
        },
    ),
    (
        "als",
        {
            "factors": 128,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 200.0,
        },
    ),
    # --- ALS: capacity --------------------------------------------------
    (
        "als",
        {
            "factors": 256,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 40.0,
        },
    ),
    (
        "als",
        {
            "factors": 256,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 100.0,
        },
    ),
    (
        "als",
        {
            "factors": 512,
            "regularization": 0.1,
            "iterations": 20,
            "alpha": 40.0,
        },
    ),
    # --- ALS: regularization around the best shape ----------------------
    (
        "als",
        {
            "factors": 256,
            "regularization": 0.01,
            "iterations": 20,
            "alpha": 40.0,
        },
    ),
    (
        "als",
        {
            "factors": 256,
            "regularization": 0.5,
            "iterations": 20,
            "alpha": 40.0,
        },
    ),
    (
        "als",
        {
            "factors": 256,
            "regularization": 2.0,
            "iterations": 20,
            "alpha": 40.0,
        },
    ),
    # --- ALS: does it want longer to converge? --------------------------
    (
        "als",
        {
            "factors": 256,
            "regularization": 0.1,
            "iterations": 40,
            "alpha": 40.0,
        },
    ),
    # --- BPR: fair tuning pass ------------------------------------------
    (
        "bpr",
        {
            "factors": 128,
            "regularization": 0.01,
            "iterations": 100,
            "learning_rate": 0.01,
        },
    ),
    (
        "bpr",
        {
            "factors": 128,
            "regularization": 0.001,
            "iterations": 200,
            "learning_rate": 0.05,
        },
    ),
    (
        "bpr",
        {
            "factors": 256,
            "regularization": 0.01,
            "iterations": 300,
            "learning_rate": 0.02,
        },
    ),
    # --- LMF: fair tuning pass ------------------------------------------
    (
        "lmf",
        {
            "factors": 64,
            "regularization": 0.6,
            "iterations": 30,
            "learning_rate": 1.0,
        },
    ),
    (
        "lmf",
        {
            "factors": 128,
            "regularization": 0.1,
            "iterations": 50,
            "learning_rate": 0.5,
        },
    ),
    # --- Neighbourhood baselines: comparable, never promotable ----------
    (
        "bm25",
        {
            "K": 100,
            "K1": 1.2,
            "B": 0.75,
        },
    ),
    (
        "bm25",
        {
            "K": 300,
            "K1": 1.2,
            "B": 0.75,
        },
    ),
    (
        "bm25",
        {
            "K": 1000,
            "K1": 1.2,
            "B": 0.3,
        },
    ),
    (
        "cosine",
        {
            "K": 300,
        },
    ),
]


class SweepError(Exception):
    """Raised when the sweep cannot run."""


def load_split(
    data_path: str = DATA_PATH,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
    int,
]:
    """
    Load interactions and build the project's common evaluation split.

    Every family is scored on this one split so the leaderboard compares
    models rather than comparing protocols.
    """
    df = pd.read_parquet(data_path)

    print(f"Interactions: {len(df):,}")

    train_df, test_df = (
        chronological_train_test_split(df)
    )

    validate_split(
        train_df,
        test_df,
    )

    num_users = int(
        df["user_idx"].nunique()
    )

    num_items = int(
        df["item_idx"].nunique()
    )

    print(
        f"Users: {num_users:,}  "
        f"Items: {num_items:,}"
    )

    print(
        f"Train: {len(train_df):,}  "
        f"Test: {len(test_df):,}"
    )

    return (
        train_df,
        test_df,
        num_users,
        num_items,
    )


def build_candidate_record(
    family: str,
    slug: str,
    params: dict[str, Any],
    metrics: dict[str, float | int],
    training_time_seconds: float,
    artifact_path: Path,
    has_factors: bool,
    embedding_dimension: int | None,
    mlflow_metadata: dict[str, Any],
    num_users: int,
    num_items: int,
    data_path: str = DATA_PATH,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Assemble the record that describes one trained candidate.

    ``deployable`` is the family's capability ANDed with whether this
    particular fit actually produced factors. A family declared servable
    that somehow yields no factors must not be promotable, because the
    FAISS index could not be built from it.
    """
    definition = get_family(family)

    deployable = bool(
        definition.deployable
        and has_factors
    )

    record: dict[str, Any] = {
        "name": slug,
        "family": family,
        "model_type": family,
        "slug": slug,
        "deployable": deployable,
        "retrieval": (
            "faiss" if deployable else None
        ),
        "top_k": top_k,
        "params": dict(params),
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if key != "users_evaluated"
        },
        "users_evaluated": int(
            metrics["users_evaluated"]
        ),
        "training_time_seconds": float(
            training_time_seconds
        ),
        "embedding_dimension": (
            embedding_dimension
        ),
        "num_users": num_users,
        "num_items": num_items,
        "artifacts": {
            "model": str(artifact_path),
        },
        "dataset": {
            "interactions": data_path,
        },
        "mlflow": dict(mlflow_metadata),
        "description": definition.description,
    }

    if not deployable:
        record["deployable_reason"] = (
            definition.deployable_reason
            or (
                "This fit produced no user/item factors, so no FAISS "
                "index can be built from it."
            )
        )

    return record


def write_candidate(
    directory: Path,
    record: dict[str, Any],
) -> None:
    """
    Write the two JSON records that make a candidate directory readable.

    ``training_run.json`` is the full provenance of the fit;
    ``evaluation.json`` is what model selection consumes. They are written
    separately so selection depends on a narrow contract rather than on
    everything the trainer happened to record.
    """
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        directory / "training_run.json"
    ).write_text(
        json.dumps(record, indent=2) + "\n"
    )

    selection_view = {
        key: record[key]
        for key in (
            "name",
            "family",
            "model_type",
            "slug",
            "deployable",
            "retrieval",
            "top_k",
            "params",
            "metrics",
            "users_evaluated",
            "embedding_dimension",
            "artifacts",
            "dataset",
            "mlflow",
        )
    }

    if "deployable_reason" in record:
        selection_view[
            "deployable_reason"
        ] = record["deployable_reason"]

    (
        directory / "evaluation.json"
    ).write_text(
        json.dumps(
            selection_view,
            indent=2,
        )
        + "\n"
    )


def run_variant(
    family: str,
    params: dict[str, Any],
    train_matrix: Any,
    test_users: np.ndarray,
    ground_truth: dict[int, set[int]],
    num_users: int,
    num_items: int,
    candidates_dir: Path = CANDIDATES_DIR,
    top_k: int = TOP_K,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """
    Train, evaluate and persist a single variant.

    The MLflow child run is opened here rather than by the caller so that a
    variant which raises still closes its own run.
    """
    definition = get_family(family)

    slug = variant_slug(family, params)

    fit_params = dict(params)

    if definition.accepts_random_state:
        fit_params["random_state"] = seed

    print(f"\n--- {slug} ---")

    model, training_time = fit_model(
        family,
        fit_params,
        train_matrix,
    )

    print(
        f"Fit: {training_time:.1f}s"
    )

    recommendations = recommend_batch(
        model=model,
        user_ids=test_users,
        user_item_matrix=train_matrix,
        k=top_k,
    )

    metrics = evaluate_top_k(
        recommendations=recommendations,
        ground_truth=ground_truth,
        k=top_k,
    )

    factors = extract_factors(model)

    embedding_dimension = (
        int(factors[1].shape[1])
        if factors is not None
        else None
    )

    directory = (
        candidates_dir / slug
    )

    artifact_path = (
        directory / "model.npz"
    )

    save_model(model, artifact_path)

    print(
        f"{PRIMARY_METRIC}="
        f"{metrics[PRIMARY_METRIC]:.6f}  "
        f"ndcg_at_{top_k}="
        f"{metrics[f'ndcg_at_{top_k}']:.6f}  "
        f"deployable="
        f"{definition.deployable and factors is not None}"
    )

    return build_candidate_record(
        family=family,
        slug=slug,
        params=params,
        metrics=metrics,
        training_time_seconds=training_time,
        artifact_path=artifact_path,
        has_factors=factors is not None,
        embedding_dimension=(
            embedding_dimension
        ),
        mlflow_metadata={},
        num_users=num_users,
        num_items=num_items,
        top_k=top_k,
    )


def log_variant_to_mlflow(
    record: dict[str, Any],
    directory: Path,
    artifact_path: Path,
    experiment_name: str = EXPERIMENT_NAME,
) -> bool:
    """
    Track one variant, writing its lineage to disk before uploading anything.

    The order below is the contract, not a style preference. An earlier
    version of this function uploaded the ``.npz`` inside the run and
    returned the run identity only afterwards, which is the exact shape of
    the failure that marked run 765681b7 FAILED and sent five downstream
    Airflow tasks to ``upstream_failed``. In a sweep it is worse: one failed
    upload would take out every remaining candidate.

    So:

    1. open the run and capture its identity
    2. write the candidate records to disk — evaluation.json is what
       selection reads, so it must exist before any network call that can
       fail
    3. log params and metrics
    4. upload artifacts, which cannot fail the run
    5. tag whether they landed

    Registration is deliberately absent: promoting a model is a decision, and
    it belongs to ``src/deployment/publish_model.py`` after selection has
    picked a winner.

    Returns
    -------
    bool
        True if every artifact reached MLflow.
    """
    import mlflow

    with mlflow.start_run(
        run_name=record["slug"],
        nested=True,
    ) as run:

        # Captured first. This is the thing whose loss caused the incident.
        record["mlflow"] = run_identity(
            run,
            experiment=experiment_name,
            run_name=record["slug"],
        )

        # Disk before network. Everything downstream reads these files.
        write_candidate(directory, record)

        mlflow.set_tags(
            {
                "model_family": record[
                    "family"
                ],
                "deployable": str(
                    record["deployable"]
                ).lower(),
                "retrieval": (
                    record["retrieval"]
                    or "none"
                ),
                "stage": "candidate",
                "dataset": record[
                    "dataset"
                ]["interactions"],
            }
        )

        mlflow.log_params(
            {
                "family": record["family"],
                **record["params"],
            }
        )

        mlflow.log_metrics(
            {
                **record["metrics"],
                "training_time_seconds": (
                    record[
                        "training_time_seconds"
                    ]
                ),
            }
        )

        if record["embedding_dimension"]:
            mlflow.log_param(
                "embedding_dimension",
                record[
                    "embedding_dimension"
                ],
            )

        uploaded = log_artifacts_safely(
            [
                artifact_path,
                directory
                / "training_run.json",
                directory
                / "evaluation.json",
            ]
        )

        mark_artifacts_uploaded(uploaded)

        report_upload_outcome(
            uploaded,
            what=(
                f"Artifacts for {record['slug']}"
            ),
        )

        return uploaded


def track_variant(
    record: dict[str, Any],
    directory: Path,
    use_mlflow: bool,
) -> None:
    """
    Persist one variant's records, tracking it when MLflow is in use.

    The candidate is written to disk whatever happens to tracking. A sweep
    that loses its tracking server must still leave twenty usable candidates
    behind for selection, so a tracking failure degrades this variant rather
    than ending the sweep — the same reasoning that makes artifact uploads
    non-fatal, applied one level out.
    """
    if not use_mlflow:
        write_candidate(directory, record)

        return

    try:
        log_variant_to_mlflow(
            record=record,
            directory=directory,
            artifact_path=Path(
                record["artifacts"]["model"]
            ),
        )
    except Exception as error:
        print(
            f"WARNING: MLflow tracking failed for "
            f"{record['slug']} "
            f"({type(error).__name__}: {error}). "
            "The candidate is still on disk and remains "
            "selectable."
        )

        record.setdefault("mlflow", {})

        # write_candidate overwrites, so this is safe whether or not the
        # tracked path already wrote these files.
        write_candidate(directory, record)


def write_leaderboard(
    records: list[dict[str, Any]],
    path: Path = LEADERBOARD_PATH,
    primary_metric: str = PRIMARY_METRIC,
) -> None:
    """Write every candidate's score, best first."""
    ordered = sorted(
        records,
        key=lambda record: -record[
            "metrics"
        ].get(primary_metric, 0.0),
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "primary_metric": (
                    primary_metric
                ),
                "candidates": [
                    {
                        "slug": record["slug"],
                        "family": record[
                            "family"
                        ],
                        "deployable": record[
                            "deployable"
                        ],
                        "params": record[
                            "params"
                        ],
                        "metrics": record[
                            "metrics"
                        ],
                        "training_time_seconds": (
                            record[
                                "training_time_seconds"
                            ]
                        ),
                    }
                    for record in ordered
                ],
            },
            indent=2,
        )
        + "\n"
    )


def resolve_sweep(
    families: list[str] | None,
    grid_path: str | None,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Decide which variants to run.

    A ``--grid`` file wins over the declared default so a sweep can be
    reproduced or narrowed without editing code — useful when re-running
    one variant against the tracking server.
    """
    if grid_path:
        entries = json.loads(
            Path(grid_path).read_text()
        )

        sweep = [
            (
                entry["family"],
                entry["params"],
            )
            for entry in entries
        ]
    else:
        sweep = list(DEFAULT_SWEEP)

    if families:
        wanted = set(families)

        sweep = [
            (family, params)
            for family, params in sweep
            if family in wanted
        ]

    if not sweep:
        raise SweepError(
            "The sweep is empty. Check --families against the "
            "registered families."
        )

    return sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a grid of recommendation models."
        )
    )

    parser.add_argument(
        "--families",
        help=(
            "Comma-separated families to include, e.g. 'als,bpr'. "
            "Defaults to every family in the declared sweep."
        ),
    )

    parser.add_argument(
        "--grid",
        help=(
            "JSON file of [{\"family\": ..., \"params\": {...}}] "
            "replacing the declared sweep."
        ),
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help=(
            "Skip MLflow tracking. Use offline or in CI."
        ),
    )

    parser.add_argument(
        "--primary-metric",
        default=PRIMARY_METRIC,
        help=(
            "Metric reported as the sweep winner."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Cut-off for the ranking metrics.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    families = (
        [
            name.strip()
            for name in args.families.split(",")
            if name.strip()
        ]
        if args.families
        else None
    )

    sweep = resolve_sweep(
        families,
        args.grid,
    )

    print("=" * 60)
    print("Model Sweep")
    print("=" * 60)

    print(
        f"\nVariants: {len(sweep)}"
    )

    print("\n=== Loading Data ===")

    (
        train_df,
        test_df,
        num_users,
        num_items,
    ) = load_split()

    train_matrix = build_user_item_matrix(
        train_df,
        num_users=num_users,
        num_items=num_items,
    )

    test_users = (
        test_df["user_idx"]
        .drop_duplicates()
        .astype(int)
        .to_numpy()
    )

    ground_truth = (
        test_df
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    use_mlflow = not args.no_mlflow

    if use_mlflow:
        import mlflow

        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI"
        )

        if tracking_uri:
            mlflow.set_tracking_uri(
                tracking_uri
            )

        mlflow.set_experiment(
            EXPERIMENT_NAME
        )

        print(
            f"\nMLflow tracking URI: "
            f"{mlflow.get_tracking_uri()}"
        )

    print("\n=== Training ===")

    records: list[dict[str, Any]] = []

    def run_all() -> None:
        for family, params in sweep:
            record = run_variant(
                family=family,
                params=params,
                train_matrix=train_matrix,
                test_users=test_users,
                ground_truth=ground_truth,
                num_users=num_users,
                num_items=num_items,
                top_k=args.top_k,
            )

            track_variant(
                record=record,
                directory=(
                    CANDIDATES_DIR
                    / record["slug"]
                ),
                use_mlflow=use_mlflow,
            )

            records.append(record)

    if use_mlflow:
        import mlflow

        with mlflow.start_run(
            run_name=(
                f"sweep-{len(sweep)}-variants"
            )
        ) as parent:

            print(
                f"Parent run: "
                f"{parent.info.run_id}"
            )

            mlflow.set_tags(
                {
                    "sweep": "true",
                    "stage": "sweep",
                    "variants": str(
                        len(sweep)
                    ),
                }
            )

            run_all()

            best = max(
                records,
                key=lambda record: record[
                    "metrics"
                ].get(
                    args.primary_metric,
                    0.0,
                ),
            )

            # Recording the winner on the parent means the experiment page
            # answers "what did this sweep conclude" without sorting the
            # children by hand.
            mlflow.set_tags(
                {
                    "best_slug": best["slug"],
                    "best_family": best[
                        "family"
                    ],
                }
            )

            mlflow.log_metric(
                f"best_{args.primary_metric}",
                float(
                    best["metrics"][
                        args.primary_metric
                    ]
                ),
            )
    else:
        print(
            "MLflow tracking skipped "
            "(--no-mlflow)."
        )

        run_all()

    write_leaderboard(
        records,
        primary_metric=args.primary_metric,
    )

    print("\n=== Leaderboard ===")

    for record in sorted(
        records,
        key=lambda record: -record[
            "metrics"
        ].get(
            args.primary_metric,
            0.0,
        ),
    ):
        print(
            f"{record['slug']:<34} "
            f"{args.primary_metric}="
            f"{record['metrics'][args.primary_metric]:.6f} "
            f"deployable="
            f"{str(record['deployable']):<5} "
            f"fit="
            f"{record['training_time_seconds']:.1f}s"
        )

    print(
        f"\nLeaderboard written to: "
        f"{LEADERBOARD_PATH}"
    )

    print(
        f"Candidates written to: "
        f"{CANDIDATES_DIR}/"
    )

    print("\n=== Sweep Complete ===")


if __name__ == "__main__":
    main()
