"""
Recommendation pipeline DAG.

The DAG is deliberately thin: every task shells out to a script that is
independently runnable and independently tested. Training logic never lives
here, so a broken Airflow environment can never be confused with a broken
model, and any step can be reproduced by hand from the command line.

    validate_raw_data
        -> preprocess
        -> validate_processed_data
        -> train_ials
        -> evaluate_ials
        -> select_model
        -> build_faiss
        -> build_deployment_manifest

Configuration comes from two environment variables, because Airflow
normally runs in its own virtual environment while the pipeline needs the
project's ML dependencies:

RECSYS_PROJECT_ROOT
    Repository root. Tasks run with this as their working directory, since
    the scripts resolve data and model paths relative to it. Defaults to
    the repository this file lives in.

RECSYS_PYTHON
    Interpreter used to run the scripts. Defaults to the project's
    ``.venv/bin/python``. Point it at whichever environment has the
    project requirements installed.

Any other environment variable set for the scheduler — MLFLOW_TRACKING_URI
in particular — is inherited by every task.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG

try:
    # Airflow 3 moved the operator into the standard provider package.
    from airflow.providers.standard.operators.bash import (
        BashOperator,
    )
except ImportError:  # pragma: no cover - Airflow 2 fallback
    from airflow.operators.bash import (
        BashOperator,
    )


DEFAULT_PROJECT_ROOT = str(
    Path(__file__).resolve().parents[2]
)

PROJECT_ROOT = os.environ.get(
    "RECSYS_PROJECT_ROOT",
    DEFAULT_PROJECT_ROOT,
)

PYTHON_BIN = os.environ.get(
    "RECSYS_PYTHON",
    str(
        Path(PROJECT_ROOT)
        / ".venv"
        / "bin"
        / "python"
    ),
)

DAG_ID = "recommendation_pipeline"

# Failures should surface immediately during the build rather than being
# masked by a retry that quietly succeeds on a second attempt.
DEFAULT_ARGS = {
    "owner": "mlops-recsys",
    "retries": 0,
    "depends_on_past": False,
}

# Each entry becomes one task: (task_id, module, arguments, description).
PIPELINE_STEPS = [
    (
        "validate_raw_data",
        "src.data.validate_data",
        "--stage raw",
        "Fail fast on a bad raw download before Spark runs.",
    ),
    (
        "preprocess",
        "src.preprocessing.preprocess_video_games",
        "",
        "Spark preprocessing: clean, encode IDs, write Parquet.",
    ),
    (
        "validate_processed_data",
        "src.data.validate_data",
        "--stage processed",
        (
            "Enforce the contract training and serving depend on: "
            "contiguous indices, no duplicates, aligned item mapping."
        ),
    ),
    (
        "train_ials",
        "src.models.train_ials",
        "",
        "Train iALS and log params, metrics and the run ID to MLflow.",
    ),
    (
        "evaluate_ials",
        "src.evaluation.evaluate_ials",
        "",
        "Evaluate the saved artifact and record its metrics.",
    ),
    (
        "select_model",
        "src.deployment.select_model",
        "",
        "Rank candidates and promote the best deployable model.",
    ),
    (
        "build_faiss",
        "src.retrieval.build_index",
        "",
        "Build the FAISS index from the promoted model's item factors.",
    ),
    (
        "build_deployment_manifest",
        "src.deployment.build_manifest",
        "",
        "Record model, run, dataset and artifact versions for serving.",
    ),
]


def build_command(
    module: str,
    arguments: str,
) -> str:
    """Build the shell command for one pipeline step."""
    command = f"{PYTHON_BIN} -m {module}"

    if arguments:
        command = f"{command} {arguments}"

    return command


with DAG(
    dag_id=DAG_ID,
    description=(
        "Validate, preprocess, train, evaluate, select and index the "
        "Video Games recommendation model."
    ),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 8, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=[
        "recsys",
        "ials",
        "faiss",
    ],
) as dag:

    dag.doc_md = __doc__

    previous_task = None

    for (
        task_id,
        module,
        arguments,
        description,
    ) in PIPELINE_STEPS:

        task = BashOperator(
            task_id=task_id,
            bash_command=build_command(
                module,
                arguments,
            ),
            cwd=PROJECT_ROOT,
            doc_md=description,
        )

        if previous_task is not None:
            previous_task >> task

        previous_task = task
