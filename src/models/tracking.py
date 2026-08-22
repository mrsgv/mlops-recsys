"""
Shared MLflow tracking helpers for every training entry point.

Why this module exists
----------------------
Run ``765681b70f16451ea79ddd222e35ef8f`` in the ``ials`` experiment is marked
FAILED even though training fully succeeded: it holds the correct
``recall_at_10`` and the model reached disk. It died uploading a 30 MB
``.npz`` to MLflow. Because the lineage record was written *after* that
upload, the record was never written at all, the run was marked FAILED, and
because the Airflow tasks are chained on exit status all five downstream
tasks went ``upstream_failed``. Two of the four runs in that experiment died
the same way.

The fix is two rules, and they live here rather than in each trainer so a new
model family cannot reintroduce the bug:

1. **An artifact upload must never fail a run.** DVC is the system of record
   for model files, so the copy MLflow holds is a convenience. Losing it must
   degrade the run, not destroy it.

2. **Ordering is the contract.** Anything a downstream stage reads — the run
   record, the leaderboard, the promoted-model pointer — must be on disk
   *before* any network call. In particular the run identity is captured
   before uploading, not after, so a failed upload cannot take the lineage
   with it.

A run whose artifacts did not land is tagged ``artifacts_uploaded=false`` so
it is identifiable in the UI rather than silently incomplete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


ARTIFACTS_UPLOADED_TAG = "artifacts_uploaded"


def log_artifact_safely(
    path: str | Path,
) -> bool:
    """
    Upload one artifact to MLflow without letting a transport failure end
    the run.

    Returns
    -------
    bool
        True if the artifact was uploaded.
    """
    try:
        mlflow.log_artifact(str(path))

        return True

    # Deliberately broad: the tracking server is reached over the network
    # through an authenticated proxy, and no useful purpose is served by
    # enumerating every transport, credential and storage-backend error that
    # can end a 30 MB upload.
    except Exception as error:
        print(
            f"WARNING: could not upload {path} to MLflow "
            f"({type(error).__name__}: {error})"
        )

        return False


def log_artifacts_safely(
    paths: list[str | Path],
) -> bool:
    """
    Upload several artifacts, reporting whether all of them landed.

    Every path is attempted even if an earlier one fails, so a single
    unreachable upload does not hide the state of the rest.
    """
    outcomes = [
        log_artifact_safely(path)
        for path in paths
    ]

    return all(outcomes)


def mark_artifacts_uploaded(
    uploaded: bool,
) -> None:
    """
    Record whether this run's artifacts reached MLflow.

    Must be called inside an active run. Tagging rather than failing is what
    keeps an incomplete upload visible without stopping the pipeline.
    """
    mlflow.set_tag(
        ARTIFACTS_UPLOADED_TAG,
        "true" if uploaded else "false",
    )


def run_identity(
    run: Any,
    experiment: str,
    run_name: str,
) -> dict[str, str]:
    """
    Capture the run's identity for the lineage record.

    Called immediately after the run opens and before anything is uploaded,
    because this is precisely what was lost when an upload failure aborted
    the function that was going to write it.

    Downstream stages read this instead of querying MLflow, which keeps them
    runnable when the tracking server is unreachable.
    """
    return {
        "run_id": run.info.run_id,
        "experiment": experiment,
        "run_name": run_name,
        "tracking_uri": (
            mlflow.get_tracking_uri()
        ),
    }


def report_upload_outcome(
    uploaded: bool,
    what: str = "Artifacts",
) -> None:
    """
    Explain a partial upload on stdout.

    Airflow captures task stdout, so this is how the degradation becomes
    visible to whoever reads the task log without the task going red.
    """
    if uploaded:
        return

    print(
        f"\n{what} were not uploaded to MLflow. Training results and "
        "the run record are complete; the model file remains versioned "
        "by DVC."
    )
