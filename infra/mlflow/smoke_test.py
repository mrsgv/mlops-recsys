"""End-to-end check of the tracking server.

Proves three things at once: run metadata reaches Cloud SQL, the artifact lands in
GCS, and the artifact URI is a real gs:// path rather than a proxied
mlflow-artifacts:/ one (which would mean --no-serve-artifacts didn't take effect).

Usage, with `gcloud run services proxy` running in another terminal:
    export MLFLOW_TRACKING_URI=http://localhost:8080
    python infra/mlflow/smoke_test.py
"""

import os

import mlflow

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise SystemExit("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("smoke-test")

with mlflow.start_run(run_name="hello") as run:
    mlflow.log_param("where", "cloud-run")
    mlflow.log_metric("ndcg_at_10", 0.42)
    mlflow.log_text("artifact round-trip ok", "hello.txt")

    artifact_uri = mlflow.get_artifact_uri()
    print(f"tracking_uri : {tracking_uri}")
    print(f"run_id       : {run.info.run_id}")
    print(f"artifact_uri : {artifact_uri}")

if not artifact_uri.startswith("gs://"):
    raise SystemExit(
        f"expected a gs:// artifact URI, got {artifact_uri!r} - the server is still "
        "proxying artifacts, so large uploads will fail at Cloud Run's 32 MiB limit"
    )

print("\nOK - metadata in Cloud SQL, artifact in GCS.")
