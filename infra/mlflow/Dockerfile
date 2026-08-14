# MLflow tracking server for Cloud Run.
#
# The official ghcr.io/mlflow/mlflow image ships neither a PostgreSQL driver nor the
# GCS client, so it starts fine and then fails on the first write. Hence our own image.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
      mlflow==3.1.4 \
      psycopg2-binary==2.9.10 \
      google-cloud-storage==2.19.0

# MLFLOW_BACKEND_URI  - injected from Secret Manager (contains the DB password)
# MLFLOW_ARTIFACT_ROOT - gs://<bucket>/artifacts
# PORT                - set by Cloud Run, 8080 locally
#
# --no-serve-artifacts is deliberate: clients upload artifacts straight to GCS with
# their own credentials. Proxying them through Cloud Run would cap every upload at
# its 32 MiB request limit, which a factor matrix blows past.
CMD ["/bin/sh", "-c", "exec mlflow server \
  --host 0.0.0.0 --port ${PORT:-8080} --workers 2 \
  --backend-store-uri \"${MLFLOW_BACKEND_URI}\" \
  --no-serve-artifacts --default-artifact-root \"${MLFLOW_ARTIFACT_ROOT}\""]
