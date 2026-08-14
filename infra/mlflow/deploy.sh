#!/usr/bin/env bash
# Build and deploy the MLflow tracking server to Cloud Run. Re-run after any change
# to the Dockerfile; each run creates a new revision and shifts traffic to it.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=config.sh
source ./config.sh

echo "==> Deploying ${SERVICE} to Cloud Run (${REGION})"
gcloud run deploy "${SERVICE}" \
  --source . \
  --region="${REGION}" \
  --service-account="${SA_EMAIL}" \
  --add-cloudsql-instances="${SQL_CONNECTION}" \
  --set-secrets="MLFLOW_BACKEND_URI=${DB_SECRET}:latest" \
  --set-env-vars="MLFLOW_ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  --cpu=1 --memory=1Gi \
  --min-instances=0 --max-instances=2 \
  --timeout=600 \
  --no-allow-unauthenticated

URL="$(gcloud run services describe "${SERVICE}" --region="${REGION}" \
        --format='value(status.url)')"

cat <<EOF

Deployed: ${URL}
  (not publicly reachable - IAM-authenticated only)

Open the UI and point clients at it:

  gcloud run services proxy ${SERVICE} --region=${REGION} --port=8080
  export MLFLOW_TRACKING_URI=http://localhost:8080

Then smoke-test:

  python infra/mlflow/smoke_test.py
EOF
