# Shared configuration for the MLflow tracking server on Google Cloud.
# Sourced by provision.sh and deploy.sh. Override any value by exporting it first,
# e.g. REGION=asia-south2 ./deploy.sh

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-south1}"          # Mumbai

if [ -z "${PROJECT}" ] || [ "${PROJECT}" = "(unset)" ]; then
  echo "ERROR: no GCP project set. Run: gcloud config set project <PROJECT_ID>" >&2
  exit 1
fi

SERVICE="mlflow"                          # Cloud Run service name
SQL_INSTANCE="mlflow-pg"                  # Cloud SQL instance name
DB_NAME="mlflow"
DB_SECRET="mlflow-db-uri"                 # Secret Manager secret holding the full DSN
SA_NAME="mlflow-server"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
BUCKET="${PROJECT}-mlflow"                # artifacts/ and (later) dvc/ live here
SQL_CONNECTION="${PROJECT}:${REGION}:${SQL_INSTANCE}"
ARTIFACT_ROOT="gs://${BUCKET}/artifacts"
