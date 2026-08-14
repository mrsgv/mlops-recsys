#!/usr/bin/env bash
# One-time provisioning of everything the MLflow server needs: APIs, artifact bucket,
# Cloud SQL instance, DB password in Secret Manager, and the runtime service account.
#
# Safe to re-run: every step checks for existing resources first.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=config.sh
source ./config.sh

echo "==> Project ${PROJECT}, region ${REGION}"

echo "==> Enabling APIs (a minute or two the first time)"
gcloud services enable \
  run.googleapis.com sqladmin.googleapis.com storage.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com

echo "==> Artifact bucket gs://${BUCKET}"
if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" \
    --location="${REGION}" --uniform-bucket-level-access
else
  echo "    already exists"
fi

echo "==> Cloud SQL instance ${SQL_INSTANCE} (~10 min on first creation)"
if ! gcloud sql instances describe "${SQL_INSTANCE}" >/dev/null 2>&1; then
  # db-f1-micro + HDD + no automated backups: ~$0.40/day, which is all this needs.
  gcloud sql instances create "${SQL_INSTANCE}" \
    --database-version=POSTGRES_16 \
    --edition=enterprise \
    --tier=db-f1-micro \
    --region="${REGION}" \
    --storage-type=HDD --storage-size=10GB \
    --availability-type=zonal \
    --no-backup
else
  echo "    already exists"
fi

echo "==> Database ${DB_NAME}"
gcloud sql databases describe "${DB_NAME}" --instance="${SQL_INSTANCE}" >/dev/null 2>&1 \
  || gcloud sql databases create "${DB_NAME}" --instance="${SQL_INSTANCE}"

echo "==> DB credentials in Secret Manager (${DB_SECRET})"
if ! gcloud secrets describe "${DB_SECRET}" >/dev/null 2>&1; then
  # Hex only: the password is embedded in a URI unescaped, so no %-encoding worries.
  DB_PASS="$(openssl rand -hex 24)"
  gcloud sql users set-password postgres \
    --instance="${SQL_INSTANCE}" --password="${DB_PASS}"
  # The built-in postgres user, not a dedicated one: on PostgreSQL 15+ a non-owner
  # role has no CREATE on schema public, and MLflow's migrations need it.
  printf 'postgresql+psycopg2://postgres:%s@/%s?host=/cloudsql/%s' \
    "${DB_PASS}" "${DB_NAME}" "${SQL_CONNECTION}" \
    | gcloud secrets create "${DB_SECRET}" --data-file=- --replication-policy=automatic
  unset DB_PASS
else
  echo "    already exists (delete the secret to rotate)"
fi

echo "==> Runtime service account ${SA_EMAIL}"
gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${SA_NAME}" \
       --display-name="MLflow tracking server"

echo "==> IAM bindings"
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role=roles/cloudsql.client --condition=None >/dev/null

# `gcloud run deploy --source` builds through Cloud Build, which runs as the Compute
# Engine default service account. Projects created after early 2024 grant that account
# nothing by default, so the build fails with a 403 reading its own source upload.
# roles/cloudbuild.builds.builder covers source read + image push + log write.
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/cloudbuild.builds.builder --condition=None >/dev/null
gcloud secrets add-iam-policy-binding "${DB_SECRET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role=roles/secretmanager.secretAccessor >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role=roles/storage.objectAdmin >/dev/null

echo
echo "Provisioning done. Next: ./deploy.sh"
