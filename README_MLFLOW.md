# MLflow Tracking Server — Setup Guide

Experiment tracking for the MLOps RecSys project. The tracking server runs on Google
Cloud, so every team member logs runs to **one shared leaderboard** instead of a local
`mlruns/` folder that nobody else can see.

Everyone trains on their own machine. The server only records results.

## Architecture

```
  your machine                        Google Cloud (asia-south1 / Mumbai)
┌──────────────┐   run metadata     ┌──────────────────────┐
│ train.py     │ ──────────────────▶│ Cloud Run: "mlflow"  │
│              │   (IAM-authed)     │ 1 vCPU · 1 GiB       │
└──────┬───────┘                    │ scales 0 → 2         │
       │                            └──────────┬───────────┘
       │ model files                           │ unix socket
       │ (direct to GCS)                       ▼
       │                            ┌──────────────────────┐
       │   ┌──────────────────┐     │ Cloud SQL PostgreSQL │
       └──▶│ GCS bucket       │     │ db-f1-micro          │
           │ …-mlflow/        │     └──────────────────────┘
           │   artifacts/     │
           └──────────────────┘
```

| Component | What it holds |
|---|---|
| Cloud Run `mlflow` | The MLflow server + web UI. Scales to zero when idle. |
| Cloud SQL PostgreSQL | Run metadata: params, metrics, tags, the model registry. |
| GCS bucket | Artifacts: model files, plots, evaluation outputs. |

Two details worth knowing. Metadata flows through the server, but **model files upload
straight from your machine to GCS** — Cloud Run caps request bodies at 32 MiB, which a
factor matrix would exceed. And the service has **no public URL**: access is
IAM-authenticated, because MLflow ships with no authentication of its own.

* Project: `mlops-project-505217`
* Region: `asia-south1`
* Service URL: `https://mlflow-23838349595.asia-south1.run.app` (not directly browsable)

## Prerequisites

* A Google account with access to the `mlops-project-505217` project. All three team
  members already have Owner, which includes permission to invoke the service.
  Anyone else needs `roles/run.invoker`.
* Python 3.12 and the project repo cloned.

---

## Step 1 — Install the Google Cloud CLI

### macOS

```bash
brew install --cask gcloud-cli
```

### Windows

Either download and run `GoogleCloudSDKInstaller.exe` from
<https://cloud.google.com/sdk/docs/install>, or:

```powershell
winget install Google.CloudSDK
```

Then reopen your terminal so `gcloud` is on PATH.

### Linux

```bash
sudo snap install google-cloud-cli --classic
```

If you prefer apt, follow <https://cloud.google.com/sdk/docs/install#deb> — but see the
caveat in Step 3, since apt installs disable `gcloud components install`.

---

## Step 2 — Authenticate (same on every OS)

```bash
gcloud init
gcloud config set project mlops-project-505217
gcloud auth application-default login
gcloud auth application-default set-quota-project mlops-project-505217
```

Pick `mlops-project-505217` when `gcloud init` lists projects. You can skip the default
region prompt.

**Both login commands are required.** `gcloud init` authenticates the CLI; `application-
default login` authenticates Python libraries. Skip the second and your metrics will log
fine but `log_model` will fail with a GCS 403 — a confusing half-working state.

Verify:

```bash
gcloud billing projects describe mlops-project-505217
```

---

## Step 3 — Install the proxy component

The proxy is how you reach an IAM-protected Cloud Run service from localhost. It handles
token refresh for you.

```bash
gcloud components install cloud-run-proxy
```

### macOS — required extra step

Homebrew only symlinks a few binaries, so gcloud won't find newly installed components
and reports `cloud-run-proxy binary not installed`. Add the SDK's own bin directory to
PATH:

```bash
echo 'export PATH=/opt/homebrew/share/google-cloud-sdk/bin:$PATH' >> ~/.zshrc
exec zsh
```

### Linux — if installed via apt

Debian/RPM installs disable `gcloud components install`. Either reinstall via snap or the
tarball, or use **Method B** in Step 5, which needs no component.

---

## Step 4 — Python dependencies

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
pip install mlflow==3.1.4 "google-cloud-storage>=2.19,<4"
```

**Expect pandas to move from 3.0.5 down to 2.3.3.** MLflow requires `pandas<3`, and
MLflow is a mandatory project component, so this is intentional — don't fight it or
reinstall pandas 3. It also pins `pyarrow<21`, `protobuf<7`, and `packaging<26`.

---

## Step 5 — Connect

### Method A — proxy (recommended: gives you the web UI)

Terminal 1, leave it running all session:

```bash
gcloud run services proxy mlflow --region=asia-south1 --port=8080
```

Terminal 2:

```bash
# macOS / Linux
export MLFLOW_TRACKING_URI=http://localhost:8080

# Windows PowerShell
$env:MLFLOW_TRACKING_URI = "http://localhost:8080"

# Windows cmd
set MLFLOW_TRACKING_URI=http://localhost:8080
```

Open <http://localhost:8080> in a browser for the UI.

### Method B — identity token (no component needed, no UI)

Useful on apt-installed Linux, in CI, or if the proxy misbehaves:

```bash
export MLFLOW_TRACKING_URI=https://mlflow-23838349595.asia-south1.run.app
export MLFLOW_TRACKING_TOKEN=$(gcloud auth print-identity-token)
```

PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "https://mlflow-23838349595.asia-south1.run.app"
$env:MLFLOW_TRACKING_TOKEN = (gcloud auth print-identity-token)
```

The token expires after about an hour, so re-run it for long sessions. Browser access
still needs Method A.

---

## Step 6 — Verify

```bash
python infra/mlflow/smoke_test.py
```

Expected:

```
artifact_uri : gs://mlops-project-505217-mlflow/artifacts/...
OK - metadata in Cloud SQL, artifact in GCS.
```

A `gs://` artifact URI confirms artifacts bypass the server correctly. If it prints an
`mlflow-artifacts:/` URI instead, stop and raise it — large uploads would fail later.

The first request after an idle period takes a few seconds while Cloud Run cold-starts.

---

## Using it from your own code

The tracking URI always comes from the environment, never hardcoded, so the same script
runs against the shared server or a local store without edits:

```python
import mlflow

mlflow.set_experiment("recsys")

with mlflow.start_run(run_name="svd-50"):
    mlflow.log_params({"model": "svd", "factors": 50, "top_k": 10})
    mlflow.log_metrics({"ndcg_at_10": 0.0201, "hit_rate_at_10": 0.0375})
```

To work offline, point at a local folder instead — no code change:

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `cloud-run-proxy binary not installed` | macOS PATH issue — see Step 3. |
| `403 Forbidden` from the proxy | Your account lacks access. Needs Owner or `roles/run.invoker`. |
| `Connection refused` on first log call | The proxy isn't running. Start Terminal 1. |
| GCS `403` on `log_model`, metrics fine | Missing ADC — run `gcloud auth application-default login`. |
| `401` after ~an hour on Method B | Token expired. Re-run `gcloud auth print-identity-token`. |
| First call hangs several seconds | Cold start. Normal, once per idle period. |

Read server logs with:

```bash
gcloud run services logs read mlflow --region=asia-south1 --limit=50
```

## Cost

Roughly **$0.45/day**, almost entirely Cloud SQL — Cloud Run scales to zero and GCS is
pennies. Charged to the shared project's trial credits.

**After the project is submitted and graded**, delete the database (stopping it still
bills for storage):

```bash
gcloud sql instances delete mlflow-pg
```

## Admin — recreating this from scratch

Everything is scripted in `infra/mlflow/`:

```bash
./infra/mlflow/provision.sh    # APIs, GCS bucket, Cloud SQL, secret, service account, IAM
./infra/mlflow/deploy.sh       # build the server image and deploy to Cloud Run
```

Both are idempotent — re-running skips resources that already exist. `config.sh` holds
every name and region in one place; override any of them with an env var, e.g.
`REGION=asia-south2 ./deploy.sh`.

The database password is generated at provision time and stored only in Secret Manager
(`mlflow-db-uri`). It is never written to the repo or passed on a command line.
