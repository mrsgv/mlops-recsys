# MLflow Quickstart — Exact Commands

Copy-paste setup for the shared MLflow tracking server. Every command is literal —
nothing to fill in except picking your OS.

For architecture, cost, and admin details see [README_MLFLOW.md](README_MLFLOW.md).

Pull the latest code first:

```bash
git pull origin main
```

---

## Step 1 — Install the gcloud CLI

**macOS**

```bash
brew install --cask gcloud-cli
```

**Windows (PowerShell)**

```powershell
winget install Google.CloudSDK
```

Then close and reopen PowerShell so `gcloud` lands on PATH.

**Linux**

```bash
sudo snap install google-cloud-cli --classic
```

If you install via apt instead, jump to the [apt note](#linux-apt-note) — the proxy
component is unavailable on deb installs.

---

## Step 2 — Authenticate (identical on every OS)

```bash
gcloud init
```

Sign in, choose project **`mlops-project-505217`**, skip the default region prompt. Then:

```bash
gcloud config set project mlops-project-505217
gcloud auth application-default login
gcloud auth application-default set-quota-project mlops-project-505217
```

**Both login commands are required.** `gcloud init` authenticates the CLI;
`application-default login` authenticates Python libraries. Skip the second and your
metrics will log fine while model uploads fail with a GCS 403.

Verify — expect `billingEnabled: true`:

```bash
gcloud billing projects describe mlops-project-505217
```

---

## Step 3 — Install the proxy component

```bash
gcloud components install cloud-run-proxy
```

**macOS only, and required.** Homebrew doesn't symlink component binaries, so without
this you get `cloud-run-proxy binary not installed`:

```bash
echo 'export PATH=/opt/homebrew/share/google-cloud-sdk/bin:$PATH' >> ~/.zshrc
exec zsh
```

---

## Step 4 — Python environment (Python 3.12)

**macOS / Linux**

```bash
cd mlops-recsys
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
cd mlops-recsys
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Then on every OS — the first command pulls pyspark and takes a while:

```bash
pip install -r requirements.txt
pip install mlflow==3.1.4 "google-cloud-storage>=2.19,<4"
```

This **downgrades pandas 3.0.5 → 2.3.3** because MLflow requires `pandas<3`. That is
intentional. Do not reinstall pandas 3; it breaks MLflow.

---

## Step 5 — Start the proxy

Terminal 1, all platforms. It prints a localhost URL and then sits there — that's
correct. Leave it open for your whole session.

```bash
gcloud run services proxy mlflow --region=asia-south1 --port=8080
```

---

## Step 6 — Point your code at it and verify

Terminal 2.

**macOS / Linux**

```bash
cd mlops-recsys
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://localhost:8080
python infra/mlflow/smoke_test.py
```

**Windows (PowerShell)**

```powershell
cd mlops-recsys
.venv\Scripts\Activate.ps1
$env:MLFLOW_TRACKING_URI = "http://localhost:8080"
python infra/mlflow/smoke_test.py
```

Expected tail of the output:

```
artifact_uri : gs://mlops-project-505217-mlflow/artifacts/...
OK - metadata in Cloud SQL, artifact in GCS.
```

The first call can take a few seconds while Cloud Run cold-starts. Normal.

---

## Step 7 — Open the UI

<http://localhost:8080> — you should see the `smoke-test` experiment.

---

## Using it in your own code

Set `MLFLOW_TRACKING_URI` in your shell as in Step 6, then:

```python
import mlflow

mlflow.set_experiment("recsys")

with mlflow.start_run(run_name="svd-50"):
    mlflow.log_params({"model": "svd", "factors": 50})
    mlflow.log_metrics({"ndcg_at_10": 0.0201})
```

Never hardcode the tracking URI — always read it from the environment, so the same
script works against the server or a local store.

To work offline with no cloud at all:

```bash
export MLFLOW_TRACKING_URI=file:./mlruns
```

---

## Linux apt note

Debian/RPM installs of gcloud disable `gcloud components install`, so the proxy can't be
installed. Skip Steps 3 and 5, and replace Step 6 with:

```bash
export MLFLOW_TRACKING_URI=https://mlflow-23838349595.asia-south1.run.app
export MLFLOW_TRACKING_TOKEN=$(gcloud auth print-identity-token)
python infra/mlflow/smoke_test.py
```

The token expires after roughly an hour; re-run that command when it does. This works for
training but gives no browser UI.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cloud-run-proxy binary not installed` | macOS PATH fix in Step 3 |
| `403 Forbidden` from the proxy | Your account lacks project access |
| `Connection refused` on the first log call | Proxy isn't running — start Terminal 1 |
| GCS `403` on `log_model`, metrics fine | Run `gcloud auth application-default login` |
| `401` after about an hour (token method) | Regenerate the identity token |
| First call hangs a few seconds | Cold start, normal |

Server logs:

```bash
gcloud run services logs read mlflow --region=asia-south1 --limit=50
```

---

## Reference

| | |
|---|---|
| Project | `mlops-project-505217` |
| Region | `asia-south1` |
| Service | `https://mlflow-23838349595.asia-south1.run.app` (IAM-only, not public) |
| Cost | ~$0.45/day on trial credits, almost all Cloud SQL |

No extra IAM is needed — all three of us are Owners on the project.
