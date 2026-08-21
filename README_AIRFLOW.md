# Airflow — Recommendation Pipeline

Orchestrates the full offline path: validate → preprocess → train → evaluate →
select → index → manifest.

## Why Airflow gets its own virtualenv

Airflow pins large parts of the dependency tree (pydantic, fsspec and others).
Resolving those pins against `pyspark`, `torch` and `mlflow` produces a slow,
fragile install. The DAG avoids the problem entirely: every task shells out to a
project module, so **Airflow never imports project code** and the two
environments stay independent.

```bash
python -m venv .venv-airflow
source .venv-airflow/bin/activate

AIRFLOW_VERSION=3.3.1
PYTHON_VERSION=3.12
pip install -r requirements-airflow.txt \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

The constraint file is what makes an Airflow install reproducible — don't install
without it.

## Configuration

Two environment variables connect Airflow to the project:

| Variable | Default | Purpose |
|---|---|---|
| `RECSYS_PROJECT_ROOT` | the repo containing the DAG | Working directory for every task. The scripts resolve data and model paths relative to it, so a wrong value silently reads and writes the wrong files. |
| `RECSYS_PYTHON` | `<root>/.venv/bin/python` | Interpreter that has the project requirements. |

Anything else exported for the scheduler is inherited by tasks —
`MLFLOW_TRACKING_URI` in particular. Note that **nothing in the codebase calls
`load_dotenv()`**, so a `.env` file is ignored: export the variable, or training
logs to a local `./mlruns` while appearing to succeed.

## Prerequisites

```bash
# 1. data and model artifacts (DVC remote is the shared Google Drive folder)
dvc pull

# 2. MLflow tracking server, reached through the authenticated proxy
gcloud run services proxy mlflow --region=asia-south1 --port=8081
export MLFLOW_TRACKING_URI=http://localhost:8081

# 3. Spark preprocessing needs a JVM
java -version
```

## The pipeline

```
validate_raw_data
  -> preprocess
  -> validate_processed_data
  -> train_ials
  -> evaluate_ials
  -> select_model
  -> build_faiss
  -> build_deployment_manifest
```

| Task | Module | Produces |
|---|---|---|
| `validate_raw_data` | `src.data.validate_data --stage raw` | fails fast on a bad download, before Spark burns time |
| `preprocess` | `src.preprocessing.preprocess_video_games` | `data/processed/video_games.parquet`, `item_mapping.parquet` |
| `validate_processed_data` | `src.data.validate_data --stage processed` | enforces contiguous indices, no duplicates, aligned mapping |
| `train_ials` | `src.models.train_ials` | `models/ials/ials_model.npz`, `training_run.json`, MLflow run |
| `evaluate_ials` | `src.evaluation.evaluate_ials` | `models/ials/evaluation.json`, `eval_*` metrics on the run |
| `select_model` | `src.deployment.select_model` | `models/deployment/selected_model.json` |
| `build_faiss` | `src.retrieval.build_index` | `models/retrieval/faiss.index`, `index_metadata.json` |
| `build_deployment_manifest` | `src.deployment.build_manifest` | `models/deployment/deployment_manifest.json` |

Two tasks go beyond the six-step outline, for reasons worth keeping:
`validate_processed_data` is the same script as the raw check and gates the
training input; `build_deployment_manifest` must run after `build_faiss` because
the manifest records the index it ships with.

`retries` is 0 by design — during a build, a failure that a retry quietly papers
over is worse than a visible one.

## Running it

```bash
source .venv-airflow/bin/activate
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/airflow/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False

airflow db migrate            # first time only
airflow dags reserialize      # register the DAG in the metadata DB
airflow dags list             # expect: recommendation_pipeline
airflow tasks list recommendation_pipeline
```

Run one task, for real:

```bash
airflow tasks test recommendation_pipeline validate_processed_data
```

Run the whole DAG:

```bash
airflow dags test recommendation_pipeline
```

`airflow dags test` executes everything in-process, including Spark preprocessing
and training, so expect it to take a while. The DAG has no schedule
(`schedule=None`) and is triggered deliberately — a schedule would kick off
training runs during the build.

## What the deployment manifest is for

`models/deployment/deployment_manifest.json` is the hand-off to serving. Its
`serving_env` block holds exactly the environment variables
`src/serving/config.py` reads, so a Cloud Run service can be configured from the
manifest instead of hardcoding artifact paths:

```json
"serving_env": {
  "MODEL_TYPE": "ials",
  "MODEL_VERSION": "ials-61ad8d28",
  "IALS_MODEL_PATH": "models/ials/ials_model.npz",
  "FAISS_INDEX_PATH": "models/retrieval/faiss.index",
  "FAISS_METADATA_PATH": "models/retrieval/index_metadata.json",
  "ITEM_MAPPING_PATH": "data/processed/item_mapping.parquet",
  "INTERACTIONS_PATH": "data/processed/video_games.parquet"
}
```

`model.version` is derived from the artifact's DVC content hash, so it is
deterministic and resolvable back to an exact artifact. An artifact that is not
DVC-tracked is reported as `ials-unversioned` with a warning rather than given a
fake version.

The `bundle` list names every file Cloud Run needs. Two of them —
`video_games.parquet` and `item_mapping.parquet` — are **directories**, because
Spark writes part files; copy them recursively.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `dvc pull` reports missing files | artifacts not pushed by whoever produced them |
| Task fails with `FileNotFoundError` on a data path | wrong working directory — check `RECSYS_PROJECT_ROOT` |
| Training succeeds but no run appears in MLflow | `MLFLOW_TRACKING_URI` not exported; the run went to `./mlruns` |
| `dags list` shows nothing | run `airflow dags reserialize` — Airflow 3 lists from the metadata DB |
| Spark task fails immediately | no JVM on `PATH` |
