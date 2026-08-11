# SVD Recommendation Baseline

This branch implements the first recommendation-system baseline for the MLOps RecSys project using the Amazon Video Games dataset.

## What is included

* Video Games dataset inspection and preprocessing
* User/item ID encoding
* Train/test split
* 50-factor SVD recommendation model
* Top-10 recommendation generation
* Previously-seen item filtering
* Offline evaluation
* DVC tracking for dataset and model artifacts
* Shared Google Drive DVC remote

## Dataset

Processed dataset:

```text
data/processed/video_games.parquet
```

Current dataset:

|              |   Count |
| ------------ | ------: |
| Interactions | 814,586 |
| Users        |  94,762 |
| Products     |  25,612 |

The raw and processed datasets are tracked using DVC and are **not stored directly in Git**.

## SVD Baseline

The model is implemented in:

```text
src/models/svd_baseline.py
```

Run:

```bash
python src/models/svd_baseline.py
```

The model:

1. Creates a train/test split.
2. Builds the training user-item matrix.
3. Computes a 50-factor SVD.
4. Predicts scores for all user-item pairs.
5. Masks items already seen during training.
6. Generates Top-10 recommendations.

Model artifacts are stored under:

```text
models/svd/
```

and tracked using:

```text
models/svd.dvc
```

## Evaluation

Run:

```bash
python src/evaluation/evaluate_svd.py
```

Results are written to:

```text
data/predictions/svd_evaluation.csv
```

Current baseline:

| Metric       |    Score |
| ------------ | -------: |
| Precision@10 | 0.003745 |
| Recall@10    | 0.037452 |
| Hit Rate@10  | 0.037452 |
| NDCG@10      | 0.020084 |

These values are the baseline for comparing subsequent recommendation models.

## DVC

DVC tracks the large ML artifacts while Git tracks the code and DVC metadata.

Currently tracked:

```text
data/raw/Video_Games.csv.gz
data/processed/video_games.parquet
models/svd/
```

The shared DVC remote is a Google Drive folder.

### First-time setup

Clone the repository and enter it:

```bash
git clone <repository-url>
cd mlops-recsys
```

Create/activate the Python environment and install dependencies:

```bash
pip install -r requirements.txt
pip install dvc-gdrive
```

Pull the project data and model artifacts:

```bash
dvc pull
```

DVC will open Google authentication. Sign in using a Google account that has access to the project's shared Drive folder.

Verify:

```bash
dvc status
```

Expected:

```text
Data and pipelines are up to date.
```

### Normal workflow

Pull the latest Git changes:

```bash
git pull
```

Pull the corresponding DVC artifacts:

```bash
dvc pull
```

After changing a DVC-tracked dataset/model:

```bash
dvc add <path>
git add <path>.dvc
git commit -m "Update <artifact>"
dvc push
git push
```

### Important

Never commit:

```text
.dvc/config.local
```

It contains local authentication credentials.

Never commit OAuth client secrets or credential JSON files.

## Branch scope

This branch establishes the SVD recommendation baseline and its reproducible data/model workflow.

The SVD metrics above should be retained as the reference point for the next recommendation approach.

