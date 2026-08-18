# Two-Tower Recommendation Model

## Overview

Stretch baseline for the MLOps recommendation-system project.

The model learns user and item embeddings in a shared latent space and scores recommendations using embedding similarity. It uses the same Amazon Video Games dataset and chronological evaluation protocol as the SVD baseline.

## Implementation

- PyTorch user and item embedding towers
- In-batch negative sampling
- Cosine similarity scoring
- Chronological train/validation split
- Recall@10 and NDCG@10 evaluation
- Apple Silicon MPS acceleration
- MLflow experiment and model tracking

Implementation:

- `src/models/two_tower.py`
- `src/models/train_two_tower.py`

## Dataset

Processed dataset:

`data/processed/video_games.parquet`

Dataset size:

- 814,586 interactions
- 94,762 users
- 25,612 items

## Training

Run from the repository root:

`PYTHONPATH=. python src/models/train_two_tower.py`

The trained model and metrics are logged to the project's MLflow server.

## Result

The initial five-epoch run established a working neural retrieval pipeline:

- Loss: 6.8426
- Recall@10: 0.00042
- NDCG@10: 0.00018

The result is substantially below the existing SVD baseline, so the Two-Tower model is treated as a completed stretch experiment rather than the primary recommendation model.

## MLflow

Experiment:

`two-tower`

The trained model was successfully logged to and loaded from the MLflow artifact store.

## Status

Two-Tower implementation is complete for the current project scope. Further model optimization is not required at this stage; the project proceeds to the streaming/MLOps infrastructure work.
