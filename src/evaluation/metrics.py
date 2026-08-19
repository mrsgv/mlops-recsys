Common train/test splitting utilities for the recommendation system.

Evaluation protocol
-------------------
The project uses deterministic chronological leave-one-out evaluation:

- For every user with at least two interactions:
    - the chronologically latest interaction is held out as test
    - all earlier interactions are used for training
- If multiple interactions have the same timestamp, item_idx is used as
  a deterministic tie-breaker.
- Users with fewer than two interactions are excluded because they cannot
  contribute both training history and a held-out test interaction.

This module is the single source of truth for model train/test splitting.
All recommendation models must use this function.
"""

from typing import Tuple

import pandas as pd


REQUIRED_COLUMNS = {
    "user_idx",
    "item_idx",
    "rating",
    "timestamp",
}


def chronological_train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a deterministic chronological leave-one-out split.

    Ordering is:
        user_idx -> timestamp -> item_idx

    For each eligible user:
        latest interaction -> test
        all earlier interactions -> train

    Parameters
    ----------
    df:
        Processed interaction dataframe.

    Returns
    -------
    train:
        Training interactions.

    test:
        One held-out interaction per eligible user.

    Raises
    ------
    ValueError
        If required columns are missing, the dataframe is empty,
        or no user has at least two interactions.
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Input dataframe is missing required columns: "
            f"{sorted(missing)}"
        )

    data = df.copy()

    # Deterministic chronological ordering.
    #
    # timestamp defines chronology.
    # item_idx breaks ties when two interactions have the same timestamp.
    data = data.sort_values(
        ["user_idx", "timestamp", "item_idx"],
        kind="mergesort",
    )

    # Only users with enough history can participate in leave-one-out
    # evaluation.
    user_counts = data.groupby("user_idx").size()

    eligible_users = user_counts[user_counts >= 2].index

    data = data[
        data["user_idx"].isin(eligible_users)
    ].copy()

    if data.empty:
        raise ValueError(
            "No users have at least two interactions; "
            "cannot create train/test split."
        )

    # Last interaction according to the deterministic ordering is test.
    test = (
        data
        .groupby("user_idx", sort=False)
        .tail(1)
    )

    # All remaining interactions are training data.
    train = data.drop(index=test.index)

    # Return deterministic dataframes.
    train = (
        train
        .sort_values(
            ["user_idx", "timestamp", "item_idx"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    test = (
        test
        .sort_values(
            ["user_idx"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return train, test


def validate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Validate invariants of a chronological leave-one-out split.

    Raises AssertionError if any invariant is violated.
    """

    # Every test user must have exactly one held-out interaction.
    test_counts = test.groupby("user_idx").size()

    assert test_counts.eq(1).all(), (
        "Every test user must have exactly one test interaction."
    )

    # Every test user must also have training history.
    train_users = set(train["user_idx"].unique())
    test_users = set(test["user_idx"].unique())

    assert test_users.issubset(train_users), (
        "Every test user must also appear in training data."
    )

    # Train/test interaction identities must not overlap.
    train_keys = set(
        zip(
            train["user_idx"],
            train["item_idx"],
            train["timestamp"],
        )
    )

    test_keys = set(
        zip(
            test["user_idx"],
            test["item_idx"],
            test["timestamp"],
        )
    )

    assert train_keys.isdisjoint(test_keys), (
        "Training and test interactions must not overlap."
    )

    # Test must never occur before the latest training timestamp.
    #
    # Equality is allowed because item_idx is used as the deterministic
    # tie-breaker when timestamps are identical.
    latest_train_timestamp = (
        train
        .groupby("user_idx")["timestamp"]
        .max()
    )

    test_timestamp = (
        test
        .set_index("user_idx")["timestamp"]
    )

    timestamp_difference = (
        test_timestamp - latest_train_timestamp
    )

    assert (timestamp_difference >= 0).all(), (
        "A test interaction occurs before the user's latest "
        "training interaction."
    )
((.venv) ) saikrishnakowda@Saikrishnas-MacBook-Pro mlops-recsys % sed -n '1,260p' src/evaluation/evaluate_svd.py
import numpy as np
import pandas as pd

from src.evaluation.split import (
    chronological_train_test_split,
    validate_split,
)


DATA_PATH = "data/processed/video_games.parquet"
RECOMMENDATIONS_PATH = "data/predictions/svd_top10.parquet"
OUTPUT_PATH = "data/predictions/svd_evaluation.csv"

TOP_K = 10


def main():

    print("\n=== Loading Data ===")

    df = pd.read_parquet(DATA_PATH)

    print(f"Interactions: {len(df):,}")
    print(f"Users: {df['user_idx'].nunique():,}")
    print(f"Products: {df['item_idx'].nunique():,}")

    # ---------------------------------------------------------
    # Common chronological evaluation split
    # ---------------------------------------------------------

    print("\n=== Creating Common Evaluation Split ===")

    # Create and validate the common split.
    train, test = chronological_train_test_split(df)
    validate_split(train, test)

    print(f"Test interactions: {len(test):,}")
    print(f"Test users: {test['user_idx'].nunique():,}")

    # ---------------------------------------------------------
    # Load recommendations
    # ---------------------------------------------------------

    print("\n=== Loading SVD Recommendations ===")

    recommendations = pd.read_parquet(
        RECOMMENDATIONS_PATH
    )

    print(
        f"Recommendation rows: "
        f"{len(recommendations):,}"
    )

    print(
        f"Users with recommendations: "
        f"{recommendations['user_idx'].nunique():,}"
    )

    # ---------------------------------------------------------
    # Convert recommendations into dictionary
    # ---------------------------------------------------------

    recommendation_map = (
        recommendations
        .sort_values(["user_idx", "rank"])
        .groupby("user_idx")["item_idx"]
        .apply(list)
        .to_dict()
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    print("\n=== Evaluating ===")

    hits = 0
    ndcg_sum = 0.0
    users_evaluated = 0

    for row in test.itertuples(index=False):

        user_idx = row.user_idx
        actual_item = row.item_idx

        if user_idx not in recommendation_map:
            continue

        recommended = recommendation_map[user_idx][:TOP_K]

        users_evaluated += 1

        if actual_item in recommended:

            hits += 1

            rank = recommended.index(actual_item) + 1

            ndcg_sum += (
                1.0 / np.log2(rank + 1)
            )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    if users_evaluated == 0:
        print("No users could be evaluated.")
        return

    hit_rate = hits / users_evaluated

    # One held-out relevant item per user:
    #
    # Precision@K = Hit Rate@K / K
    # Recall@K    = Hit Rate@K

    precision = hit_rate / TOP_K
    recall = hit_rate
    ndcg = ndcg_sum / users_evaluated

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n=== SVD Baseline Results ===")

    print(f"Users evaluated: {users_evaluated:,}")

    print(
        f"Precision@{TOP_K}: "
        f"{precision:.6f}"
    )

    print(
        f"Recall@{TOP_K}:    "
        f"{recall:.6f}"
    )

    print(
        f"Hit Rate@{TOP_K}:  "
        f"{hit_rate:.6f}"
    )

    print(
        f"NDCG@{TOP_K}:      "
        f"{ndcg:.6f}"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "SVD",
                "factors": 50,
                "users_evaluated": users_evaluated,
                "precision_at_10": precision,
                "recall_at_10": recall,
                "hit_rate_at_10": hit_rate,
                "ndcg_at_10": ndcg,
            }
        ]
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== Saved ===")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
((.venv) ) saikrishnakowda@Saikrishnas-MacBook-Pro mlops-recsys % sed -n '1,320p' src/models/svd_baseline.py
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src.evaluation.split import (
    chronological_train_test_split,
    validate_split,
)


DATA_PATH = "data/processed/video_games.parquet"
MODEL_DIR = "models/svd"
RECOMMENDATIONS_PATH = "data/predictions/svd_top10.parquet"

N_FACTORS = 50
TOP_K = 10


def main():

    print("\n=== Loading Processed Data ===")

    df = pd.read_parquet(DATA_PATH)

    print(f"Interactions: {len(df):,}")
    print(f"Users: {df['user_idx'].nunique():,}")
    print(f"Products: {df['item_idx'].nunique():,}")

    # ---------------------------------------------------------
    # Common chronological train/test split
    # ---------------------------------------------------------

    print("\n=== Creating Common Chronological Split ===")

    train, test = chronological_train_test_split(df)

    validate_split(train, test)

    print(f"Training interactions: {len(train):,}")
    print(f"Test interactions: {len(test):,}")
    print(f"Training users: {train['user_idx'].nunique():,}")
    print(f"Test users: {test['user_idx'].nunique():,}")

    # ---------------------------------------------------------
    # Build training user-item matrix
    # ---------------------------------------------------------

    print("\n=== Building Training User-Item Matrix ===")

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    matrix = csr_matrix(
        (
            train["rating"].values,
            (
                train["user_idx"].values,
                train["item_idx"].values,
            ),
        ),
        shape=(n_users, n_items),
    )

    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero entries: {matrix.nnz:,}")

    total_entries = matrix.shape[0] * matrix.shape[1]
    sparsity = 1 - (matrix.nnz / total_entries)

    print(f"Sparsity: {sparsity:.4%}")

    # ---------------------------------------------------------
    # SVD
    # ---------------------------------------------------------

    print(f"\n=== Computing SVD ({N_FACTORS} factors) ===")

    start_time = time.time()

    U, sigma, Vt = svds(
        matrix.astype(float),
        k=N_FACTORS,
    )

    # scipy.sparse.linalg.svds returns singular values in ascending
    # order. Reverse them so the largest factors come first.
    order = np.argsort(sigma)[::-1]

    sigma = sigma[order]
    U = U[:, order]
    Vt = Vt[order, :]

    elapsed = time.time() - start_time

    print(f"SVD completed in {elapsed:.2f} seconds")

    print("\n=== Factor Shapes ===")
    print(f"U:     {U.shape}")
    print(f"Sigma: {sigma.shape}")
    print(f"Vt:    {Vt.shape}")

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    print("\n=== Generating Predictions ===")

    predictions = (U * sigma) @ Vt

    print(f"Prediction matrix shape: {predictions.shape}")

    # ---------------------------------------------------------
    # Remove items already seen during training
    # ---------------------------------------------------------

    print("\n=== Masking Previously Seen Items ===")

    train_users = train["user_idx"].to_numpy()
    train_items = train["item_idx"].to_numpy()

    predictions[train_users, train_items] = -np.inf

    # ---------------------------------------------------------
    # Generate Top-K recommendations
    # ---------------------------------------------------------

    print("\n=== Generating Top-K Recommendations ===")

    rows = []

    for user_idx in range(n_users):

        user_scores = predictions[user_idx]

        top_indices = np.argpartition(
            user_scores,
            -TOP_K,
        )[-TOP_K:]

        top_indices = top_indices[
            np.argsort(user_scores[top_indices])[::-1]
        ]

        for rank, item_idx in enumerate(
            top_indices,
            start=1,
        ):
            rows.append(
                {
                    "user_idx": user_idx,
                    "item_idx": int(item_idx),
                    "rank": rank,
                    "predicted_score": float(
                        user_scores[item_idx]
                    ),
                }
            )

    recommendations = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Save model factors
    # ---------------------------------------------------------

    print("\n=== Saving SVD Factors ===")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data/predictions", exist_ok=True)

    np.save(f"{MODEL_DIR}/U.npy", U)
    np.save(f"{MODEL_DIR}/sigma.npy", sigma)
    np.save(f"{MODEL_DIR}/Vt.npy", Vt)

    # ---------------------------------------------------------
    # Save recommendations
    # ---------------------------------------------------------

    print("\n=== Saving Recommendations ===")

    recommendations.to_parquet(
        RECOMMENDATIONS_PATH,
        index=False,
    )

    print("\n=== Saved ===")
    print(f"Model: {MODEL_DIR}/")
    print(f"Recommendations: {RECOMMENDATIONS_PATH}")

    print("\n=== Sample Recommendations ===")

    print(
        recommendations
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
((.venv) ) saikrishnakowda@Saikrishnas-MacBook-Pro mlops-recsys % sed -n '1,320p' src/models/train_two_tower.py
import os

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.two_tower import TwoTowerModel


DATA_PATH = "data/processed/video_games.parquet"

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 1e-3
K = 10


class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]


def chronological_split(df):
    df = df.sort_values(["user_idx", "timestamp"])

    train_parts = []
    val_parts = []

    for _, group in df.groupby("user_idx", sort=False):
        if len(group) >= 2:
            train_parts.append(group.iloc[:-1])
            val_parts.append(group.iloc[-1:])
        else:
            train_parts.append(group)

    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)

    return train, val


@torch.no_grad()
def evaluate(model, train_df, val_df, num_items, device, k=10):
    model.eval()

    item_ids = torch.arange(num_items, device=device)
    item_embeddings = model.encode_items(item_ids)

    train_items = train_df.groupby("user_idx")["item_idx"].apply(set)

    recalls = []
    ndcgs = []

    for row in val_df.itertuples(index=False):
        user = torch.tensor([row.user_idx], device=device)
        user_embedding = model.encode_users(user)

        scores = (user_embedding @ item_embeddings.T).squeeze(0)

        # Don't recommend items already seen during training.
        seen = train_items.get(row.user_idx, set())
        if seen:
            scores[list(seen)] = -float("inf")

        topk = torch.topk(scores, k=min(k, num_items)).indices.cpu().numpy()

        target = row.item_idx

        if target in topk:
            rank = np.where(topk == target)[0][0] + 1
            recalls.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            recalls.append(0.0)
            ndcgs.append(0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    df = pd.read_parquet(DATA_PATH)

    train_df, val_df = chronological_split(df)

    print(f"Total interactions: {len(df):,}")
    print(f"Train interactions: {len(train_df):,}")
    print(f"Validation interactions: {len(val_df):,}")

    num_users = df["user_idx"].nunique()
    num_items = df["item_idx"].nunique()

    dataset = InteractionDataset(train_df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("two-tower")

    with mlflow.start_run(run_name="two-tower-v1"):
        mlflow.log_params(
            {
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "num_users": num_users,
                "num_items": num_items,
                "train_interactions": len(train_df),
                "device": device,
            }
        )

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for users, items in loader:
                users = users.to(device)
                items = items.to(device)

                scores = model(users, items)

                labels = torch.arange(
                    len(users),
                    device=device,
                )

                loss = F.cross_entropy(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            recall, ndcg = evaluate(
                model,
                train_df,
                val_df,
                num_items,
                device,
                K,
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"loss={avg_loss:.4f} "
                f"Recall@{K}={recall:.4f} "
                f"NDCG@{K}={ndcg:.4f}"
            )

            mlflow.log_metrics(
                {
                    "loss": avg_loss,
                    f"recall_at_{K}": recall,
                    f"ndcg_at_{K}": ndcg,
                },
                step=epoch,
            )

        mlflow.pytorch.log_model(
            model,
            name="two_tower_model",
        )


if __name__ == "__main__":
    main()
((.venv) ) saikrishnakowda@Saikrishnas-MacBook-Pro mlops-recsys % 