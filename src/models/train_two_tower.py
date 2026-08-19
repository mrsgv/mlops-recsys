from __future__ import annotations

import os

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import evaluate_top_k
from src.evaluation.split import (
    chronological_train_test_split,
    validate_split,
)
from src.models.two_tower import TwoTowerModel


DATA_PATH = "data/processed/video_games.parquet"

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 1e-3
TOP_K = 10


class InteractionDataset(Dataset):
    """PyTorch dataset for positive user-item interactions."""

    def __init__(self, df: pd.DataFrame):
        self.users = torch.tensor(
            df["user_idx"].values,
            dtype=torch.long,
        )
        self.items = torch.tensor(
            df["item_idx"].values,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx]


@torch.no_grad()
def generate_recommendations(
    model: TwoTowerModel,
    train_df: pd.DataFrame,
    evaluation_users: pd.Series,
    num_items: int,
    device: str,
    k: int = TOP_K,
) -> dict[int, list[int]]:
    """
    Generate ranked Top-K recommendations for evaluation users.

    Previously seen training items are excluded from recommendations.
    """
    model.eval()

    item_ids = torch.arange(
        num_items,
        device=device,
    )

    item_embeddings = model.encode_items(item_ids)

    seen_items_by_user = (
        train_df
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    recommendation_map: dict[int, list[int]] = {}

    for user_idx in evaluation_users:
        user_tensor = torch.tensor(
            [int(user_idx)],
            device=device,
            dtype=torch.long,
        )

        user_embedding = model.encode_users(user_tensor)

        scores = (
            user_embedding @ item_embeddings.T
        ).squeeze(0)

        seen_items = seen_items_by_user.get(
            int(user_idx),
            set(),
        )

        if seen_items:
            seen_indices = list(seen_items)
            scores[seen_indices] = -float("inf")

        top_k = min(k, num_items)

        top_indices = torch.topk(
            scores,
            k=top_k,
        ).indices.cpu().numpy()

        recommendation_map[int(user_idx)] = [
            int(item_idx)
            for item_idx in top_indices
        ]

    return recommendation_map


def build_ground_truth(
    test_df: pd.DataFrame,
) -> dict[int, set[int]]:
    """
    Build user -> held-out item mapping.

    The common chronological leave-one-out protocol produces
    exactly one relevant test item per eligible user.
    """
    return (
        test_df
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def select_device() -> str:
    """Select the best available PyTorch device."""
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def main() -> None:
    device = select_device()

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Load processed data
    # ---------------------------------------------------------

    print("\n=== Loading Data ===")

    df = pd.read_parquet(DATA_PATH)

    print(f"Total interactions: {len(df):,}")
    print(f"Users: {df['user_idx'].nunique():,}")
    print(f"Items: {df['item_idx'].nunique():,}")

    # ---------------------------------------------------------
    # Common chronological train/test split
    # ---------------------------------------------------------

    print("\n=== Creating Common Chronological Split ===")

    train_df, test_df = chronological_train_test_split(df)

    validate_split(
        train_df,
        test_df,
    )

    print(
        f"Train interactions: {len(train_df):,}"
    )
    print(
        f"Test interactions: {len(test_df):,}"
    )
    print(
        f"Training users: "
        f"{train_df['user_idx'].nunique():,}"
    )
    print(
        f"Test users: "
        f"{test_df['user_idx'].nunique():,}"
    )

    # ---------------------------------------------------------
    # Training setup
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # MLflow
    # ---------------------------------------------------------

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )

    if tracking_uri:
        mlflow.set_tracking_uri(
            tracking_uri
        )

    mlflow.set_experiment("two-tower")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    with mlflow.start_run(
        run_name="two-tower-v1"
    ):
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
                "test_interactions": len(test_df),
                "device": device,
                "top_k": TOP_K,
            }
        )

        for epoch in range(EPOCHS):
            model.train()

            total_loss = 0.0

            for users, items in loader:
                users = users.to(device)
                items = items.to(device)

                scores = model(
                    users,
                    items,
                )

                labels = torch.arange(
                    len(users),
                    device=device,
                )

                loss = F.cross_entropy(
                    scores,
                    labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = (
                total_loss / len(loader)
            )

            # -------------------------------------------------
            # Evaluation using the common framework
            # -------------------------------------------------

            recommendations = generate_recommendations(
                model=model,
                train_df=train_df,
                evaluation_users=test_df["user_idx"],
                num_items=num_items,
                device=device,
                k=TOP_K,
            )

            ground_truth = build_ground_truth(
                test_df
            )

            metrics = evaluate_top_k(
                recommendations=recommendations,
                ground_truth=ground_truth,
                k=TOP_K,
            )

            print(
                f"\nEpoch {epoch + 1}/{EPOCHS}"
            )
            print(
                f"Loss: "
                f"{avg_loss:.4f}"
            )
            print(
                f"Precision@{TOP_K}: "
                f"{metrics[f'precision_at_{TOP_K}']:.6f}"
            )
            print(
                f"Recall@{TOP_K}: "
                f"{metrics[f'recall_at_{TOP_K}']:.6f}"
            )
            print(
                f"Hit Rate@{TOP_K}: "
                f"{metrics[f'hit_rate_at_{TOP_K}']:.6f}"
            )
            print(
                f"NDCG@{TOP_K}: "
                f"{metrics[f'ndcg_at_{TOP_K}']:.6f}"
            )

            mlflow.log_metrics(
                {
                    "loss": avg_loss,
                    f"precision_at_{TOP_K}": metrics[
                        f"precision_at_{TOP_K}"
                    ],
                    f"recall_at_{TOP_K}": metrics[
                        f"recall_at_{TOP_K}"
                    ],
                    f"hit_rate_at_{TOP_K}": metrics[
                        f"hit_rate_at_{TOP_K}"
                    ],
                    f"ndcg_at_{TOP_K}": metrics[
                        f"ndcg_at_{TOP_K}"
                    ],
                },
                step=epoch,
            )

        # ---------------------------------------------------------
        # Log final model
        # ---------------------------------------------------------

        mlflow.pytorch.log_model(
            model,
            name="two_tower_model",
        )


if __name__ == "__main__":
    main()