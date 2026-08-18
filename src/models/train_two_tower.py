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
