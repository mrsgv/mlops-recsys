from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import evaluate_top_k
from src.evaluation.split import (
    chronological_train_test_split,
    validate_split,
)
from src.models.item_features import ItemFeatureEncoder
from src.models.two_tower_v2 import TwoTowerV2


# ============================================================
# Paths
# ============================================================

DATA_PATH = "data/processed/video_games.parquet"

ITEM_FEATURES_PATH = (
    "data/processed/video_games_items.parquet"
)

MODEL_DIR = Path("models/two_tower_v2")

ENCODER_PATH = (
    MODEL_DIR / "item_feature_encoder.pkl"
)

MODEL_PATH = (
    MODEL_DIR / "model.pt"
)


# ============================================================
# Training configuration
# ============================================================

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
CATEGORICAL_DIM = 16

BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 1e-3

TOP_K = 10
MAX_TEXT_FEATURES = 256

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class TwoTowerV2Dataset(Dataset):
    """
    Positive user-item interactions enriched with precomputed
    item metadata features.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        item_features: dict[str, object],
    ) -> None:
        item_indices = (
            interactions["item_idx"]
            .to_numpy()
        )

        self.users = torch.tensor(
            interactions["user_idx"].to_numpy(),
            dtype=torch.long,
        )

        self.items = torch.tensor(
            item_indices,
            dtype=torch.long,
        )

        self.main_category = torch.tensor(
            item_features["main_category"][
                item_indices
            ],
            dtype=torch.long,
        )

        self.brand = torch.tensor(
            item_features["brand"][
                item_indices
            ],
            dtype=torch.long,
        )

        self.store = torch.tensor(
            item_features["store"][
                item_indices
            ],
            dtype=torch.long,
        )

        self.price_bucket = torch.tensor(
            item_features["price_bucket"][
                item_indices
            ],
            dtype=torch.long,
        )

        self.numeric_features = torch.tensor(
            item_features["numeric_features"][
                item_indices
            ],
            dtype=torch.float32,
        )

        self.text_features = torch.tensor(
            item_features["text_features"][
                item_indices
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return (
            self.users[idx],
            self.items[idx],
            self.main_category[idx],
            self.brand[idx],
            self.store[idx],
            self.price_bucket[idx],
            self.numeric_features[idx],
            self.text_features[idx],
        )


# ============================================================
# Utility functions
# ============================================================

def select_device() -> str:
    """Select the best available PyTorch device."""
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def build_ground_truth(
    test_df: pd.DataFrame,
) -> dict[int, set[int]]:
    """
    Build user -> held-out relevant-item mapping.

    Under the chronological leave-one-out protocol there is
    exactly one relevant item per eligible user.
    """
    return (
        test_df
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def prepare_all_item_features(
    item_features: dict[str, object],
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Convert all item features to tensors once for evaluation.
    """
    return {
        "main_category": torch.tensor(
            item_features["main_category"],
            dtype=torch.long,
            device=device,
        ),
        "brand": torch.tensor(
            item_features["brand"],
            dtype=torch.long,
            device=device,
        ),
        "store": torch.tensor(
            item_features["store"],
            dtype=torch.long,
            device=device,
        ),
        "price_bucket": torch.tensor(
            item_features["price_bucket"],
            dtype=torch.long,
            device=device,
        ),
        "numeric_features": torch.tensor(
            item_features["numeric_features"],
            dtype=torch.float32,
            device=device,
        ),
        "text_features": torch.tensor(
            item_features["text_features"],
            dtype=torch.float32,
            device=device,
        ),
    }


@torch.no_grad()
def generate_recommendations(
    model: TwoTowerV2,
    train_df: pd.DataFrame,
    evaluation_users: pd.Series,
    all_item_features: dict[str, torch.Tensor],
    num_items: int,
    device: str,
    k: int = TOP_K,
) -> dict[int, list[int]]:
    """
    Generate ranked Top-K recommendations.

    Previously seen training items are masked so that they cannot
    appear in the recommendation list.
    """
    model.eval()

    item_ids = torch.arange(
        num_items,
        device=device,
        dtype=torch.long,
    )

    item_embeddings = model.encode_items(
        item_idx=item_ids,
        main_category_idx=all_item_features[
            "main_category"
        ],
        brand_idx=all_item_features["brand"],
        store_idx=all_item_features["store"],
        price_bucket_idx=all_item_features[
            "price_bucket"
        ],
        numeric_features=all_item_features[
            "numeric_features"
        ],
        text_features=all_item_features[
            "text_features"
        ],
    )

    seen_items_by_user = (
        train_df
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    recommendations: dict[int, list[int]] = {}

    top_k = min(k, num_items)

    for user_idx in evaluation_users.unique():
        user_idx = int(user_idx)

        user_tensor = torch.tensor(
            [user_idx],
            device=device,
            dtype=torch.long,
        )

        user_embedding = model.encode_users(
            user_tensor
        )

        scores = (
            user_embedding
            @ item_embeddings.T
        ).squeeze(0)

        seen_items = seen_items_by_user.get(
            user_idx,
            set(),
        )

        if seen_items:
            scores[
                list(seen_items)
            ] = -float("inf")

        top_indices = torch.topk(
            scores,
            k=top_k,
            largest=True,
        ).indices.tolist()

        recommendations[user_idx] = [
            int(item_idx)
            for item_idx in top_indices
        ]

    return recommendations


def evaluate_model(
    model: TwoTowerV2,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_features: dict[str, object],
    num_items: int,
    device: str,
) -> dict[str, float | int]:
    """Generate recommendations and evaluate using the common framework."""
    all_item_features = prepare_all_item_features(
        item_features,
        device,
    )

    recommendations = generate_recommendations(
        model=model,
        train_df=train_df,
        evaluation_users=test_df["user_idx"],
        all_item_features=all_item_features,
        num_items=num_items,
        device=device,
        k=TOP_K,
    )

    ground_truth = build_ground_truth(
        test_df
    )

    return evaluate_top_k(
        recommendations=recommendations,
        ground_truth=ground_truth,
        k=TOP_K,
    )


def log_epoch_metrics(
    metrics: dict[str, float | int],
    epoch: int,
) -> None:
    """Log common ranking metrics to MLflow."""
    mlflow.log_metrics(
        {
            f"precision_at_{TOP_K}": float(
                metrics[f"precision_at_{TOP_K}"]
            ),
            f"recall_at_{TOP_K}": float(
                metrics[f"recall_at_{TOP_K}"]
            ),
            f"hit_rate_at_{TOP_K}": float(
                metrics[f"hit_rate_at_{TOP_K}"]
            ),
            f"ndcg_at_{TOP_K}": float(
                metrics[f"ndcg_at_{TOP_K}"]
            ),
        },
        step=epoch,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed()

    device = select_device()

    print("=" * 60)
    print("Two-Tower V2 — Metadata-Aware Retrieval")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed:   {RANDOM_SEED}")

    # --------------------------------------------------------
    # 1. Load interactions
    # --------------------------------------------------------

    print("\n=== Loading Interaction Data ===")

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Interactions: "
        f"{len(df):,}"
    )

    print(
        f"Users: "
        f"{df['user_idx'].nunique():,}"
    )

    print(
        f"Items: "
        f"{df['item_idx'].nunique():,}"
    )

    # --------------------------------------------------------
    # 2. Common chronological train/test split
    # --------------------------------------------------------

    print(
        "\n=== Creating Common Chronological Split ==="
    )

    train_df, test_df = (
        chronological_train_test_split(df)
    )

    validate_split(
        train_df,
        test_df,
    )

    print(
        f"Train interactions: "
        f"{len(train_df):,}"
    )

    print(
        f"Test interactions: "
        f"{len(test_df):,}"
    )

    print(
        f"Training users: "
        f"{train_df['user_idx'].nunique():,}"
    )

    print(
        f"Test users: "
        f"{test_df['user_idx'].nunique():,}"
    )

    # --------------------------------------------------------
    # 3. Load item feature table
    # --------------------------------------------------------

    print(
        "\n=== Loading Item Feature Table ==="
    )

    item_df = (
        pd.read_parquet(
            ITEM_FEATURES_PATH
        )
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    num_items = len(item_df)

    if (
        item_df["item_idx"]
        .tolist()
        != list(range(num_items))
    ):
        raise ValueError(
            "Item feature table must contain "
            "contiguous item_idx values from 0 "
            "to num_items - 1."
        )

    print(
        f"Item feature rows: "
        f"{len(item_df):,}"
    )

    print(
        f"Metadata coverage: "
        f"{item_df['metadata_found'].mean():.2%}"
    )

    # --------------------------------------------------------
    # 4. Fit item feature encoder
    # --------------------------------------------------------

    print(
        "\n=== Fitting Item Feature Encoder ==="
    )

    encoder = ItemFeatureEncoder.fit(
        item_df,
        max_text_features=MAX_TEXT_FEATURES,
    )

    item_features = encoder.transform(
        item_df
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder.save(
        ENCODER_PATH
    )

    print(
        f"Encoder saved to: "
        f"{ENCODER_PATH}"
    )

    print(
        "Text features:",
        item_features[
            "text_features"
        ].shape,
    )

    print(
        "Numeric features:",
        item_features[
            "numeric_features"
        ].shape,
    )

    # --------------------------------------------------------
    # 5. Training dataset
    # --------------------------------------------------------

    print(
        "\n=== Building Training Dataset ==="
    )

    dataset = TwoTowerV2Dataset(
        interactions=train_df,
        item_features=item_features,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    print(
        f"Training examples: "
        f"{len(dataset):,}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Batches per epoch: "
        f"{len(loader):,}"
    )

    # --------------------------------------------------------
    # 6. Create model
    # --------------------------------------------------------

    num_users = (
        df["user_idx"].nunique()
    )

    model = TwoTowerV2(
        num_users=num_users,
        num_items=num_items,
        main_category_size=len(
            encoder.vocabularies[
                "main_category"
            ]
        ),
        brand_size=len(
            encoder.vocabularies["brand"]
        ),
        store_size=len(
            encoder.vocabularies["store"]
        ),
        price_bucket_size=len(
            encoder.vocabularies[
                "price_bucket"
            ]
        ),
        text_dim=item_features[
            "text_features"
        ].shape[1],
        numeric_dim=item_features[
            "numeric_features"
        ].shape[1],
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        categorical_dim=CATEGORICAL_DIM,
    ).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{total_parameters:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # 7. MLflow configuration
    # --------------------------------------------------------

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )

    if tracking_uri:
        mlflow.set_tracking_uri(
            tracking_uri
        )

    mlflow.set_experiment(
        "two-tower-v2"
    )

    # --------------------------------------------------------
    # 8. Training
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="two-tower-v2-metadata"
    ):
        mlflow.log_params(
            {
                "model_version": "two-tower-v2",
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "categorical_dim": CATEGORICAL_DIM,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "top_k": TOP_K,
                "max_text_features": (
                    MAX_TEXT_FEATURES
                ),
                "num_users": num_users,
                "num_items": num_items,
                "train_interactions": len(
                    train_df
                ),
                "test_interactions": len(
                    test_df
                ),
                "metadata_coverage": float(
                    item_df[
                        "metadata_found"
                    ].mean()
                ),
                "device": device,
                "seed": RANDOM_SEED,
            }
        )

        best_recall = float("-inf")
        best_epoch = -1

        for epoch in range(EPOCHS):
            model.train()

            total_loss = 0.0
            batch_count = 0

            for batch in loader:
                (
                    users,
                    items,
                    main_category,
                    brand,
                    store,
                    price_bucket,
                    numeric_features,
                    text_features,
                ) = batch

                users = users.to(device)
                items = items.to(device)
                main_category = (
                    main_category.to(device)
                )
                brand = brand.to(device)
                store = store.to(device)
                price_bucket = (
                    price_bucket.to(device)
                )
                numeric_features = (
                    numeric_features.to(device)
                )
                text_features = (
                    text_features.to(device)
                )

                scores = model(
                    users,
                    items,
                    main_category,
                    brand,
                    store,
                    price_bucket,
                    numeric_features,
                    text_features,
                )

                labels = torch.arange(
                    len(users),
                    device=device,
                )

                loss = F.cross_entropy(
                    scores,
                    labels,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item()
                )

                batch_count += 1

            avg_loss = (
                total_loss / batch_count
            )

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            metrics = evaluate_model(
                model=model,
                train_df=train_df,
                test_df=test_df,
                item_features=item_features,
                num_items=num_items,
                device=device,
            )

            recall = float(
                metrics[
                    f"recall_at_{TOP_K}"
                ]
            )

            print(
                f"\nEpoch {epoch + 1}/{EPOCHS}"
            )

            print(
                f"Loss: "
                f"{avg_loss:.6f}"
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

            mlflow.log_metric(
                "loss",
                avg_loss,
                step=epoch,
            )

            log_epoch_metrics(
                metrics,
                epoch,
            )

            # ------------------------------------------------
            # Track best epoch by Recall@K
            # ------------------------------------------------

            if recall > best_recall:
                best_recall = recall
                best_epoch = epoch

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),
                        "embedding_dim":
                            EMBEDDING_DIM,
                        "hidden_dim":
                            HIDDEN_DIM,
                        "categorical_dim":
                            CATEGORICAL_DIM,
                        "num_users":
                            num_users,
                        "num_items":
                            num_items,
                        "text_dim":
                            item_features[
                                "text_features"
                            ].shape[1],
                        "numeric_dim":
                            item_features[
                                "numeric_features"
                            ].shape[1],
                        "top_k":
                            TOP_K,
                        "seed":
                            RANDOM_SEED,
                    },
                    MODEL_PATH,
                )

        # ----------------------------------------------------
        # 9. Final MLflow logging
        # ----------------------------------------------------

        mlflow.log_metric(
            "best_recall_at_10",
            best_recall,
        )

        mlflow.log_metric(
            "best_epoch",
            best_epoch,
        )

        mlflow.log_artifact(
            str(MODEL_PATH)
        )

        mlflow.log_artifact(
            str(ENCODER_PATH)
        )

        print("\n=== Training Complete ===")
        print(
            f"Best epoch: "
            f"{best_epoch + 1}"
        )
        print(
            f"Best Recall@{TOP_K}: "
            f"{best_recall:.6f}"
        )
        print(
            f"Model saved to: "
            f"{MODEL_PATH}"
        )
        print(
            f"Encoder saved to: "
            f"{ENCODER_PATH}"
        )


if __name__ == "__main__":
    main()