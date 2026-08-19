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
from src.models.negative_sampling import NegativeSampler
from src.models.two_tower_v2 import TwoTowerV2


# ============================================================
# Paths
# ============================================================

DATA_PATH = "data/processed/video_games.parquet"

ITEM_FEATURES_PATH = (
    "data/processed/video_games_items.parquet"
)

MODEL_DIR = Path(
    "models/two_tower_v2_negative_sampling"
)

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

NUM_NEGATIVES = 5

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

class TwoTowerV21Dataset(Dataset):
    """
    Positive user-item interactions plus explicit,
    user-history-aware negative item IDs.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        negative_items: torch.Tensor,
    ) -> None:
        if len(interactions) != len(
            negative_items
        ):
            raise ValueError(
                "Number of negative-sample rows must "
                "match the number of interactions."
            )

        self.users = torch.tensor(
            interactions["user_idx"].to_numpy(),
            dtype=torch.long,
        )

        self.positive_items = torch.tensor(
            interactions["item_idx"].to_numpy(),
            dtype=torch.long,
        )

        self.negative_items = negative_items

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return (
            self.users[idx],
            self.positive_items[idx],
            self.negative_items[idx],
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
    """Convert all item features to tensors once."""
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


def build_dataset_for_epoch(
    train_df: pd.DataFrame,
    negative_sampler: NegativeSampler,
) -> TwoTowerV21Dataset:
    """
    Generate a fresh set of negatives and construct the dataset
    for one training epoch.
    """
    negative_matrix = (
        negative_sampler.sample_for_interactions(
            train_df
        )
    )

    negative_tensor = torch.tensor(
        negative_matrix,
        dtype=torch.long,
    )

    return TwoTowerV21Dataset(
        interactions=train_df,
        negative_items=negative_tensor,
    )


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

    Previously seen training items are masked.
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
        brand_idx=all_item_features[
            "brand"
        ],
        store_idx=all_item_features[
            "store"
        ],
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

    recommendations: dict[
        int,
        list[int],
    ] = {}

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

        seen_items = (
            seen_items_by_user.get(
                user_idx,
                set(),
            )
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
    """Evaluate using the common recommendation framework."""

    all_item_features = (
        prepare_all_item_features(
            item_features,
            device,
        )
    )

    recommendations = (
        generate_recommendations(
            model=model,
            train_df=train_df,
            evaluation_users=test_df[
                "user_idx"
            ],
            all_item_features=(
                all_item_features
            ),
            num_items=num_items,
            device=device,
            k=TOP_K,
        )
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
    """Log ranking metrics to MLflow."""
    mlflow.log_metrics(
        {
            f"precision_at_{TOP_K}": float(
                metrics[
                    f"precision_at_{TOP_K}"
                ]
            ),
            f"recall_at_{TOP_K}": float(
                metrics[
                    f"recall_at_{TOP_K}"
                ]
            ),
            f"hit_rate_at_{TOP_K}": float(
                metrics[
                    f"hit_rate_at_{TOP_K}"
                ]
            ),
            f"ndcg_at_{TOP_K}": float(
                metrics[
                    f"ndcg_at_{TOP_K}"
                ]
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
    print(
        "Two-Tower V2.1 — Explicit Negative Sampling"
    )
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Seed:   {RANDOM_SEED}")
    print(
        f"Negatives per positive: "
        f"{NUM_NEGATIVES}"
    )

    # --------------------------------------------------------
    # 1. Load interactions
    # --------------------------------------------------------

    print(
        "\n=== Loading Interaction Data ==="
    )

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
    # 2. Common chronological split
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
    # 3. Item feature table
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

    expected_item_ids = list(
        range(num_items)
    )

    if (
        item_df["item_idx"].tolist()
        != expected_item_ids
    ):
        raise ValueError(
            "Item feature table must contain "
            "contiguous item_idx values."
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
    # 4. Item feature encoder
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
    # 5. Model
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
            encoder.vocabularies[
                "brand"
            ]
        ),
        store_size=len(
            encoder.vocabularies[
                "store"
            ]
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
    # 6. Shared item features on device
    # --------------------------------------------------------

    all_item_features = (
        prepare_all_item_features(
            item_features,
            device,
        )
    )

    # --------------------------------------------------------
    # 7. Negative sampler
    # --------------------------------------------------------

    negative_sampler = NegativeSampler(
        num_items=num_items,
        num_negatives=NUM_NEGATIVES,
        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # 8. MLflow
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

    with mlflow.start_run(
        run_name=(
            "two-tower-v2.1-negative-sampling"
        )
    ):
        mlflow.log_params(
            {
                "model_version": (
                    "two-tower-v2.1"
                ),
                "negative_sampling": (
                    "explicit_user_history"
                ),
                "num_negatives": (
                    NUM_NEGATIVES
                ),
                "negative_seed": (
                    RANDOM_SEED
                ),
                "embedding_dim": (
                    EMBEDDING_DIM
                ),
                "hidden_dim": (
                    HIDDEN_DIM
                ),
                "categorical_dim": (
                    CATEGORICAL_DIM
                ),
                "batch_size": (
                    BATCH_SIZE
                ),
                "epochs": EPOCHS,
                "learning_rate": (
                    LEARNING_RATE
                ),
                "top_k": TOP_K,
                "max_text_features": (
                    MAX_TEXT_FEATURES
                ),
                "num_users": num_users,
                "num_items": num_items,
                "train_interactions": (
                    len(train_df)
                ),
                "test_interactions": (
                    len(test_df)
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

        best_recall = float(
            "-inf"
        )

        best_epoch = -1

        # ----------------------------------------------------
        # 9. Training
        # ----------------------------------------------------

        for epoch in range(EPOCHS):

            print(
                f"\n=== Epoch "
                f"{epoch + 1}/{EPOCHS} ==="
            )

            # Fresh deterministic negatives per epoch.
            epoch_sampler = NegativeSampler(
                num_items=num_items,
                num_negatives=NUM_NEGATIVES,
                seed=(
                    RANDOM_SEED + epoch
                ),
            )

            dataset = (
                build_dataset_for_epoch(
                    train_df,
                    epoch_sampler,
                )
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
            )

            model.train()

            total_loss = 0.0

            batch_count = 0

            for batch in loader:

                (
                    users,
                    positive_items,
                    negative_items,
                ) = batch

                users = users.to(device)

                positive_items = (
                    positive_items.to(device)
                )

                negative_items = (
                    negative_items.to(device)
                )

                batch_size = (
                    users.shape[0]
                )

                num_negatives = (
                    negative_items.shape[1]
                )

                # --------------------------------------------
                # Positive + explicit negatives
                # --------------------------------------------

                all_item_ids = torch.cat(
                    [
                        positive_items.unsqueeze(
                            1
                        ),
                        negative_items,
                    ],
                    dim=1,
                )

                flat_item_ids = (
                    all_item_ids.reshape(-1)
                )

                # --------------------------------------------
                # Look up item features
                # --------------------------------------------

                flat_main_category = (
                    all_item_features[
                        "main_category"
                    ][flat_item_ids]
                )

                flat_brand = (
                    all_item_features[
                        "brand"
                    ][flat_item_ids]
                )

                flat_store = (
                    all_item_features[
                        "store"
                    ][flat_item_ids]
                )

                flat_price_bucket = (
                    all_item_features[
                        "price_bucket"
                    ][flat_item_ids]
                )

                flat_numeric_features = (
                    all_item_features[
                        "numeric_features"
                    ][flat_item_ids]
                )

                flat_text_features = (
                    all_item_features[
                        "text_features"
                    ][flat_item_ids]
                )

                # --------------------------------------------
                # Encode items
                # --------------------------------------------

                item_embeddings = (
                    model.encode_items(
                        item_idx=flat_item_ids,
                        main_category_idx=(
                            flat_main_category
                        ),
                        brand_idx=(
                            flat_brand
                        ),
                        store_idx=(
                            flat_store
                        ),
                        price_bucket_idx=(
                            flat_price_bucket
                        ),
                        numeric_features=(
                            flat_numeric_features
                        ),
                        text_features=(
                            flat_text_features
                        ),
                    )
                )

                # --------------------------------------------
                # Encode users
                # --------------------------------------------

                user_embeddings = (
                    model.encode_users(
                        users
                    )
                )

                # Repeat each user embedding for
                # positive + negatives.
                user_embeddings = (
                    user_embeddings
                    .unsqueeze(1)
                    .expand(
                        -1,
                        1 + num_negatives,
                        -1,
                    )
                    .reshape(
                        -1,
                        EMBEDDING_DIM,
                    )
                )

                # --------------------------------------------
                # Similarity scores
                # --------------------------------------------

                scores = (
                    user_embeddings
                    * item_embeddings
                ).sum(dim=1)

                scores = scores.reshape(
                    batch_size,
                    1 + num_negatives,
                )

                # Positive item is always column 0.
                labels = torch.zeros(
                    batch_size,
                    dtype=torch.long,
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
            # Save best epoch by Recall@K
            # ------------------------------------------------

            if recall > best_recall:

                best_recall = recall

                best_epoch = epoch

                checkpoint = {
                    "model_state_dict":
                        model.state_dict(),
                    "model_version":
                        "two-tower-v2.1",
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
                    "num_negatives":
                        NUM_NEGATIVES,
                    "seed":
                        RANDOM_SEED,
                    "top_k":
                        TOP_K,
                }

                torch.save(
                    checkpoint,
                    MODEL_PATH,
                )

        # ----------------------------------------------------
        # 10. Final MLflow logging
        # ----------------------------------------------------

        mlflow.log_metric(
            f"best_recall_at_{TOP_K}",
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

        print(
            "\n=== Training Complete ==="
        )

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