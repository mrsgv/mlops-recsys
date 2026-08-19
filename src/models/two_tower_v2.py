from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerV2(nn.Module):
    """
    Metadata-aware two-tower retrieval model.

    User tower:
        user ID -> embedding -> MLP

    Item tower:
        item ID
        + categorical metadata
        + numeric metadata
        + TF-IDF text features
        -> MLP
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        main_category_size: int,
        brand_size: int,
        store_size: int,
        price_bucket_size: int,
        text_dim: int,
        numeric_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        categorical_dim: int = 16,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # -----------------------------
        # User tower
        # -----------------------------

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        self.user_mlp = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

        # -----------------------------
        # Item categorical features
        # -----------------------------

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
        )

        self.main_category_embedding = nn.Embedding(
            main_category_size,
            categorical_dim,
        )

        self.brand_embedding = nn.Embedding(
            brand_size,
            categorical_dim,
        )

        self.store_embedding = nn.Embedding(
            store_size,
            categorical_dim,
        )

        self.price_bucket_embedding = nn.Embedding(
            price_bucket_size,
            categorical_dim,
        )

        # -----------------------------
        # Item feature projection
        # -----------------------------

        item_input_dim = (
            embedding_dim
            + categorical_dim * 4
            + numeric_dim
            + text_dim
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(
                item_input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def encode_users(
        self,
        user_idx: torch.Tensor,
    ) -> torch.Tensor:

        embedding = self.user_embedding(
            user_idx
        )

        output = self.user_mlp(
            embedding
        )

        return F.normalize(
            output,
            dim=1,
        )

    def encode_items(
        self,
        item_idx: torch.Tensor,
        main_category_idx: torch.Tensor,
        brand_idx: torch.Tensor,
        store_idx: torch.Tensor,
        price_bucket_idx: torch.Tensor,
        numeric_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:

        item_embedding = self.item_embedding(
            item_idx
        )

        main_category = (
            self.main_category_embedding(
                main_category_idx
            )
        )

        brand = self.brand_embedding(
            brand_idx
        )

        store = self.store_embedding(
            store_idx
        )

        price_bucket = (
            self.price_bucket_embedding(
                price_bucket_idx
            )
        )

        item_features = torch.cat(
            [
                item_embedding,
                main_category,
                brand,
                store,
                price_bucket,
                numeric_features,
                text_features,
            ],
            dim=1,
        )

        output = self.item_mlp(
            item_features
        )

        return F.normalize(
            output,
            dim=1,
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        main_category_idx: torch.Tensor,
        brand_idx: torch.Tensor,
        store_idx: torch.Tensor,
        price_bucket_idx: torch.Tensor,
        numeric_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:

        user_embeddings = self.encode_users(
            user_idx
        )

        item_embeddings = self.encode_items(
            item_idx,
            main_category_idx,
            brand_idx,
            store_idx,
            price_bucket_idx,
            numeric_features,
            text_features,
        )

        return (
            user_embeddings
            @ item_embeddings.T
        )