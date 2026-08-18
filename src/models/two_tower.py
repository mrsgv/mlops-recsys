import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.user_tower = nn.Sequential(
            nn.Embedding(num_users, embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.item_tower = nn.Sequential(
            nn.Embedding(num_items, embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_users(self, user_idx):
        return F.normalize(self.user_tower(user_idx), dim=1)

    def encode_items(self, item_idx):
        return F.normalize(self.item_tower(item_idx), dim=1)

    def forward(self, user_idx, item_idx):
        user_embeddings = self.encode_users(user_idx)
        item_embeddings = self.encode_items(item_idx)

        return user_embeddings @ item_embeddings.T
