from __future__ import annotations

import torch
from torch import nn


class LightGCN(nn.Module):
    """LightGCN on the positive training user-item graph.

    Propagation contains no feature transforms or nonlinearities: normalized
    neighbour embeddings from each graph depth are averaged with layer zero.
    """

    def __init__(
        self, num_users: int, num_items: int, edge_users: torch.Tensor,
        edge_items: torch.Tensor, embedding_dim: int = 32, num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_layers = int(num_layers)
        self.user_embedding = nn.Embedding(self.num_users, int(embedding_dim))
        self.item_embedding = nn.Embedding(self.num_items, int(embedding_dim))
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        edge_users = edge_users.long()
        edge_items = edge_items.long()
        if edge_users.shape != edge_items.shape or edge_users.ndim != 1:
            raise ValueError("edge user/item arrays must be equal one-dimensional tensors")
        self.register_buffer("edge_users", edge_users)
        self.register_buffer("edge_items", edge_items)
        user_degree = torch.bincount(edge_users, minlength=self.num_users).float().clamp_min(1)
        item_degree = torch.bincount(edge_items, minlength=self.num_items).float().clamp_min(1)
        normalizer = (user_degree[edge_users] * item_degree[edge_items]).rsqrt()
        self.register_buffer("edge_normalizer", normalizer)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        users = self.user_embedding.weight
        items = self.item_embedding.weight
        user_sum, item_sum = users, items
        norm = self.edge_normalizer.unsqueeze(1)
        for _ in range(self.num_layers):
            next_users = torch.zeros_like(users)
            next_items = torch.zeros_like(items)
            next_users.index_add_(0, self.edge_users, items[self.edge_items] * norm)
            next_items.index_add_(0, self.edge_items, users[self.edge_users] * norm)
            users, items = next_users, next_items
            user_sum = user_sum + users
            item_sum = item_sum + items
        denominator = float(self.num_layers + 1)
        return user_sum / denominator, item_sum / denominator

    @staticmethod
    def score(
        users: torch.Tensor, items: torch.Tensor,
        propagated: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        user_embedding, item_embedding = propagated
        return (user_embedding[users.long()] * item_embedding[items.long()]).sum(dim=1)
