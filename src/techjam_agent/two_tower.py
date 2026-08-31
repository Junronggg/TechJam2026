from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .common import FeatureEmbedding, MLP


class TwoTowerCandidateModel(nn.Module):
    """User/context and candidate towers trained with shared ranking feedback."""

    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_dim: int = 16,
        numeric_dim: int = 0,
        tower_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.field_dims = [int(value) for value in field_dims]
        self.numeric_dim = int(numeric_dim)
        self.embedding_dim = int(embedding_dim)
        self.embedding = FeatureEmbedding(self.field_dims, self.embedding_dim)
        # user_id and tab are the stable/request-side fields. Candidate and
        # engineered fields live in the item tower; dense context is supplied
        # to both towers so request time can interact with candidate identity.
        self.user_fields = (0, 3)
        self.item_fields = tuple(index for index in range(len(self.field_dims)) if index not in self.user_fields)
        user_input = len(self.user_fields) * self.embedding_dim + self.numeric_dim
        item_input = len(self.item_fields) * self.embedding_dim + self.numeric_dim
        self.user_tower = MLP(user_input, (128,), dropout=dropout, output_dim=tower_dim)
        self.item_tower = MLP(item_input, (128,), dropout=dropout, output_dim=tower_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, categorical_x: torch.Tensor, numeric_x: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(categorical_x.long())
        user = embedded[:, self.user_fields, :].flatten(start_dim=1)
        item = embedded[:, self.item_fields, :].flatten(start_dim=1)
        if self.numeric_dim:
            if numeric_x is None or numeric_x.shape[1] != self.numeric_dim:
                raise ValueError(f"two-tower model expected {self.numeric_dim} dense features")
            dense = numeric_x.float()
            user = torch.cat((user, dense), dim=1)
            item = torch.cat((item, dense), dim=1)
        user = torch.nn.functional.normalize(self.user_tower(user), dim=1)
        item = torch.nn.functional.normalize(self.item_tower(item), dim=1)
        return (user * item).sum(dim=1) * 8.0 + self.bias
