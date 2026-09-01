from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class FeatureEmbedding(nn.Module):
    """
    Shared embedding layer for multiple categorical fields.

    Input:
        x: LongTensor of shape [batch_size, num_fields]

    Each column uses IDs local to that particular field.

    Example:
        field 0 = user_id
        field 1 = video_id
        field 2 = author_id
        field 3 = tab
        field 4 = duration bucket

    field_dims contains the vocabulary size of every field.
    """

    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.field_dims = [int(x) for x in field_dims]
        self.embedding_dim = int(embedding_dim)

        offsets = [0]

        for dim in self.field_dims[:-1]:
            offsets.append(offsets[-1] + dim)

        self.register_buffer(
            "offsets",
            torch.tensor(
                offsets,
                dtype=torch.long,
            ),
        )

        self.embedding = nn.Embedding(
            sum(self.field_dims),
            self.embedding_dim,
        )

        nn.init.xavier_uniform_(
            self.embedding.weight
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.dtype != torch.long:
            x = x.long()

        x = x + self.offsets.unsqueeze(0)

        return self.embedding(x)


class FeatureLinear(nn.Module):
    """
    First-order linear term for categorical fields.

    Equivalent to learning one scalar weight for every
    categorical value plus a global bias.
    """

    def __init__(
        self,
        field_dims: Sequence[int],
    ) -> None:
        super().__init__()

        self.field_dims = [int(x) for x in field_dims]

        offsets = [0]

        for dim in self.field_dims[:-1]:
            offsets.append(offsets[-1] + dim)

        self.register_buffer(
            "offsets",
            torch.tensor(
                offsets,
                dtype=torch.long,
            ),
        )

        self.fc = nn.Embedding(
            sum(self.field_dims),
            1,
        )

        self.bias = nn.Parameter(
            torch.zeros(1)
        )

        nn.init.zeros_(
            self.fc.weight
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.dtype != torch.long:
            x = x.long()

        x = x + self.offsets.unsqueeze(0)

        return (
            self.fc(x).sum(dim=1)
            + self.bias
        )


class MLP(nn.Module):
    """
    Generic multilayer perceptron used by DeepFM,
    DCNv2, and other neural recommendation models.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        current_dim = int(input_dim)

        for hidden_dim in hidden_dims:

            hidden_dim = int(hidden_dim)

            layers.extend(
                [
                    nn.Linear(
                        current_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

            current_dim = hidden_dim

        layers.append(
            nn.Linear(
                current_dim,
                output_dim,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)