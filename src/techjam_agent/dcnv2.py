from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .common import FeatureEmbedding, MLP


class CrossLayer(nn.Module):
    """
    One cross layer.

    x0 is the original embedded feature vector.
    xl is the representation from the previous cross layer.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            input_dim,
            bias=True,
        )

        nn.init.xavier_uniform_(
            self.linear.weight
        )

        nn.init.zeros_(
            self.linear.bias
        )

    def forward(
        self,
        x0: torch.Tensor,
        xl: torch.Tensor,
    ) -> torch.Tensor:
        return x0 * self.linear(xl) + xl


class CrossNetwork(nn.Module):
    """
    Stack of cross layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CrossLayer(input_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x0 = x
        xl = x

        for layer in self.layers:
            xl = layer(x0, xl)

        return xl


class DCNv2(nn.Module):
    """
    Lightweight Deep & Cross Network recommender.

    Input
    -----
    categorical_x:
        LongTensor with shape

            [batch_size, number_of_fields]

    Example fields:

        user_id
        video_id
        author_id
        tab
        duration_bucket

    Output
    ------
    Raw ranking score/logit with shape

        [batch_size]

    Do NOT apply sigmoid inside the model.

    This allows the model to support both:

        BCEWithLogitsLoss
        BPR loss
    """

    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_dim: int = 16,
        cross_layers: int = 3,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        numeric_dim: int = 0,
    ) -> None:
        super().__init__()

        self.field_dims = [
            int(value)
            for value in field_dims
        ]

        self.embedding_dim = int(
            embedding_dim
        )

        self.embedding = FeatureEmbedding(
            self.field_dims,
            self.embedding_dim,
        )

        self.numeric_dim = int(numeric_dim)
        input_dim = (
            len(self.field_dims)
            * self.embedding_dim
        ) + self.numeric_dim

        self.cross = CrossNetwork(
            input_dim=input_dim,
            num_layers=int(cross_layers),
        )

        #
        # We use MLP to generate a deep representation.
        #
        # Its final output dimension is hidden_dims[-1].
        #
        self.deep = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            dropout=dropout,
            output_dim=hidden_dims[-1],
        )

        final_dim = (
            input_dim
            + hidden_dims[-1]
        )

        self.output = nn.Linear(
            final_dim,
            1,
        )

        nn.init.xavier_uniform_(
            self.output.weight
        )

        nn.init.zeros_(
            self.output.bias
        )

    def forward(
        self,
        categorical_x: torch.Tensor,
        numeric_x: torch.Tensor | None = None,
    ) -> torch.Tensor:

        embedded = self.embedding(
            categorical_x
        )

        #
        # [batch, fields, embedding]
        # ->
        # [batch, fields * embedding]
        #
        x = embedded.flatten(
            start_dim=1
        )
        if self.numeric_dim:
            if numeric_x is None or numeric_x.ndim != 2 or numeric_x.shape[1] != self.numeric_dim:
                raise ValueError(f"DCNv2 expected {self.numeric_dim} dense features")
            x = torch.cat((x, numeric_x.float()), dim=1)

        cross_output = self.cross(x)

        deep_output = self.deep(x)

        combined = torch.cat(
            [
                cross_output,
                deep_output,
            ],
            dim=1,
        )

        score = self.output(
            combined
        )

        return score.squeeze(-1)
