from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .common import FeatureEmbedding, FeatureLinear, MLP


class DeepFM(nn.Module):
    """
    DeepFM recommender for categorical features.

    The model combines:

    1. First-order linear feature effects
    2. FM pairwise feature interactions
    3. A deep neural-network branch for nonlinear interactions

    Expected input
    --------------
    categorical_x:
        LongTensor of shape:

            [batch_size, num_fields]

    Example fields:

        user_id
        video_id
        author_id
        tab
        duration_bucket

    Output
    ------
    Raw score/logit:

        [batch_size]

    IMPORTANT:
    Do NOT apply sigmoid inside the model.

    This allows the same scoring function to be used with:

        BCEWithLogitsLoss
        BPR pairwise loss
    """

    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_dim: int = 16,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(field_dims) == 0:
            raise ValueError(
                "DeepFM requires at least one feature field"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero"
            )

        self.field_dims = [
            int(value)
            for value in field_dims
        ]

        self.embedding_dim = int(
            embedding_dim
        )

        self.num_fields = len(
            self.field_dims
        )

        #
        # First-order linear terms.
        #
        self.linear = FeatureLinear(
            self.field_dims
        )

        #
        # Shared embeddings used by both:
        #
        #   FM branch
        #   Deep branch
        #
        self.embedding = FeatureEmbedding(
            self.field_dims,
            self.embedding_dim,
        )

        deep_input_dim = (
            self.num_fields
            * self.embedding_dim
        )

        #
        # Deep branch.
        #
        self.deep = MLP(
            input_dim=deep_input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=1,
        )

    @staticmethod
    def _fm_interaction(
        embedded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the standard FM second-order interaction term.

        embedded:
            [batch, fields, embedding_dim]

        FM identity:

            0.5 * [
                (sum v_i)^2
                -
                sum(v_i^2)
            ]
        """

        summed_embeddings = (
            embedded.sum(dim=1)
        )

        square_of_sum = (
            summed_embeddings.pow(2)
        )

        sum_of_square = (
            embedded.pow(2).sum(dim=1)
        )

        pairwise_vector = (
            0.5
            * (
                square_of_sum
                - sum_of_square
            )
        )

        #
        # Reduce embedding dimension:
        #
        # [batch, embedding_dim]
        # ->
        # [batch, 1]
        #
        return pairwise_vector.sum(
            dim=1,
            keepdim=True,
        )

    def forward(
        self,
        categorical_x: torch.Tensor,
    ) -> torch.Tensor:

        if categorical_x.ndim != 2:
            raise ValueError(
                "categorical_x must have shape "
                "[batch_size, num_fields]"
            )

        if (
            categorical_x.shape[1]
            != self.num_fields
        ):
            raise ValueError(
                f"DeepFM expected "
                f"{self.num_fields} fields, "
                f"received "
                f"{categorical_x.shape[1]}"
            )

        categorical_x = (
            categorical_x.long()
        )

        #
        # First-order score.
        #
        linear_score = self.linear(
            categorical_x
        )

        #
        # [batch, fields, embedding_dim]
        #
        embedded = self.embedding(
            categorical_x
        )

        #
        # Second-order FM interaction.
        #
        fm_score = self._fm_interaction(
            embedded
        )

        #
        # Deep branch.
        #
        deep_input = embedded.flatten(
            start_dim=1
        )

        deep_score = self.deep(
            deep_input
        )

        #
        # Combine all three components.
        #
        score = (
            linear_score
            + fm_score
            + deep_score
        )

        return score.squeeze(-1)


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.

    For a user:

        score(positive item)
        >
        score(negative item)

    Loss:

        -log sigmoid(
            positive_score - negative_score
        )
    """

    if (
        positive_scores.shape
        != negative_scores.shape
    ):
        raise ValueError(
            "positive_scores and negative_scores "
            "must have the same shape"
        )

    return (
        -torch.nn.functional.logsigmoid(
            positive_scores
            - negative_scores
        )
        .mean()
    )