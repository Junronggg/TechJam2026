from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .common import FeatureEmbedding


class MultiTaskRecommender(nn.Module):
    """
    Shared-bottom multi-task recommender.

    The same user/item/context representation is shared across
    multiple prediction heads.

    Default tasks:

        long_view
        click
        like

    IMPORTANT:
    click/like/current-row feedback are TRAINING TARGETS.

    They must NOT be fed into the model as current-row input
    features.

    Expected input
    --------------
    categorical_x:

        LongTensor
        [batch_size, num_fields]

    Output
    ------
    Dictionary:

        {
            "long_view": Tensor [batch],
            "click": Tensor [batch],
            "like": Tensor [batch],
        }

    All values are RAW LOGITS.
    """

    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_dim: int = 16,
        hidden_dims: Sequence[int] = (
            128,
            64,
        ),
        tasks: Sequence[str] = (
            "long_view",
            "click",
            "like",
        ),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(field_dims) == 0:
            raise ValueError(
                "MultiTaskRecommender requires "
                "at least one feature field"
            )

        if len(tasks) == 0:
            raise ValueError(
                "at least one task is required"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        self.field_dims = [
            int(value)
            for value in field_dims
        ]

        self.embedding_dim = int(
            embedding_dim
        )

        self.tasks = list(tasks)

        if "long_view" not in self.tasks:
            raise ValueError(
                "long_view must be one of the tasks"
            )

        #
        # Shared categorical embeddings.
        #
        self.embedding = FeatureEmbedding(
            self.field_dims,
            self.embedding_dim,
        )

        input_dim = (
            len(self.field_dims)
            * self.embedding_dim
        )

        #
        # Shared bottom network.
        #
        shared_layers: list[nn.Module] = []

        current_dim = input_dim

        for hidden_dim in hidden_dims:

            hidden_dim = int(
                hidden_dim
            )

            shared_layers.extend(
                [
                    nn.Linear(
                        current_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Dropout(
                        dropout
                    ),
                ]
            )

            current_dim = hidden_dim

        self.shared = nn.Sequential(
            *shared_layers
        )

        #
        # One output head for every task.
        #
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(
                    current_dim,
                    1,
                )
                for task in self.tasks
            }
        )

        for head in self.heads.values():
            nn.init.xavier_uniform_(
                head.weight
            )

            nn.init.zeros_(
                head.bias
            )

    def shared_representation(
        self,
        categorical_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the shared representation before task heads.
        """

        if categorical_x.ndim != 2:
            raise ValueError(
                "categorical_x must have shape "
                "[batch_size, num_fields]"
            )

        if (
            categorical_x.shape[1]
            != len(self.field_dims)
        ):
            raise ValueError(
                f"expected {len(self.field_dims)} "
                f"feature fields but received "
                f"{categorical_x.shape[1]}"
            )

        embedded = self.embedding(
            categorical_x.long()
        )

        flattened = embedded.flatten(
            start_dim=1
        )

        return self.shared(
            flattened
        )

    def forward(
        self,
        categorical_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        shared = (
            self.shared_representation(
                categorical_x
            )
        )

        return {
            task: (
                head(shared)
                .squeeze(-1)
            )
            for task, head
            in self.heads.items()
        }


def multitask_bce_loss(
    outputs: Mapping[
        str,
        torch.Tensor,
    ],
    targets: Mapping[
        str,
        torch.Tensor,
    ],
    weights: Mapping[
        str,
        float,
    ] | None = None,
) -> torch.Tensor:
    """
    Weighted BCE loss across all available tasks.

    Suggested initial weights:

        long_view = 1.0
        click     = 0.2
        like      = 0.1

    Example:

        loss = multitask_bce_loss(
            outputs,
            targets,
            {
                "long_view": 1.0,
                "click": 0.2,
                "like": 0.1,
            }
        )
    """

    if not outputs:
        raise ValueError(
            "outputs cannot be empty"
        )

    if weights is None:
        weights = {
            task: 1.0
            for task in outputs
        }

    device = next(
        iter(outputs.values())
    ).device

    total = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    )

    used_tasks = 0

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    for task, logits in outputs.items():

        if task not in targets:
            continue

        target = targets[
            task
        ].float().to(
            logits.device
        )

        if target.shape != logits.shape:
            raise ValueError(
                f"target shape for "
                f"{task!r} is "
                f"{target.shape}, "
                f"expected "
                f"{logits.shape}"
            )

        weight = float(
            weights.get(
                task,
                1.0,
            )
        )

        total = (
            total
            + weight
            * criterion(
                logits,
                target,
            )
        )

        used_tasks += 1

    if used_tasks == 0:
        raise ValueError(
            "no matching task targets "
            "were supplied"
        )

    return total


def ranking_aware_multitask_loss(
    positive_outputs: Mapping[
        str,
        torch.Tensor,
    ],
    negative_outputs: Mapping[
        str,
        torch.Tensor,
    ],
    positive_auxiliary_targets: Mapping[
        str,
        torch.Tensor,
    ],
    auxiliary_weights: Mapping[
        str,
        float,
    ],
) -> torch.Tensor:
    """
    Experimental multi-task ranking objective.

    Main target:
        long_view → BPR

    Auxiliary targets:
        click / like → BCE

    This should only be used AFTER simple multi-task BCE works.

    NOTE:
    Auxiliary BCE targets here correspond to the positive rows.
    """

    if (
        "long_view"
        not in positive_outputs
        or "long_view"
        not in negative_outputs
    ):
        raise ValueError(
            "long_view output is required "
            "for ranking-aware multitask loss"
        )

    positive_score = (
        positive_outputs[
            "long_view"
        ]
    )

    negative_score = (
        negative_outputs[
            "long_view"
        ]
    )

    if (
        positive_score.shape
        != negative_score.shape
    ):
        raise ValueError(
            "positive and negative "
            "long_view scores must "
            "have the same shape"
        )

    #
    # Main long_view BPR objective.
    #
    total = (
        -torch.nn.functional.logsigmoid(
            positive_score
            - negative_score
        )
        .mean()
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    #
    # Auxiliary targets.
    #
    for task, weight in (
        auxiliary_weights.items()
    ):

        if task == "long_view":
            continue

        if (
            task not in positive_outputs
            or task
            not in positive_auxiliary_targets
        ):
            continue

        logits = (
            positive_outputs[
                task
            ]
        )

        target = (
            positive_auxiliary_targets[
                task
            ]
            .float()
            .to(logits.device)
        )

        if target.shape != logits.shape:
            raise ValueError(
                f"target shape for "
                f"{task!r} does not match "
                "the model output"
            )

        total = (
            total
            + float(weight)
            * criterion(
                logits,
                target,
            )
        )

    return total