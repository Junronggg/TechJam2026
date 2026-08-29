"""Stable training boundary; the actual model adapter is intentionally not built yet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from experiment.schemas import ModelConfig


@dataclass(frozen=True)
class TrainingOutput:
    prediction_path: Path
    checkpoint_path: Path | None


class Trainer(Protocol):
    def train_model(self, config: ModelConfig, run_dir: Path) -> TrainingOutput:
        """Fit on training data and emit validation predictions."""


def train_model(config: ModelConfig, run_dir: Path) -> TrainingOutput:
    raise NotImplementedError(
        "The architecture is ready, but no recommender adapter is connected. "
        "Wire the untouched official FM baseline here in the next build phase."
    )

