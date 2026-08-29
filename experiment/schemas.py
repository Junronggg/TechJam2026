"""Frozen interfaces shared by the agent, runner, and recommender layers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class Operation(str, Enum):
    CHANGE_HYPERPARAMETER = "CHANGE_HYPERPARAMETER"
    CHANGE_OBJECTIVE = "CHANGE_OBJECTIVE"
    ADD_FEATURE = "ADD_FEATURE"
    REMOVE_FEATURE = "REMOVE_FEATURE"
    CHANGE_MODEL = "CHANGE_MODEL"
    CHANGE_LOSS_WEIGHT = "CHANGE_LOSS_WEIGHT"
    ENSEMBLE = "ENSEMBLE"
    NOVEL_PATCH = "NOVEL_PATCH"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    PRUNED = "pruned"


class Decision(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    REPAIR = "repair"
    FOLLOW_UP = "follow_up"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class MetricBundle:
    gauc: float
    ndcg_at_5: float

    @property
    def primary(self) -> float:
        return (self.gauc + self.ndcg_at_5) / 2.0

    def to_dict(self) -> dict[str, float]:
        return {
            "GAUC": self.gauc,
            "nDCG@5": self.ndcg_at_5,
            "primary": self.primary,
        }


@dataclass(frozen=True)
class ModelConfig:
    model: str
    features: tuple[str, ...]
    hyperparameters: Mapping[str, float | int | str | bool]
    seed: int = 0
    objective: str = "pointwise"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "features": list(self.features),
            "hyperparameters": dict(self.hyperparameters),
            "seed": self.seed,
            "objective": self.objective,
        }

    def signature(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    parent_id: str
    branch: str
    hypothesis: str
    operation: Operation
    parameters: Mapping[str, Any]
    expected_effect: Mapping[str, str] = field(default_factory=dict)
    estimated_cost: str = "low"
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operation"] = self.operation.value
        return value

    def signature(self) -> str:
        comparable = {
            "parent_id": self.parent_id,
            "operation": self.operation.value,
            "parameters": self.parameters,
        }
        payload = json.dumps(comparable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: ExperimentStatus
    metrics: MetricBundle | None = None
    runtime_seconds: float = 0.0
    checkpoint: str | None = None
    prediction_path: str | None = None
    error: str | None = None
    recovery_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "runtime_seconds": self.runtime_seconds,
            "checkpoint": self.checkpoint,
            "prediction_path": self.prediction_path,
            "error": self.error,
            "recovery_attempts": self.recovery_attempts,
        }


@dataclass(frozen=True)
class CriticResult:
    observation: str
    interpretation: str
    confidence: Confidence
    decision: Decision
    next_test: str

    def to_dict(self) -> dict[str, str]:
        return {
            "observation": self.observation,
            "interpretation": self.interpretation,
            "confidence": self.confidence.value,
            "decision": self.decision.value,
            "next_test": self.next_test,
        }


@dataclass(frozen=True)
class RunBudget:
    max_iterations: int = 50
    max_wall_clock_seconds: float = 6 * 60 * 60
    convergence_epsilon: float = 0.002
    convergence_rounds: int = 3
    max_repair_attempts: int = 1


@dataclass(frozen=True)
class BudgetState:
    completed_iterations: int
    elapsed_seconds: float
    budget: RunBudget

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.budget.max_iterations - self.completed_iterations)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.budget.max_wall_clock_seconds - self.elapsed_seconds)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
