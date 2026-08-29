"""Preflight validation for safe, bounded experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from experiment.schemas import ExperimentSpec, ModelConfig, Operation


class ExperimentValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationPolicy:
    allowed_operations: frozenset[Operation]
    allowed_features: frozenset[str]
    allowed_models: frozenset[str]
    hyperparameter_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    protected_paths: tuple[str, ...] = (
        "kuairand-starter-kit/evaluate.py",
        "kuairand-starter-kit/baseline_scores.json",
    )

    @classmethod
    def safe_default(cls) -> "ValidationPolicy":
        return cls(
            allowed_operations=frozenset(
                {
                    Operation.CHANGE_HYPERPARAMETER,
                    Operation.ADD_FEATURE,
                    Operation.REMOVE_FEATURE,
                    Operation.CHANGE_MODEL,
                    Operation.CHANGE_LOSS_WEIGHT,
                }
            ),
            allowed_features=frozenset(
                {"user_id", "video_id", "author_id", "tab", "dur_bucket"}
            ),
            allowed_models=frozenset({"fm"}),
            hyperparameter_ranges={
                "k": (4, 128),
                "lr": (1e-5, 0.1),
                "epochs": (1, 100),
                "l2": (0.0, 1.0),
            },
        )


class ExperimentValidator:
    def __init__(self, policy: ValidationPolicy | None = None) -> None:
        self.policy = policy or ValidationPolicy.safe_default()

    def validate_spec(self, spec: ExperimentSpec) -> None:
        if spec.operation not in self.policy.allowed_operations:
            raise ExperimentValidationError(f"Operation is not enabled: {spec.operation.value}")
        if not spec.hypothesis.strip():
            raise ExperimentValidationError("Hypothesis must not be empty")
        if spec.estimated_cost not in {"low", "medium", "high"}:
            raise ExperimentValidationError("estimated_cost must be low, medium, or high")
        self._validate_finite(spec.parameters)
        self._validate_protected_paths(spec)

        required = {
            Operation.CHANGE_HYPERPARAMETER: {"name", "value"},
            Operation.ADD_FEATURE: {"feature"},
            Operation.REMOVE_FEATURE: {"feature"},
            Operation.CHANGE_MODEL: {"model"},
            Operation.CHANGE_LOSS_WEIGHT: {"task", "value"},
        }.get(spec.operation, set())
        missing = required.difference(spec.parameters)
        if missing:
            raise ExperimentValidationError(f"Missing operation parameters: {sorted(missing)}")

    def validate_config(self, config: ModelConfig) -> None:
        if config.model not in self.policy.allowed_models:
            raise ExperimentValidationError(f"Model is not registered as available: {config.model}")
        unknown_features = set(config.features).difference(self.policy.allowed_features)
        if unknown_features:
            raise ExperimentValidationError(
                f"Features are not registered as available: {sorted(unknown_features)}"
            )
        if len(config.features) != len(set(config.features)):
            raise ExperimentValidationError("Duplicate features are not allowed")
        self._validate_finite(config.hyperparameters)
        for name, value in config.hyperparameters.items():
            if name not in self.policy.hyperparameter_ranges:
                continue
            if not isinstance(value, (int, float)):
                raise ExperimentValidationError(f"Hyperparameter {name} must be numeric")
            lower, upper = self.policy.hyperparameter_ranges[name]
            if not lower <= float(value) <= upper:
                raise ExperimentValidationError(
                    f"Hyperparameter {name}={value} is outside [{lower}, {upper}]"
                )

    def _validate_protected_paths(self, spec: ExperimentSpec) -> None:
        for key, value in spec.parameters.items():
            if "path" not in str(key).lower() or not isinstance(value, str):
                continue
            candidate = Path(value).as_posix().lower()
            for protected in self.policy.protected_paths:
                if candidate.endswith(Path(protected).as_posix().lower()):
                    raise ExperimentValidationError(f"Protected file cannot be modified: {value}")

    def _validate_finite(self, values: Mapping[str, object]) -> None:
        for key, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ExperimentValidationError(f"Non-finite value for {key}")

