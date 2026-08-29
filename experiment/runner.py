"""Canonical experiment orchestration with a swappable execution backend."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from experiment.schemas import (
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    MetricBundle,
    ModelConfig,
    write_json,
)
from experiment.validator import ExperimentValidationError, ExperimentValidator
from recommender.config import UnsupportedOperation, apply_experiment


class ExperimentBackend(Protocol):
    def execute(
        self,
        spec: ExperimentSpec,
        config: ModelConfig,
        parent_metrics: MetricBundle,
        run_dir: Path,
    ) -> ExperimentResult:
        """Train, predict, and evaluate one already-validated candidate."""


@dataclass(frozen=True)
class RunOutcome:
    config: ModelConfig
    result: ExperimentResult
    run_dir: Path


class ExperimentRunner:
    def __init__(
        self,
        runs_dir: Path,
        backend: ExperimentBackend,
        validator: ExperimentValidator | None = None,
    ) -> None:
        self.runs_dir = runs_dir
        self.backend = backend
        self.validator = validator or ExperimentValidator()

    def run(
        self,
        parent_config: ModelConfig,
        parent_metrics: MetricBundle,
        spec: ExperimentSpec,
    ) -> RunOutcome:
        run_dir = self.runs_dir / spec.experiment_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "spec.json", spec.to_dict())
        candidate = parent_config
        try:
            self.validator.validate_spec(spec)
            candidate = apply_experiment(parent_config, spec)
            self.validator.validate_config(candidate)
            write_json(run_dir / "config.json", candidate.to_dict())
            result = self.backend.execute(spec, candidate, parent_metrics, run_dir)
        except (ExperimentValidationError, UnsupportedOperation, KeyError, ValueError) as exc:
            result = ExperimentResult(
                experiment_id=spec.experiment_id,
                status=ExperimentStatus.REJECTED,
                error=str(exc),
            )
        except Exception:
            result = ExperimentResult(
                experiment_id=spec.experiment_id,
                status=ExperimentStatus.FAILED,
                error=traceback.format_exc(),
            )
        write_json(run_dir / "result.json", result.to_dict())
        return RunOutcome(config=candidate, result=result, run_dir=run_dir)


class DryRunBackend:
    """Deterministic fake metrics for testing control flow; never trains a model."""

    def __init__(self, branch_deltas: Mapping[str, tuple[float, float]] | None = None) -> None:
        self.branch_deltas = dict(
            branch_deltas
            or {
                "capacity": (0.0010, 0.0004),
                "optimization": (0.0025, 0.0020),
                "feature_ablation": (-0.0010, -0.0020),
            }
        )

    def execute(
        self,
        spec: ExperimentSpec,
        config: ModelConfig,
        parent_metrics: MetricBundle,
        run_dir: Path,
    ) -> ExperimentResult:
        del config, run_dir
        started = time.monotonic()
        gauc_delta, ndcg_delta = self.branch_deltas.get(spec.branch, (0.0, 0.0))
        metrics = MetricBundle(
            gauc=parent_metrics.gauc + gauc_delta,
            ndcg_at_5=parent_metrics.ndcg_at_5 + ndcg_delta,
        )
        return ExperimentResult(
            experiment_id=spec.experiment_id,
            status=ExperimentStatus.SUCCESS,
            metrics=metrics,
            runtime_seconds=time.monotonic() - started,
            checkpoint=f"dry-run://{spec.experiment_id}",
        )
