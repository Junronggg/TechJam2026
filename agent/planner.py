"""Planner contract plus a deterministic implementation for architecture tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from experiment.schemas import BudgetState, ExperimentSpec, MetricBundle, ModelConfig, Operation


@dataclass(frozen=True)
class BranchSummary:
    node_id: str
    branch: str
    primary: float
    visits: int
    lesson: str = ""


@dataclass(frozen=True)
class PlannerContext:
    parent_id: str
    parent_config: ModelConfig
    parent_metrics: MetricBundle
    top_branches: tuple[BranchSummary, ...]
    lessons: tuple[str, ...]
    rejected_signatures: frozenset[str]
    allowed_operations: tuple[Operation, ...]
    budget: BudgetState


class Planner(Protocol):
    def propose(self, context: PlannerContext) -> ExperimentSpec:
        """Return one strictly structured experiment proposal."""


@dataclass(frozen=True)
class ExperimentTemplate:
    branch: str
    hypothesis: str
    operation: Operation
    parameters: Mapping[str, Any]
    expected_effect: Mapping[str, str]
    estimated_cost: str = "low"
    evidence: str = "Architecture smoke-test proposal"


class DeterministicPlanner:
    """A no-LLM planner used to prove orchestration before provider integration."""

    def __init__(self, templates: Sequence[ExperimentTemplate]) -> None:
        if not templates:
            raise ValueError("At least one experiment template is required")
        self._templates = tuple(templates)
        self._cursor = 0

    def propose(self, context: PlannerContext) -> ExperimentSpec:
        if self._cursor >= len(self._templates):
            raise StopIteration("The deterministic planner has no unused proposals")
        template = self._templates[self._cursor]
        self._cursor += 1
        experiment_number = context.budget.completed_iterations + 1
        return ExperimentSpec(
            experiment_id=f"exp_{experiment_number:03d}",
            parent_id=context.parent_id,
            branch=template.branch,
            hypothesis=template.hypothesis,
            operation=template.operation,
            parameters=dict(template.parameters),
            expected_effect=dict(template.expected_effect),
            estimated_cost=template.estimated_cost,
            evidence=template.evidence,
        )

