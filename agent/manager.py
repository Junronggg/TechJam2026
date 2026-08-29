"""Budgeted Planner/Runner/Critic loop coordinated by lightweight tree search."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from agent.critic import Critic
from agent.memory import ExperimentNode, ResearchMemory
from agent.planner import BranchSummary, Planner, PlannerContext
from agent.tree import TreeSearchPolicy
from experiment.logger import ExperimentLogger
from experiment.runner import ExperimentRunner
from experiment.schemas import (
    BudgetState,
    ExperimentResult,
    ExperimentStatus,
    MetricBundle,
    ModelConfig,
    RunBudget,
    write_json,
)
from recommender.config import UnsupportedOperation, apply_experiment


@dataclass(frozen=True)
class RunSummary:
    stop_reason: str
    completed_iterations: int
    elapsed_seconds: float
    best_node_id: str
    best_metrics: MetricBundle
    manual_interventions: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "stop_reason": self.stop_reason,
            "completed_iterations": self.completed_iterations,
            "elapsed_seconds": self.elapsed_seconds,
            "best_node_id": self.best_node_id,
            "best_metrics": self.best_metrics.to_dict(),
            "manual_interventions": self.manual_interventions,
            "llm_tokens": {
                "input": self.llm_input_tokens,
                "output": self.llm_output_tokens,
                "total": self.llm_input_tokens + self.llm_output_tokens,
            },
        }


class ResearchManager:
    def __init__(
        self,
        planner: Planner,
        critic: Critic,
        runner: ExperimentRunner,
        logger: ExperimentLogger,
        budget: RunBudget | None = None,
        memory: ResearchMemory | None = None,
        tree_policy: TreeSearchPolicy | None = None,
    ) -> None:
        self.planner = planner
        self.critic = critic
        self.runner = runner
        self.logger = logger
        self.budget = budget or RunBudget()
        self.memory = memory or ResearchMemory()
        self.tree_policy = tree_policy or TreeSearchPolicy()

    def run(
        self,
        baseline_config: ModelConfig,
        baseline_metrics: MetricBundle,
        baseline_result: ExperimentResult | None = None,
        initial_elapsed_seconds: float = 0.0,
    ) -> RunSummary:
        started = time.monotonic()
        if not self.memory.nodes:
            baseline_result = baseline_result or ExperimentResult(
                experiment_id="baseline", status=ExperimentStatus.SUCCESS, metrics=baseline_metrics
            )
            if baseline_result.metrics is None:
                raise ValueError("Baseline result must contain validation metrics")
            root = self.memory.add_root(baseline_config, baseline_result)
            self.logger.append({"event": "baseline", "node": root.to_dict()})
            self._persist_tree()

        completed = 0
        best_improvements: list[float] = []
        rejected_signatures: set[str] = set()
        stop_reason = "iteration_budget"

        while completed < self.budget.max_iterations:
            elapsed = initial_elapsed_seconds + (time.monotonic() - started)
            if elapsed >= self.budget.max_wall_clock_seconds:
                stop_reason = "wall_clock_budget"
                break
            budget_state = BudgetState(completed, elapsed, self.budget)
            selection = self.tree_policy.select(self.memory, budget_state.remaining_seconds)
            parent = selection.node
            context = self._planner_context(parent, budget_state, rejected_signatures)

            try:
                spec = self.planner.propose(context)
            except StopIteration:
                stop_reason = "planner_exhausted"
                break

            try:
                preview = apply_experiment(parent.config, spec)
            except (UnsupportedOperation, KeyError, ValueError):
                preview = None
            if preview is not None and self.memory.contains_config(preview):
                rejected_signatures.add(preview.signature())
                self.logger.append(
                    {
                        "event": "duplicate_rejected",
                        "experiment": spec.to_dict(),
                        "candidate_config_signature": preview.signature(),
                    }
                )
                continue

            best_before = self.memory.best_node().primary
            outcome = self.runner.run(parent.config, parent.result.metrics, spec)
            critique = self.critic.review(parent.result.metrics, outcome.result, spec)
            node = self.memory.add_child(
                parent.node_id, outcome.config, spec, outcome.result, critique
            )
            completed += 1
            best_after = self.memory.best_node().primary
            best_improvements.append(max(0.0, best_after - best_before))
            self.logger.append(
                {
                    "event": "experiment_completed",
                    "selection": {
                        "parent_id": parent.node_id,
                        "priority": selection.priority,
                        "exploitation": selection.exploitation,
                        "exploration_bonus": selection.exploration_bonus,
                        "novelty_bonus": selection.novelty_bonus,
                        "runtime_penalty": selection.runtime_penalty,
                    },
                    "config_diff": self._config_diff(parent.config, outcome.config),
                    "node": node.to_dict(),
                }
            )
            self._persist_tree()
            if self._converged(best_improvements):
                stop_reason = "converged"
                break

        elapsed = initial_elapsed_seconds + (time.monotonic() - started)
        best = self.memory.best_node()
        if best.result.metrics is None:
            raise RuntimeError("Best node unexpectedly has no metrics")
        summary = RunSummary(
            stop_reason=stop_reason,
            completed_iterations=completed,
            elapsed_seconds=elapsed,
            best_node_id=best.node_id,
            best_metrics=best.result.metrics,
        )
        write_json(self.logger.log_path.parent / "final_summary.json", summary.to_dict())
        return summary

    def _persist_tree(self) -> None:
        write_json(self.logger.log_path.parent / "tree_snapshot.json", self.memory.to_dict())

    def _planner_context(
        self,
        parent: ExperimentNode,
        budget: BudgetState,
        rejected_signatures: set[str],
    ) -> PlannerContext:
        frontier = self.tree_policy.frontier(self.memory)
        summaries = tuple(
            BranchSummary(
                node_id=node.node_id,
                branch=node.branch,
                primary=node.primary,
                visits=node.visits,
                lesson=node.critic.observation if node.critic else "",
            )
            for node in frontier
        )
        if parent.result.metrics is None:
            raise RuntimeError("Planner parent must have successful metrics")
        return PlannerContext(
            parent_id=parent.node_id,
            parent_config=parent.config,
            parent_metrics=parent.result.metrics,
            top_branches=summaries,
            lessons=self.memory.lessons(),
            rejected_signatures=frozenset(rejected_signatures),
            allowed_operations=tuple(self.runner.validator.policy.allowed_operations),
            budget=budget,
        )

    def _converged(self, improvements: list[float]) -> bool:
        rounds = self.budget.convergence_rounds
        if len(improvements) < rounds:
            return False
        return all(value <= self.budget.convergence_epsilon for value in improvements[-rounds:])

    @staticmethod
    def _config_diff(parent: ModelConfig, child: ModelConfig) -> dict[str, object]:
        return {
            "model": {"before": parent.model, "after": child.model}
            if parent.model != child.model
            else None,
            "features_added": sorted(set(child.features) - set(parent.features)),
            "features_removed": sorted(set(parent.features) - set(child.features)),
            "hyperparameters": {
                key: {"before": parent.hyperparameters.get(key), "after": value}
                for key, value in child.hyperparameters.items()
                if parent.hyperparameters.get(key) != value
            },
        }
