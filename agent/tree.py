"""Lightweight branch-preserving selection policy (not full MCTS)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent.memory import ExperimentNode, ResearchMemory
from experiment.schemas import Decision


@dataclass(frozen=True)
class TreePolicyConfig:
    max_active_branches: int = 3
    exploration_weight: float = 0.002
    novelty_weight: float = 0.001
    runtime_penalty_weight: float = 0.001


@dataclass(frozen=True)
class Selection:
    node: ExperimentNode
    priority: float
    exploitation: float
    exploration_bonus: float
    novelty_bonus: float
    runtime_penalty: float


class TreeSearchPolicy:
    def __init__(self, config: TreePolicyConfig | None = None) -> None:
        self.config = config or TreePolicyConfig()

    def frontier(self, memory: ResearchMemory) -> tuple[ExperimentNode, ...]:
        """Keep the strongest node from each of the best few research branches."""
        by_branch: dict[str, ExperimentNode] = {}
        for node in memory.successful_nodes():
            if node.critic is not None and node.critic.decision not in {
                Decision.KEEP,
                Decision.FOLLOW_UP,
            }:
                continue
            current = by_branch.get(node.branch)
            if current is None or node.primary > current.primary:
                by_branch[node.branch] = node
        ranked = sorted(by_branch.values(), key=lambda node: node.primary, reverse=True)
        return tuple(ranked[: self.config.max_active_branches])

    def select(self, memory: ResearchMemory, remaining_seconds: float) -> Selection:
        frontier = self.frontier(memory)
        if not frontier:
            raise RuntimeError("Tree has no expandable successful nodes")
        total_visits = sum(node.visits for node in frontier) + 1
        branch_counts: dict[str, int] = {}
        for node in memory.nodes.values():
            branch_counts[node.branch] = branch_counts.get(node.branch, 0) + 1

        selections = []
        time_scale = max(1.0, remaining_seconds)
        for node in frontier:
            exploration = self.config.exploration_weight * math.sqrt(
                math.log(total_visits + 1) / (node.visits + 1)
            )
            novelty = self.config.novelty_weight / branch_counts[node.branch]
            runtime = max(0.0, node.result.runtime_seconds)
            runtime_penalty = self.config.runtime_penalty_weight * runtime / time_scale
            priority = node.primary + exploration + novelty - runtime_penalty
            selections.append(
                Selection(node, priority, node.primary, exploration, novelty, runtime_penalty)
            )
        selected = max(selections, key=lambda value: value.priority)
        selected.node.visits += 1
        return selected
