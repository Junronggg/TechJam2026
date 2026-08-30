from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TreePolicyConfig:
    """Weights for the lightweight, branch-preserving search policy.

    The values deliberately live on the same scale as validation Primary.  A
    clearly better node therefore wins on exploitation, while nodes within the
    convergence epsilon can still receive a turn because of exploration and
    novelty.
    """

    max_active_branches: int = 3
    exploration_weight: float = 0.002
    novelty_weight: float = 0.001
    runtime_penalty_weight: float = 0.001
    repetition_penalty_weight: float = 0.001
    failed_child_penalty_weight: float = 0.002
    rejected_node_penalty_weight: float = 0.002


def branch_name(changes: dict[str, Any]) -> str:
    if not changes:
        return "baseline"
    feature_keys = {
        "user_long_view_rate",
        "item_long_view_rate",
        "continuous_history_stats",
        "user_tab_long_view_rate",
        "user_tab_cross",
        "user_author_cross",
        "user_recent_3d_activity",
        "item_recent_3d_exposure",
        "prior_video_positive",
        "author_positive_recency",
        "prior_video_count",
        "previous_author_same",
        "global_context",
    }
    if feature_keys.intersection(changes):
        return "features"
    if "model" in changes:
        return "model"
    if "training_objective" in changes:
        return "ranking_objective"
    return "optimization"


def node_id_for(iteration: int | None) -> str | None:
    if iteration is None:
        return None
    return "baseline" if iteration == 0 else f"exp_{iteration:03d}"


@dataclass(frozen=True)
class ExperimentParent:
    """Controller-owned lineage handle for the node an experiment expands."""

    node_id: str
    iteration: int
    config: dict[str, Any]
    primary: float
    branch: str

    def as_record(self) -> dict[str, Any]:
        return {"parent_id": self.node_id, "parent_iteration": self.iteration,
                "parent_primary": self.primary, "parent_branch": self.branch}


@dataclass(frozen=True)
class ParentSelection:
    parent: ExperimentParent
    priority: float
    exploitation: float
    exploration_bonus: float
    novelty_bonus: float
    runtime_penalty: float
    repetition_penalty: float
    failed_child_penalty: float
    rejected_node_penalty: float
    visits: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent.node_id,
            "parent_branch": self.parent.branch,
            "priority": self.priority,
            "exploitation": self.exploitation,
            "exploration_bonus": self.exploration_bonus,
            "novelty_bonus": self.novelty_bonus,
            "runtime_penalty": self.runtime_penalty,
            "repetition_penalty": self.repetition_penalty,
            "failed_child_penalty": self.failed_child_penalty,
            "rejected_node_penalty": self.rejected_node_penalty,
            "visits": self.visits,
        }


def _finite_primary(item: dict[str, Any]) -> float | None:
    metrics = item.get("metrics")
    if not isinstance(metrics, dict) or "primary" not in metrics:
        return None
    try:
        value = float(metrics["primary"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def select_parent(history: list[dict[str, Any]] | None) -> ExperimentParent | None:
    """Phase A policy: expand the global-best successful node, ties going to the earliest.

    Selection is deterministic and uses validation Primary only, so the iteration-0
    baseline is the parent until a candidate strictly beats it.
    """
    best: ExperimentParent | None = None
    for item in history or []:
        if (not isinstance(item, dict) or item.get("status") != "success"
                or item.get("decision") == "CONTROL"):
            continue
        primary = _finite_primary(item)
        config = item.get("config")
        iteration = item.get("iteration")
        if primary is None or not isinstance(config, dict) or not isinstance(iteration, int):
            continue
        if best is not None and primary <= best.primary:
            continue
        changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
        best = ExperimentParent(node_id_for(iteration), iteration, config, primary,
                                branch_name(changes))
    return best


def _successful_parents(history: list[dict[str, Any]] | None) -> list[ExperimentParent]:
    parents: list[ExperimentParent] = []
    for item in history or []:
        if (not isinstance(item, dict) or item.get("status") != "success"
                or item.get("decision") == "CONTROL"):
            continue
        primary = _finite_primary(item)
        config = item.get("config")
        iteration = item.get("iteration")
        if primary is None or not isinstance(config, dict) or not isinstance(iteration, int):
            continue
        changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
        parents.append(ExperimentParent(node_id_for(iteration), iteration, config, primary,
                                        branch_name(changes)))
    return parents


class TreeSearchPolicy:
    """Select an expansion parent from up to three active research branches.

    This is intentionally a small best-first tree policy rather than full MCTS.
    It balances measured validation quality with under-explored branches and
    penalizes expensive, repetitive, rejected, or failure-producing paths.
    """

    def __init__(self, config: TreePolicyConfig | None = None) -> None:
        self.config = config or TreePolicyConfig()
        if self.config.max_active_branches < 1:
            raise ValueError("max_active_branches must be at least one")

    def frontier(self, history: list[dict[str, Any]] | None) -> tuple[ExperimentParent, ...]:
        """Return the strongest successful node from each active branch."""
        by_branch: dict[str, ExperimentParent] = {}
        for parent in _successful_parents(history):
            current = by_branch.get(parent.branch)
            if (current is None or parent.primary > current.primary or
                    (parent.primary == current.primary and parent.iteration < current.iteration)):
                by_branch[parent.branch] = parent
        ranked = sorted(by_branch.values(), key=lambda node: (-node.primary, node.iteration))
        return tuple(ranked[: self.config.max_active_branches])

    def select(
        self,
        history: list[dict[str, Any]] | None,
        remaining_seconds: float,
    ) -> ParentSelection:
        rows = [item for item in (history or []) if isinstance(item, dict)]
        frontier = self.frontier(rows)
        if not frontier:
            raise RuntimeError("tree has no expandable successful nodes")

        child_rows: dict[str, list[dict[str, Any]]] = {}
        branch_counts: dict[str, int] = {}
        row_by_id: dict[str, dict[str, Any]] = {}
        for item in rows:
            iteration = item.get("iteration")
            if isinstance(iteration, int):
                node_id = node_id_for(iteration)
                row_by_id[node_id] = item
                changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
                branch = branch_name(changes)
                branch_counts[branch] = branch_counts.get(branch, 0) + 1
            parent_id = item.get("parent_id")
            if isinstance(parent_id, str):
                child_rows.setdefault(parent_id, []).append(item)

        total_visits = sum(len(child_rows.get(node.node_id, [])) for node in frontier) + 1
        recent_parent_ids = [
            item.get("parent_id") for item in rows[-3:] if isinstance(item.get("parent_id"), str)
        ]
        time_scale = max(1.0, float(remaining_seconds))
        selections: list[ParentSelection] = []
        for node in frontier:
            children = child_rows.get(node.node_id, [])
            visits = len(children)
            exploration = self.config.exploration_weight * math.sqrt(
                math.log(total_visits + 1.0) / (visits + 1.0)
            )
            novelty = self.config.novelty_weight / max(1, branch_counts.get(node.branch, 1))

            record = row_by_id.get(node.node_id, {})
            metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            runtime = metrics.get("runtime_seconds", 0.0)
            try:
                runtime_seconds = max(0.0, float(runtime))
            except (TypeError, ValueError):
                runtime_seconds = 0.0
            runtime_penalty = self.config.runtime_penalty_weight * runtime_seconds / time_scale

            repetitions = sum(parent_id == node.node_id for parent_id in recent_parent_ids)
            repetition_penalty = self.config.repetition_penalty_weight * repetitions
            failed_children = sum(child.get("status") != "success" for child in children)
            failed_child_penalty = (
                self.config.failed_child_penalty_weight * failed_children / max(1, visits)
            )
            critique = record.get("critique") if isinstance(record.get("critique"), dict) else {}
            rejected_node_penalty = (
                self.config.rejected_node_penalty_weight
                if critique.get("verdict") in {"reject", "failed"} else 0.0
            )
            priority = (
                node.primary + exploration + novelty - runtime_penalty
                - repetition_penalty - failed_child_penalty - rejected_node_penalty
            )
            selections.append(ParentSelection(
                node, priority, node.primary, exploration, novelty, runtime_penalty,
                repetition_penalty, failed_child_penalty, rejected_node_penalty, visits,
            ))
        return max(
            selections,
            key=lambda value: (value.priority, value.parent.primary, -value.parent.iteration),
        )


class ExperimentTree:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []

    def add(self, iteration: int, parent_id: str | None, item: dict[str, Any]) -> None:
        self.nodes.append({
            "node_id": node_id_for(iteration),
            "parent_id": parent_id,
            "branch": branch_name(item.get("changes", {})),
            "status": item["status"], "decision": item["decision"],
            "primary": None if item.get("metrics") is None else item["metrics"]["primary"],
            "config": item["config"],
        })

    def snapshot(self) -> dict[str, Any]:
        return {"nodes": self.nodes}
