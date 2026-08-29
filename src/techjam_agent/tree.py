from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def branch_name(changes: dict[str, Any]) -> str:
    if not changes:
        return "baseline"
    if "training_objective" in changes:
        return "ranking_objective"
    if "model" in changes:
        return "model"
    if any("rate" in key or "stats" in key for key in changes):
        return "features"
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
        if not isinstance(item, dict) or item.get("status") != "success":
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
