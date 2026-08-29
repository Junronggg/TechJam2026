from __future__ import annotations

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


class ExperimentTree:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []

    def add(self, iteration: int, parent_iteration: int | None, item: dict[str, Any]) -> None:
        self.nodes.append({
            "node_id": f"exp_{iteration:03d}" if iteration else "baseline",
            "parent_id": None if parent_iteration is None else
                         (f"exp_{parent_iteration:03d}" if parent_iteration else "baseline"),
            "branch": branch_name(item.get("changes", {})),
            "status": item["status"], "decision": item["decision"],
            "primary": None if item.get("metrics") is None else item["metrics"]["primary"],
            "config": item["config"],
        })

    def snapshot(self) -> dict[str, Any]:
        return {"nodes": self.nodes}
