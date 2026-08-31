"""Branch-preserving parent selection tests. No model training or API calls."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes
from techjam_agent.tree import TreePolicyConfig, TreeSearchPolicy


def base_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def row(iteration: int, primary: float | None, changes: dict, *, parent_id: str | None,
        status: str = "success", verdict: str = "noise", runtime: float = 10.0) -> dict:
    config = copy.deepcopy(base_config())
    if changes:
        config = apply_changes(config, changes)
    metrics = None if primary is None else {
        "GAUC": primary, "nDCG@5": primary, "primary": primary,
        "runtime_seconds": runtime,
    }
    return {
        "iteration": iteration,
        "parent_id": parent_id,
        "status": status,
        "decision": "KEEP" if iteration == 0 or verdict == "promote" else "REJECT",
        "changes": changes,
        "config": config,
        "metrics": metrics,
        "critique": {"verdict": verdict},
    }


class FrontierTests(unittest.TestCase):
    def test_frontier_keeps_at_most_three_distinct_branches(self) -> None:
        history = [
            row(0, 0.6015, {}, parent_id=None),
            row(1, 0.6020, {"training_objective": "bpr"}, parent_id="baseline",
                verdict="promote"),
            row(2, 0.6018, {"model": "lightgbm"}, parent_id="baseline",
                verdict="promote"),
            row(3, 0.6017, {"learning_rate": 0.002}, parent_id="baseline",
                verdict="promote"),
        ]
        frontier = TreeSearchPolicy(TreePolicyConfig(max_active_branches=3)).frontier(history)
        self.assertEqual(len(frontier), 3)
        self.assertEqual(len({node.branch for node in frontier}), 3)
        self.assertEqual({node.branch for node in frontier}, {
            "ranking_objective", "model", "optimization",
        })

    def test_clear_validation_winner_still_wins_exploitation(self) -> None:
        history = [
            row(0, 0.6015, {}, parent_id=None),
            row(1, 0.6200, {"training_objective": "bpr"}, parent_id="baseline",
                verdict="promote"),
            row(2, 0.6000, {"model": "lightgbm"}, parent_id="baseline", verdict="reject"),
        ]
        selection = TreeSearchPolicy().select(history, remaining_seconds=3600)
        self.assertEqual(selection.parent.node_id, "exp_001")
        self.assertEqual(selection.exploitation, 0.6200)

    def test_rejected_and_failure_producing_branch_is_not_expandable(self) -> None:
        history = [
            row(0, 0.6015, {}, parent_id=None),
            row(1, 0.6016, {"model": "lightgbm"}, parent_id="baseline", verdict="reject"),
            row(2, None, {"continuous_history_stats": True, "model": "lightgbm"},
                parent_id="exp_001", status="error", verdict="failed"),
        ]
        selection = TreeSearchPolicy().select(history, remaining_seconds=3600)
        self.assertEqual(selection.parent.node_id, "baseline")
        self.assertEqual(
            [candidate.node_id for candidate in TreeSearchPolicy().frontier(history)],
            ["baseline"],
        )
        self.assertGreaterEqual(selection.priority, selection.exploitation)

    def test_selection_is_deterministic_and_json_safe(self) -> None:
        history = [row(0, 0.6015, {}, parent_id=None)]
        policy = TreeSearchPolicy()
        first = policy.select(history, remaining_seconds=100)
        second = policy.select(history, remaining_seconds=100)
        self.assertEqual(first, second)
        json.dumps(first.as_dict())


if __name__ == "__main__":
    unittest.main()
