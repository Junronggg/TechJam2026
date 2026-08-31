"""Explicit experiment-parent plumbing (P2.6 phase A). Fake runner/researcher only."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import ALLOWED_VALUES, apply_changes
from techjam_agent.controller import Controller
from techjam_agent.memory import build_memory_summary
from techjam_agent.proposals import (
    Proposal,
    build_planner_prompt,
    compact_history_for_planner,
    empty_token_usage,
)
from techjam_agent.tree import ExperimentParent, branch_name, select_parent


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def load_project() -> dict:
    return json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))


def bpr_config() -> dict:
    return apply_changes(load_config(), {"training_objective": "bpr"})


class SweepRunner:
    """Returns queued validation Primary values. Never trains."""

    def __init__(self, primaries) -> None:
        self.primaries = list(primaries)
        self.configs: list[dict] = []

    @property
    def calls(self) -> int:
        return len(self.configs)

    def run(self, config, checkpoint):
        self.configs.append(copy.deepcopy(config))
        value = self.primaries.pop(0) if self.primaries else 0.6015
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"GAUC": value, "nDCG@5": value, "primary": value}

    def finalize(self, config, checkpoint, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
        return {"GAUC": 0.62, "nDCG@5": 0.58, "primary": 0.5953}


class SweepResearcher:
    """Walks the allowed learning-rate grid so proposals never repeat."""

    def __init__(self) -> None:
        self.grid = [{"learning_rate": value} for value in ALLOWED_VALUES["learning_rate"]]
        self.grid += [{"epochs": value} for value in ALLOWED_VALUES["epochs"]]
        self.index = 0
        self.seen_parents: list[dict] = []

    def propose(self, best, history):
        self.seen_parents.append(copy.deepcopy(best))
        while self.index < len(self.grid):
            changes = self.grid[self.index]
            self.index += 1
            candidate = apply_changes(best, changes)
            if candidate == best:
                continue
            return Proposal(f"Sweep {changes}.", "Controlled single-field sweep.",
                            changes, "fake", empty_token_usage())
        raise StopIteration("fake sweep exhausted")


class BprThenSweepResearcher(SweepResearcher):
    def propose(self, best, history):
        self.seen_parents.append(copy.deepcopy(best))
        if best["training_objective"] == "bce":
            return Proposal("Replace BCE with pairwise BPR.",
                            "Ranking metrics reward within-user ordering.",
                            {"training_objective": "bpr"}, "fake", empty_token_usage())
        return super().propose(best, history)


def run_controller(runner, researcher, max_iterations, controller_cls=Controller):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        controller = controller_cls(runner, researcher, load_config(), load_project(),
                                    base / "logs", base / "artifacts", base / "submissions")
        with patch("sys.stdout", new=io.StringIO()):
            summary = controller.run(max_iterations)
        tree = json.loads((base / "logs" / "tree_snapshot.json").read_text(encoding="utf-8"))
        return controller, summary, tree


class SelectParentTests(unittest.TestCase):
    def record(self, iteration, primary, changes=None, status="success", config=None):
        metrics = None if primary is None else {"GAUC": primary, "nDCG@5": primary,
                                                "primary": primary}
        return {"iteration": iteration, "status": status, "metrics": metrics,
                "changes": {} if changes is None else changes,
                "config": config or load_config()}

    def test_returns_global_best_successful_node(self) -> None:
        history = [
            self.record(0, 0.6015),
            self.record(1, 0.6300, {"training_objective": "bpr"}, config=bpr_config()),
            self.record(2, 0.6100, {"learning_rate": 0.002}),
        ]
        parent = select_parent(history)
        self.assertEqual(parent.node_id, "exp_001")
        self.assertEqual(parent.iteration, 1)
        self.assertEqual(parent.primary, 0.6300)
        self.assertEqual(parent.branch, "ranking_objective")
        self.assertEqual(parent.config, bpr_config())

    def test_baseline_is_the_parent_before_any_candidate_improves(self) -> None:
        parent = select_parent([self.record(0, 0.6015)])
        self.assertEqual(parent.node_id, "baseline")
        self.assertEqual(parent.iteration, 0)
        self.assertEqual(parent.branch, "baseline")
        self.assertEqual(parent.config, load_config())

    def test_ties_keep_the_earliest_node(self) -> None:
        history = [self.record(0, 0.6015),
                   self.record(1, 0.6015, {"learning_rate": 0.002})]
        self.assertEqual(select_parent(history).node_id, "baseline")

    def test_failed_and_non_finite_nodes_are_never_parents(self) -> None:
        history = [
            self.record(0, 0.6015),
            self.record(1, None, {"learning_rate": 0.002}, status="error"),
            {"iteration": 2, "status": "success", "changes": {},
             "metrics": {"primary": float("nan")}, "config": load_config()},
        ]
        self.assertEqual(select_parent(history).iteration, 0)

    def test_empty_or_malformed_history_returns_none(self) -> None:
        self.assertIsNone(select_parent([]))
        self.assertIsNone(select_parent(None))
        self.assertIsNone(select_parent(["nope", {"status": "success"}]))

    def test_selection_is_deterministic(self) -> None:
        history = [self.record(0, 0.6015),
                   self.record(1, 0.6300, {"model": "lightgbm"})]
        first, second = select_parent(history), select_parent(history)
        self.assertEqual(first, second)


class LineageRecordingTests(unittest.TestCase):
    def test_baseline_has_no_parent_and_candidate_points_at_baseline(self) -> None:
        runner = SweepRunner([0.601470, 0.603396])
        controller, _, tree = run_controller(runner, BprThenSweepResearcher(), 2)
        baseline, candidate = controller.history

        self.assertIsNone(baseline["parent_id"])
        self.assertIsNone(baseline["parent_primary"])
        self.assertIsNone(baseline["global_best_primary_before"])
        self.assertIsNone(baseline["delta_from_parent"])
        self.assertIsNone(baseline["delta_from_best"])

        self.assertEqual(candidate["parent_id"], "baseline")
        self.assertEqual(candidate["parent_primary"], 0.601470)
        self.assertEqual(candidate["parent_selection"]["parent_id"], "baseline")
        self.assertIn("exploration_bonus", candidate["parent_selection"])
        self.assertIn("runtime_penalty", candidate["parent_selection"])
        self.assertEqual(candidate["global_best_primary_before"], 0.601470)
        self.assertAlmostEqual(candidate["delta_from_parent"], 0.001926, places=6)
        self.assertAlmostEqual(candidate["delta_from_best"], 0.001926, places=6)

        self.assertEqual([node["parent_id"] for node in tree["nodes"]], [None, "baseline"])
        self.assertEqual([node["node_id"] for node in tree["nodes"]], ["baseline", "exp_001"])

    def test_parent_score_remains_an_alias_of_parent_primary(self) -> None:
        runner = SweepRunner([0.6015, 0.6100])
        controller, _, _ = run_controller(runner, SweepResearcher(), 2)
        for row in controller.history:
            self.assertEqual(row["parent_score"], row["parent_primary"])

    def test_researcher_receives_the_selected_parent_config(self) -> None:
        researcher = BprThenSweepResearcher()
        runner = SweepRunner([0.6015, 0.6300, 0.6400])
        controller, _, _ = run_controller(runner, researcher, 3)
        self.assertEqual(researcher.seen_parents[0], load_config())
        self.assertEqual(researcher.seen_parents[1], bpr_config())
        self.assertEqual(controller.history[2]["parent_id"], "exp_001")

    def test_lineage_chain_follows_improving_nodes(self) -> None:
        runner = SweepRunner([0.6015, 0.6100, 0.6200, 0.6300])
        controller, _, tree = run_controller(runner, SweepResearcher(), 4)
        self.assertEqual([node["parent_id"] for node in tree["nodes"]],
                         [None, "baseline", "exp_001", "exp_002"])
        self.assertEqual([row["parent_id"] for row in controller.history],
                         [None, "baseline", "exp_001", "exp_002"])


class CriticAndKeepSeparationTests(unittest.TestCase):
    def test_critic_delta_uses_the_actual_parent(self) -> None:
        runner = SweepRunner([0.601470, 0.603396])
        controller, _, _ = run_controller(runner, BprThenSweepResearcher(), 2)
        critique = controller.history[1]["critique"]
        self.assertAlmostEqual(critique["delta"], 0.001926, places=6)
        self.assertEqual(critique["verdict"], "noise")
        self.assertFalse(critique["meaningful_improvement"])
        self.assertEqual(controller.history[1]["decision"], "KEEP")

    def test_keep_still_protects_the_global_best(self) -> None:
        runner = SweepRunner([0.6015, 0.6300, 0.6100])
        controller, summary, _ = run_controller(runner, SweepResearcher(), 3)
        self.assertEqual([row["decision"] for row in controller.history],
                         ["KEEP", "KEEP", "REJECT"])
        self.assertEqual(summary["best_primary"], 0.6300)
        self.assertEqual(controller.best_iteration, 1)

    def test_convergence_compares_against_global_best_progress(self) -> None:
        runner = SweepRunner([0.6015, 0.6016, 0.6016, 0.6016, 0.7000])
        controller, summary, _ = run_controller(runner, SweepResearcher(), 10)
        self.assertEqual(summary["stop_reason"], "converged")
        self.assertEqual(summary["convergence_streak"], 3)
        self.assertEqual(runner.calls, 8)
        self.assertEqual(summary["best_primary"], 0.7000)


class WeakerParentTests(unittest.TestCase):
    """Phase A never picks a weaker parent; a forced override must stay safe."""

    def test_forced_weaker_parent_cannot_overwrite_global_best(self) -> None:
        class WeakParentController(Controller):
            def _select_parent(self):
                for row in self.history:
                    if row["iteration"] == 0:
                        return ExperimentParent("baseline", 0, row["config"],
                                                row["metrics"]["primary"],
                                                branch_name(row["changes"]))
                return None

        runner = SweepRunner([0.6015, 0.6300, 0.6100])
        controller, summary, tree = run_controller(runner, SweepResearcher(), 3,
                                                   controller_cls=WeakParentController)
        weak = controller.history[2]
        self.assertEqual(weak["parent_id"], "baseline")
        self.assertEqual(weak["parent_primary"], 0.6015)
        self.assertEqual(weak["global_best_primary_before"], 0.6300)
        self.assertAlmostEqual(weak["delta_from_parent"], 0.0085, places=6)
        self.assertAlmostEqual(weak["delta_from_best"], -0.02, places=6)
        self.assertEqual(weak["decision"], "REJECT")

        self.assertEqual(summary["best_primary"], 0.6300)
        self.assertEqual(controller.best_iteration, 1)
        self.assertEqual(controller.best_config, runner.configs[1])
        self.assertEqual(tree["nodes"][2]["parent_id"], "baseline")

    def test_weaker_parent_child_wins_only_by_beating_global_best(self) -> None:
        class WeakParentController(Controller):
            def _select_parent(self):
                for row in self.history:
                    if row["iteration"] == 0:
                        return ExperimentParent("baseline", 0, row["config"],
                                                row["metrics"]["primary"],
                                                branch_name(row["changes"]))
                return None

        runner = SweepRunner([0.6015, 0.6300, 0.6500])
        controller, summary, _ = run_controller(runner, SweepResearcher(), 3,
                                                controller_cls=WeakParentController)
        winner = controller.history[2]
        self.assertEqual(winner["parent_id"], "baseline")
        self.assertEqual(winner["decision"], "KEEP")
        self.assertEqual(summary["best_primary"], 0.6500)
        self.assertEqual(controller.best_iteration, 2)


class PlannerPromptTests(unittest.TestCase):
    def history(self):
        return [{
            "iteration": 0, "status": "success", "decision": "KEEP", "changes": {},
            "hypothesis": "Reproduce the official FM baseline.",
            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
            "config": load_config(),
            "critique": {"verdict": "noise", "observation": "Validation Primary=0.601500"},
        }]

    def test_prompt_distinguishes_global_best_and_expansion_parent(self) -> None:
        prompt = build_planner_prompt(load_config(), self.history())
        self.assertIn("global_best", prompt)
        self.assertIn("expansion_parent", prompt)
        self.assertNotIn("current_best", prompt)
        self.assertEqual(prompt["global_best"]["config"], load_config())
        parent = prompt["expansion_parent"]
        self.assertEqual(parent["node_id"], "baseline")
        self.assertEqual(parent["validation_primary"], 0.6015)
        self.assertTrue(parent["same_as_global_best"])
        self.assertNotIn("config", parent)
        self.assertIn("expansion_parent", prompt["change_rule"])

    def test_divergent_parent_is_representable_in_the_schema(self) -> None:
        history = self.history()
        history.append({
            "iteration": 1, "status": "success", "decision": "REJECT",
            "changes": {"model": "lightgbm"},
            "metrics": {"GAUC": 0.60, "nDCG@5": 0.60, "primary": 0.60},
            "config": apply_changes(load_config(), {"model": "lightgbm"}),
            "critique": {"verdict": "reject"},
        })
        prompt = build_planner_prompt(load_config(), history, expansion_parent={
            "node_id": "exp_001", "iteration": 1, "branch": "model",
            "validation_primary": 0.60, "same_as_global_best": False,
            "config": apply_changes(load_config(), {"model": "lightgbm"}),
        })
        self.assertEqual(prompt["global_best"]["config"]["model"], "fm")
        self.assertEqual(prompt["expansion_parent"]["node_id"], "exp_001")
        self.assertFalse(prompt["expansion_parent"]["same_as_global_best"])
        self.assertEqual(prompt["expansion_parent"]["config"]["model"], "lightgbm")

    def test_empty_history_still_builds_a_prompt(self) -> None:
        prompt = build_planner_prompt(load_config(), [])
        self.assertIsNone(prompt["expansion_parent"]["node_id"])
        self.assertTrue(prompt["expansion_parent"]["same_as_global_best"])


class ProposalOwnershipTests(unittest.TestCase):
    def test_llm_response_with_parent_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as raised:
            Proposal.parse({
                "hypothesis": "Try BPR.",
                "reason": "Ranking loss matches ranking metrics.",
                "changes": {"training_objective": "bpr"},
                "parent_id": "exp_003",
            }, "llm")
        self.assertIn("parent_id", str(raised.exception))

    def test_proposal_has_no_parent_id_field(self) -> None:
        proposal = Proposal("h", "r", {"training_objective": "bpr"}, "llm")
        self.assertNotIn("parent_id", proposal.as_dict())
        self.assertFalse(hasattr(proposal, "parent_id"))


class BackwardCompatibilityTests(unittest.TestCase):
    def old_row(self):
        return {
            "iteration": 0,
            "hypothesis": "Reproduce the official FM baseline.",
            "changes": {},
            "decision": "KEEP",
            "status": "success",
            "parent_score": None,
            "delta_from_best": None,
            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
            "config": load_config(),
            "critique": {"observation": "Validation Primary=0.601500", "confidence": "high"},
        }

    def test_history_with_only_parent_score_remains_readable(self) -> None:
        history = [self.old_row()]
        self.assertEqual(select_parent(history).node_id, "baseline")
        self.assertEqual(build_memory_summary(history)["baseline_reference"]["validation_primary"],
                         0.6015)
        self.assertEqual(compact_history_for_planner(history)[0]["decision"], "KEEP")
        prompt = build_planner_prompt(load_config(), history)
        self.assertEqual(prompt["expansion_parent"]["node_id"], "baseline")

    def test_new_records_keep_every_pre_existing_key(self) -> None:
        runner = SweepRunner([0.6015, 0.6100])
        controller, _, _ = run_controller(runner, SweepResearcher(), 2)
        for row in controller.history:
            for key in ("iteration", "timestamp", "hypothesis", "reason", "changes", "source",
                        "token_usage", "parent_score", "config", "manual_intervention",
                        "status", "metrics", "delta_from_best", "decision", "error", "critique"):
                self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
