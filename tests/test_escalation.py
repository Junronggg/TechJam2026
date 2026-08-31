from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from techjam_agent.config import apply_changes
from techjam_agent.controller import Controller
from techjam_agent.evidence_escalator import ConfirmationAction, EvidenceEscalator
from techjam_agent.proposals import Proposal


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def load_project() -> dict:
    return json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))


def discovery(delta: float, novelty: float = 0.5) -> tuple[dict, dict]:
    reference = load_config()
    changes = {"model": "deepfm"}
    candidate = apply_changes(reference, changes)
    item = {
        "iteration": 1,
        "status": "success",
        "decision": "KEEP" if delta > 0 else "REJECT",
        "changes": changes,
        "config": candidate,
        "delta_from_parent": delta,
        "diagnostics": {},
        "candidate_selection": {
            "selected_family": "novel_model",
            "retrieved_pattern": None,
            "ranked_candidates": [{"changes": changes, "novelty": novelty}],
        },
    }
    return item, reference


class EvidenceEscalatorTests(unittest.TestCase):
    def test_negative_discovery_is_rejected_without_confirmation(self) -> None:
        item, reference = discovery(-0.0001)
        decision = EvidenceEscalator().plan_discovery(item, reference)
        self.assertEqual(decision.scientific_status, "REJECTED")
        self.assertIsNone(decision.next_action)

    def test_tiny_redundant_gain_does_not_consume_confirmation_budget(self) -> None:
        item, reference = discovery(0.0001, novelty=0.2)
        decision = EvidenceEscalator().plan_discovery(item, reference)
        self.assertEqual(decision.scientific_status, "INSUFFICIENT")
        self.assertIsNone(decision.next_action)

    def test_promising_discovery_schedules_rolling_before_seeds(self) -> None:
        item, reference = discovery(0.0003)
        decision = EvidenceEscalator().plan_discovery(item, reference)
        self.assertEqual(decision.next_action.kind, "rolling")
        passed = EvidenceEscalator().evaluate(decision.next_action, {
            "mean_delta": 0.00025, "wins": 3, "folds": 3,
        })
        self.assertEqual(passed.scientific_status, "PROMISING_NOT_CONFIRMED")
        self.assertEqual(passed.next_action.kind, "paired_seeds")

    def test_failed_rolling_stops_before_paired_seeds(self) -> None:
        item, reference = discovery(0.0003)
        action = EvidenceEscalator().plan_discovery(item, reference).next_action
        decision = EvidenceEscalator().evaluate(action, {
            "mean_delta": -0.0001, "wins": 1, "folds": 3,
        })
        self.assertEqual(decision.scientific_status, "REJECTED")
        self.assertIsNone(decision.next_action)

    def test_positive_seed_mean_with_crossing_interval_remains_uncertain(self) -> None:
        item, reference = discovery(0.0003)
        rolling = EvidenceEscalator().plan_discovery(item, reference).next_action
        paired = EvidenceEscalator().evaluate(rolling, {
            "mean_delta": 0.0002, "wins": 3, "folds": 3,
        }).next_action
        decision = EvidenceEscalator().evaluate(paired, {
            "paired_mean_delta": 0.0003,
            "wins": 3,
            "seeds": 4,
            "approx_95_interval": [-0.0001, 0.0007],
        })
        self.assertEqual(decision.scientific_status, "UNCERTAIN")
        self.assertEqual(decision.competition_status, "ELIGIBLE")

    def test_scoped_final_artifact_status_prevents_duplicate_confirmation(self) -> None:
        item, reference = discovery(0.0003)
        item["candidate_selection"]["retrieved_pattern"] = {
            "scientific_verdict": "VALIDATED",
            "competition_status": "ELIGIBLE",
        }
        decision = EvidenceEscalator().plan_discovery(item, reference)
        self.assertEqual(decision.scientific_status, "VALIDATED")
        self.assertIsNone(decision.next_action)

    def test_existing_artifact_preserves_research_only_competition_status(self) -> None:
        item, reference = discovery(0.0003)
        item["candidate_selection"]["retrieved_pattern"] = {
            "scientific_verdict": "UNCERTAIN",
            "competition_status": "RESEARCH_ONLY",
        }
        decision = EvidenceEscalator().plan_discovery(item, reference)
        self.assertEqual(decision.scientific_status, "UNCERTAIN")
        self.assertEqual(decision.competition_status, "RESEARCH_ONLY")
        self.assertIsNone(decision.next_action)


class FakeConfirmationRunner:
    def __init__(self) -> None:
        self.train_calls = 0
        self.confirm_calls: list[str] = []

    def run(self, config, checkpoint):
        self.train_calls += 1
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        score = 0.6004 if config["model"] == "deepfm" else 0.6000
        return {"GAUC": score, "nDCG@5": score, "primary": score}

    def confirm(self, action: ConfirmationAction, output_dir: Path):
        self.confirm_calls.append(action.kind)
        if action.kind == "rolling":
            return {
                "test_labels_used": False,
                "mean_delta": 0.0003,
                "wins": 3,
                "folds": 3,
                "training_runs": 6,
            }
        return {
            "test_labels_used": False,
            "paired_mean_delta": 0.00035,
            "wins": 4,
            "seeds": 4,
            "approx_95_interval": [0.0001, 0.0006],
            "training_runs": 8,
        }

    def finalize(self, config, checkpoint, output):
        raise AssertionError("test finalization is disabled in this test")


class OneNovelModelResearcher:
    def __init__(self) -> None:
        changes = {"model": "deepfm"}
        self.last_selection = {
            "selected_family": "novel_model",
            "selected_score": 0.5,
            "retrieved_pattern": None,
            "ranked_candidates": [{"changes": changes, "novelty": 0.9}],
        }

    def propose(self, best, history):
        return Proposal(
            "Test a novel model.",
            "The model may add a different interaction mechanism.",
            {"model": "deepfm"},
            "fake",
        )


class ControllerEscalationTests(unittest.TestCase):
    def test_controller_runs_rolling_then_seeds_without_human_intervention(self) -> None:
        runner = FakeConfirmationRunner()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                runner,
                OneNovelModelResearcher(),
                load_config(),
                load_project(),
                base / "logs",
                base / "artifacts",
                base / "submissions",
            )
            summary = controller.run(
                max_iterations=4,
                finalize_test=False,
                auto_confirm=True,
            )
            candidates = json.loads(
                Path(summary["submission_candidates"]).read_text(encoding="utf-8")
            )
        self.assertEqual(runner.confirm_calls, ["rolling", "paired_seeds"])
        self.assertEqual(summary["total_experiments"], 4)
        self.assertEqual(summary["discovery_actions"], 1)
        self.assertEqual(summary["confirmation_actions"], 2)
        self.assertEqual(summary["confirmation_training_runs"], 14)
        self.assertEqual(summary["manual_interventions"], 0)
        target = controller.history[1]
        self.assertEqual(target["diagnostics"]["scientific_status"], "VALIDATED")
        self.assertEqual(target["diagnostics"]["confirmation_status"], "complete")
        by_iteration = {row["iteration"]: row for row in candidates}
        self.assertEqual(by_iteration[1]["scientific_status"], "VALIDATED")
        self.assertEqual(by_iteration[1]["submission_status"], "ALLOW")


if __name__ == "__main__":
    unittest.main()
