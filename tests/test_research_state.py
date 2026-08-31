from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes
from techjam_agent.controller import Controller
from techjam_agent.proposals import DeterministicResearcher
from techjam_agent.research_state import load_prior_history, load_research_state


def baseline_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text())


def project_config() -> dict:
    return json.loads((ROOT / "configs" / "project.json").read_text())


class FixedRunner:
    def __init__(self, score: float) -> None:
        self.score = score

    def run(self, config, checkpoint):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"model")
        return {"GAUC": self.score, "nDCG@5": self.score, "primary": self.score}

    def finalize(self, config, checkpoint, output):
        return {"GAUC": self.score, "nDCG@5": self.score, "primary": self.score}


class ResearchStateTests(unittest.TestCase):
    def test_committed_incumbent_is_valid(self) -> None:
        state = load_research_state(ROOT / "configs" / "research_state.json")
        self.assertEqual(state["incumbent"]["config"]["model"], "fm_ensemble")
        self.assertAlmostEqual(
            state["incumbent"]["validation_metrics"]["primary"], 0.6042155720287861
        )

    def test_prior_history_is_validation_only_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            run = logs / "run_old"
            run.mkdir()
            config = baseline_config()
            rows = [
                {"iteration": 0, "status": "success", "decision": "KEEP",
                 "changes": {}, "config": config,
                 "metrics": {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55,
                             "test_primary": 0.99}},
                {"iteration": 1, "status": "success", "decision": "KEEP",
                 "changes": {}, "config": config,
                 "metrics": {"GAUC": 0.61, "nDCG@5": 0.51, "primary": 0.56,
                             "runtime_seconds": 12.5, "best_epoch": 7}},
            ]
            (run / "experiment_history.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            history = load_prior_history(logs)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["metrics"]["primary"], 0.56)
        self.assertNotIn("test_primary", history[0]["metrics"])
        self.assertEqual(history[0]["metrics"]["runtime_seconds"], 12.5)
        self.assertEqual(history[0]["metrics"]["best_epoch"], 7.0)
        self.assertTrue(history[0]["evidence_id"].startswith("prior_run_old_"))

    def test_prior_experiment_is_not_repeated(self) -> None:
        baseline = baseline_config()
        prior = [{
            "evidence_id": "old_bpr", "iteration": 1, "historical": True,
            "status": "success", "decision": "KEEP",
            "changes": {"training_objective": "bpr"},
            "config": apply_changes(baseline, {"training_objective": "bpr"}),
            "metrics": {"GAUC": 0.6, "nDCG@5": 0.6, "primary": 0.6},
            "critique": {"verdict": "noise"},
        }]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                FixedRunner(0.6), DeterministicResearcher(), baseline, project_config(),
                base / "run", base / "artifacts", base / "submissions",
                prior_history=prior,
            )
            with patch("sys.stdout", new=io.StringIO()):
                controller.run(max_iterations=2, final_evaluation=False)
        from techjam_agent.config import experiment_key
        self.assertNotEqual(
            experiment_key(controller.history[1]["config"]),
            experiment_key(prior[0]["config"]),
        )

    def test_weaker_run_cannot_overwrite_shared_incumbent(self) -> None:
        incumbent = {
            "config": baseline_config(),
            "validation_metrics": {"GAUC": 0.7, "nDCG@5": 0.7, "primary": 0.7},
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                FixedRunner(0.6), DeterministicResearcher(), baseline_config(), project_config(),
                base / "run", base / "artifacts", base / "submissions",
                shared_incumbent=incumbent,
            )
            with patch("sys.stdout", new=io.StringIO()):
                summary = controller.run(max_iterations=1, final_evaluation=False)
            self.assertFalse((base / "artifacts" / "best_config.json").exists())
            self.assertTrue((base / "run" / "best" / "config.json").exists())
        self.assertEqual(summary["best_primary"], 0.6)
        self.assertEqual(summary["shared_best_primary"], 0.7)


if __name__ == "__main__":
    unittest.main()
