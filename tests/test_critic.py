"""Structured Critic tests. Fake metrics only; no training, dataset, or LLM."""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.controller import Controller
from techjam_agent.critic import CriticResult, review
from techjam_agent.proposals import compact_history_for_planner


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def load_project() -> dict:
    return json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))


def critique(primary: float, parent: float | None, **kwargs) -> dict:
    metrics = {"GAUC": primary, "nDCG@5": primary, "primary": primary}
    metrics.update(kwargs.pop("metrics", {}))
    return review(metrics, parent, 0.002, kwargs.pop("status", "success"),
                  kwargs.pop("error", None),
                  history=kwargs.pop("history", None),
                  changes=kwargs.pop("changes", {"training_objective": "bpr"}))


class FakeRunner:
    def run(self, config, checkpoint):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015,
                "best_epoch": 1, "runtime_seconds": 0.01}

    def finalize(self, config, checkpoint, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
        return {"GAUC": 0.62, "nDCG@5": 0.58, "primary": 0.5953}


class VerdictTests(unittest.TestCase):
    def test_clear_improvement_is_promote(self) -> None:
        result = critique(0.6040, 0.6016)
        self.assertEqual(result["verdict"], "promote")
        self.assertTrue(result["meaningful_improvement"])
        self.assertGreater(result["delta"], 0.002)

    def test_tiny_positive_is_noise(self) -> None:
        result = critique(0.6017, 0.6016)
        self.assertEqual(result["verdict"], "noise")
        self.assertEqual(result["hypothesis_status"], "inconclusive")
        self.assertFalse(result["meaningful_improvement"])
        self.assertAlmostEqual(result["delta"], 0.0001, places=6)

    def test_component_metric_deltas_and_tradeoff_are_explicit(self) -> None:
        result = review(
            {"GAUC": 0.6710, "nDCG@5": 0.5340, "primary": 0.6025},
            0.6015,
            0.002,
            "success",
            parent_metrics={"GAUC": 0.6670, "nDCG@5": 0.5360, "primary": 0.6015},
            changes={"training_objective": "bpr"},
        )
        self.assertAlmostEqual(result["metric_deltas"]["GAUC"], 0.004)
        self.assertAlmostEqual(result["metric_deltas"]["nDCG@5"], -0.002)
        self.assertAlmostEqual(result["metric_deltas"]["primary"], 0.001)
        self.assertIn("trade-off", result["interpretation"])
        self.assertIn("component_metric_tradeoff", result["reasons"])

    def test_tiny_negative_is_noise(self) -> None:
        result = critique(0.6015, 0.6016)
        self.assertEqual(result["verdict"], "noise")
        self.assertFalse(result["meaningful_improvement"])

    def test_clear_regression_is_reject(self) -> None:
        result = critique(0.5980, 0.6016)
        self.assertEqual(result["verdict"], "reject")
        self.assertFalse(result["meaningful_improvement"])

    def test_bpr_single_seed_gain_is_noise(self) -> None:
        result = critique(0.603396, 0.601470)
        self.assertEqual(result["verdict"], "noise")
        self.assertFalse(result["meaningful_improvement"])
        self.assertAlmostEqual(result["delta"], 0.001926, places=6)

    def test_timeout_or_error_is_failed(self) -> None:
        result = review(None, 0.6016, 0.002, "error",
                        {"type": "TimeoutError", "message": "experiment exceeded 900s timeout"})
        self.assertEqual(result["verdict"], "failed")
        self.assertFalse(result["meaningful_improvement"])
        self.assertIsNone(result["delta"])

    def test_missing_metrics_are_failed(self) -> None:
        result = review(None, 0.6016, 0.002, "success")
        self.assertEqual(result["verdict"], "failed")

    def test_nan_and_inf_are_failed(self) -> None:
        nan = review({"primary": float("nan")}, 0.6016, 0.002, "success")
        inf = review({"primary": float("inf")}, 0.6016, 0.002, "success")
        self.assertEqual(nan["verdict"], "failed")
        self.assertEqual(inf["verdict"], "failed")
        self.assertTrue(math.isnan(float("nan")))


class TextTests(unittest.TestCase):
    def test_observation_contains_exact_measured_values(self) -> None:
        result = critique(0.603396, 0.601470)
        self.assertIn("0.603396", result["observation"])
        self.assertIn("0.601470", result["observation"])
        self.assertIn("+0.001926", result["observation"])

    def test_interpretation_does_not_invent_metrics(self) -> None:
        result = critique(0.603396, 0.601470)
        self.assertNotIn("0.9999", result["interpretation"])
        self.assertNotIn("0.5953", result["interpretation"])
        self.assertIn("0.001926", result["interpretation"])
        self.assertIn("epsilon", result["interpretation"])
        promote = review({"primary": 0.6100}, 0.6016, 0.002, "success")
        self.assertIn("single-seed", promote["interpretation"])
        self.assertIn("not a statistical significance test", promote["interpretation"])

    def test_history_affects_reasons_and_next_test(self) -> None:
        history = [
            {
                "iteration": 1,
                "changes": {"learning_rate": 0.002},
                "metrics": {"GAUC": 0.6672, "nDCG@5": 0.5358, "primary": 0.6015},
                "critique": {"verdict": "noise"},
            },
            {
                "iteration": 2,
                "changes": {"learning_rate": 0.0005},
                "metrics": {"GAUC": 0.6670, "nDCG@5": 0.5357, "primary": 0.6014},
                "critique": {"verdict": "noise"},
            },
        ]
        without = critique(0.6017, 0.6016, changes={"learning_rate": 0.005}, history=[])
        with_hist = critique(0.6017, 0.6016, changes={"learning_rate": 0.005}, history=history)
        self.assertIn("repeated_noisy_changes", with_hist["reasons"])
        self.assertIn("used_recent_validation_history", with_hist["reasons"])
        self.assertIn("iteration 1 validation Primary=0.601500", with_hist["observation"])
        self.assertNotEqual(without["next_test"], with_hist["next_test"])
        self.assertIn("distinct hypothesis", with_hist["next_test"])

    def test_test_metrics_never_enter_critic_output(self) -> None:
        history = [{
            "iteration": 0,
            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015,
                        "test_GAUC": 0.9999},
            "final_test_metrics": {"primary": 0.5953},
            "critique": {"verdict": "noise"},
        }]
        result = review(
            {"GAUC": 0.6697, "nDCG@5": 0.5371, "primary": 0.6034, "test_GAUC": 0.8888},
            0.6015, 0.002, "success",
            history=history, changes={"training_objective": "bpr"},
        )
        blob = json.dumps(result)
        self.assertNotIn("0.9999", blob)
        self.assertNotIn("0.5953", blob)
        self.assertNotIn("0.8888", blob)
        self.assertNotIn("test_GAUC", blob)

    def test_old_history_records_remain_compatible(self) -> None:
        old = {
            "iteration": 0,
            "hypothesis": "Reproduce the official FM baseline.",
            "changes": {},
            "decision": "KEEP",
            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
            "critique": {
                "observation": "Validation Primary=0.601500",
                "interpretation": "This establishes the validation baseline.",
                "confidence": "high",
                "next_test": "Repeat promising results across seeds; otherwise test a distinct hypothesis.",
            },
        }
        compact = compact_history_for_planner([old])
        self.assertEqual(compact[0]["critique"]["observation"], "Validation Primary=0.601500")
        result = review({"primary": 0.6017}, 0.6015, 0.002, "success", history=[old])
        self.assertIn(result["verdict"], ("promote", "noise", "reject", "failed"))
        self.assertIn("observation", result)


class ControllerIntegrationTests(unittest.TestCase):
    def test_controller_history_contains_structured_critique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            from techjam_agent.proposals import DeterministicResearcher
            controller = Controller(
                FakeRunner(), DeterministicResearcher(), load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            with patch("sys.stdout", new=io.StringIO()):
                controller.run(max_iterations=1)
            record = json.loads((base / "logs" / "iteration_000.json").read_text(encoding="utf-8"))
            critique = record["critique"]
            for key in ("observation", "interpretation", "confidence", "verdict",
                        "delta", "meaningful_improvement", "next_test", "reasons",
                        "metric_deltas", "hypothesis_status", "evidence_strength",
                        "seed_count"):
                self.assertIn(key, critique)
            self.assertEqual(critique["verdict"], "noise")
            self.assertFalse(critique["meaningful_improvement"])
            self.assertIn("0.601500", critique["observation"])
            planner_prompt_history = compact_history_for_planner(controller.history)
            self.assertEqual(planner_prompt_history[0]["critique"]["verdict"], "noise")


class ShapeTests(unittest.TestCase):
    def test_result_serializes_to_json(self) -> None:
        payload = CriticResult("obs", "interp", "low", "noise", 0.0001, False, "next", ["r"]).as_dict()
        encoded = json.dumps(payload)
        self.assertIn("noise", encoded)
        self.assertEqual(json.loads(encoded)["verdict"], "noise")


if __name__ == "__main__":
    unittest.main()
