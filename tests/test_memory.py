"""Compact research memory and duplicate prevention. Fake history/runner only."""

from __future__ import annotations

import copy
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

from techjam_agent.config import apply_changes, experiment_key
from techjam_agent.controller import Controller
from techjam_agent.memory import (
    LESSON_LIMIT,
    PLANNER_RECENT_HISTORY,
    SIGNATURE_LIMIT,
    build_memory_summary,
    collect_tried_keys,
    is_duplicate_config,
)
from techjam_agent.proposals import (
    DeterministicResearcher,
    Proposal,
    build_planner_prompt,
)


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def load_project() -> dict:
    return json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))


def bpr_config() -> dict:
    return apply_changes(load_config(), {"training_objective": "bpr"})


def record(
    iteration: int,
    *,
    config: dict | None = None,
    changes: dict | None = None,
    hypothesis: str = "hypothesis",
    verdict: str | None = "noise",
    primary: float | None = 0.6015,
    decision: str = "REJECT",
    status: str = "success",
    extra_metrics: dict | None = None,
    error: dict | None = None,
    **extra,
) -> dict:
    metrics = None
    if primary is not None or extra_metrics:
        metrics = {}
        if primary is not None:
            metrics.update({"GAUC": primary, "nDCG@5": primary, "primary": primary})
        if extra_metrics:
            metrics.update(extra_metrics)
    critique = None
    if verdict is not None:
        critique = {
            "observation": f"Validation Primary={primary}",
            "interpretation": "structured critic",
            "confidence": "low",
            "verdict": verdict,
            "delta": 0.0 if primary is not None else None,
            "meaningful_improvement": verdict == "promote",
            "next_test": f"next for {verdict}",
            "reasons": [verdict],
        }
    item = {
        "iteration": iteration,
        "hypothesis": hypothesis,
        "changes": {} if changes is None else changes,
        "decision": decision,
        "status": status,
        "metrics": metrics,
        "config": copy.deepcopy(load_config() if config is None else config),
        "critique": critique,
        "error": error,
    }
    item.update(extra)
    return item


class CountingRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, config, checkpoint):
        self.calls += 1
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015}

    def finalize(self, config, checkpoint, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
        return {"GAUC": 0.62, "nDCG@5": 0.58, "primary": 0.5953}


class AlwaysBprResearcher(DeterministicResearcher):
    def propose(self, best, history):
        return Proposal(
            "Rewrite the BPR hypothesis in new words.",
            "Wording is not part of the scientific configuration.",
            {"training_objective": "bpr"},
            "deterministic",
        )


def example_history() -> list[dict]:
    baseline = record(
        0,
        config=load_config(),
        changes={},
        hypothesis="Reproduce the official FM baseline.",
        verdict="noise",
        primary=0.601470,
        decision="KEEP",
    )
    bpr = record(
        1,
        config=bpr_config(),
        changes={"training_objective": "bpr"},
        hypothesis="Replace BCE with pairwise BPR.",
        verdict="noise",
        primary=0.603396,
        decision="KEEP",
    )
    bpr["critique"]["delta"] = 0.001926
    bpr["critique"]["meaningful_improvement"] = False
    failed = record(
        2,
        config=apply_changes(bpr_config(), {"learning_rate": 0.005}),
        changes={"learning_rate": 0.005},
        hypothesis="Raise learning rate.",
        verdict="failed",
        primary=None,
        decision="REJECT",
        status="error",
        error={"type": "TimeoutError", "message": "experiment exceeded 900s timeout"},
        stdout="TRAINING LOG SECRET",
        stderr="traceback body",
    )
    return [baseline, bpr, failed]


class CategoryTests(unittest.TestCase):
    def test_verdicts_enter_the_correct_category(self) -> None:
        history = [
            record(0, verdict="promote", primary=0.6100, decision="KEEP",
                   changes={"training_objective": "bpr"}, config=bpr_config()),
            record(1, verdict="noise", primary=0.6017),
            record(2, verdict="reject", primary=0.5900),
            record(3, verdict="failed", primary=None, status="error",
                   error={"type": "TimeoutError"}),
        ]
        summary = build_memory_summary(history)
        self.assertEqual([row["verdict"] for row in summary["promising"]], ["promote"])
        self.assertEqual([row["verdict"] for row in summary["uncertain"]], ["noise"])
        self.assertEqual([row["verdict"] for row in summary["negative"]], ["reject"])
        self.assertEqual([row["verdict"] for row in summary["failed"]], ["failed"])
        self.assertEqual(summary["counts"], {
            "total": 4,
            "baseline_reference": 0,
            "candidate_verdicts": 4,
            "unclassified_candidates": 0,
            "promising": 1,
            "uncertain": 1,
            "negative": 1,
            "failed": 1,
        })
        self.assertEqual(summary["best_observed"]["validation_primary"], 0.6100)
        self.assertEqual(summary["best_observed"]["training_objective"], "bpr")
        self.assertEqual(summary["failed"][0]["error_type"], "TimeoutError")

    def test_example_baseline_bpr_and_failure(self) -> None:
        summary = build_memory_summary(example_history())
        self.assertEqual(summary["baseline_reference"], {
            "iteration": 0,
            "validation_primary": 0.601470,
            "model": "fm",
            "training_objective": "bce",
        })
        self.assertEqual(summary["counts"]["baseline_reference"], 1)
        self.assertEqual(summary["counts"]["candidate_verdicts"], 2)
        self.assertEqual(summary["counts"]["uncertain"], 1)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["best_observed"]["validation_primary"], 0.603396)
        self.assertEqual(summary["best_observed"]["training_objective"], "bpr")
        self.assertEqual(summary["uncertain"][0]["primary"], 0.603396)
        self.assertNotIn(0, [item["iteration"] for item in summary["uncertain"]])
        self.assertEqual(summary["failed"][0]["error_type"], "TimeoutError")


class CompatibilityTests(unittest.TestCase):
    def test_old_history_records_still_work(self) -> None:
        old = {
            "iteration": 0,
            "hypothesis": "Reproduce the official FM baseline.",
            "changes": {},
            "decision": "KEEP",
            "status": "success",
            "metrics": {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015},
            "config": load_config(),
            "critique": {
                "observation": "Validation Primary=0.601500",
                "interpretation": "This establishes the validation baseline.",
                "confidence": "high",
                "next_test": "Repeat promising results across seeds.",
            },
        }
        broken = {"iteration": 1, "metrics": {"primary": float("nan")}, "status": "success"}
        summary = build_memory_summary([old, broken, "ignore", None])
        self.assertEqual(summary["best_observed"]["validation_primary"], 0.6015)
        self.assertEqual(summary["baseline_reference"]["validation_primary"], 0.6015)
        self.assertEqual(summary["counts"]["total"], 2)
        self.assertEqual(summary["counts"]["baseline_reference"], 1)
        self.assertEqual(summary["counts"]["unclassified_candidates"], 1)
        self.assertEqual(summary["counts"]["failed"], 0)
        self.assertEqual(summary["promising"], [])
        encoded = json.dumps(summary)
        self.assertIn("0.6015", encoded)

    def test_first_reference_without_iteration_field_is_compatible(self) -> None:
        old_baseline = record(0, decision="KEEP", changes={}, primary=0.6015)
        old_baseline.pop("iteration")
        summary = build_memory_summary([old_baseline])
        self.assertIsNone(summary["baseline_reference"]["iteration"])
        self.assertEqual(summary["baseline_reference"]["validation_primary"], 0.6015)
        self.assertEqual(summary["uncertain"], [])

    def test_non_finite_metrics_are_ignored(self) -> None:
        history = [
            record(0, verdict="noise", extra_metrics={"primary": float("inf")}),
        ]
        history[0]["metrics"] = {"GAUC": math.nan, "nDCG@5": math.inf, "primary": math.nan}
        summary = build_memory_summary(history)
        self.assertIsNone(summary["baseline_reference"]["validation_primary"])
        self.assertEqual(summary["uncertain"], [])
        self.assertIsNone(summary["best_observed"])


class BoundAndSafetyTests(unittest.TestCase):
    def test_summary_is_bounded_after_fifty_experiments(self) -> None:
        history = [
            record(0, decision="KEEP", changes={}, primary=0.6015),
        ]
        for index in range(1, 50):
            cfg = dict(load_config())
            cfg["marker"] = index
            history.append(record(
                index, config=cfg, verdict="noise", primary=0.6015 + index * 1e-6,
                changes={"learning_rate": 0.001},
            ))
        summary = build_memory_summary(history)
        self.assertEqual(summary["counts"]["total"], 50)
        self.assertEqual(summary["counts"]["baseline_reference"], 1)
        self.assertEqual(summary["counts"]["candidate_verdicts"], 49)
        self.assertEqual(summary["counts"]["uncertain"], 49)
        self.assertEqual(len(summary["uncertain"]), LESSON_LIMIT)
        self.assertEqual(summary["uncertain"][0]["iteration"], 45)
        self.assertEqual(len(summary["tried_signatures"]), SIGNATURE_LIMIT)
        self.assertEqual(len(collect_tried_keys(history)), 50)

    def test_summary_is_deterministic_and_json_serializable(self) -> None:
        history = example_history()
        first = build_memory_summary(history)
        second = build_memory_summary(history)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        json.dumps(first)

    def test_raw_test_metrics_stdout_and_secrets_never_enter_summary_or_prompt(self) -> None:
        dirty = record(
            0,
            verdict="noise",
            primary=0.6015,
            decision="KEEP",
            extra_metrics={"test_GAUC": 0.9999},
            stdout="TRAINING LOG SECRET",
            stderr="Traceback (most recent call last)",
            api_key="sk-secret-do-not-leak",
        )
        dirty["final_test_metrics"] = {"primary": 0.5953}
        dirty["traceback"] = "full traceback body"
        dirty["metrics"]["test"] = {"primary": 0.5953}
        history = [dirty]
        summary = build_memory_summary(history)
        prompt = build_planner_prompt(load_config(), history)
        blob = json.dumps({"summary": summary, "prompt": prompt})
        self.assertNotIn("0.9999", blob)
        self.assertNotIn("0.5953", blob)
        self.assertNotIn("test_GAUC", blob)
        self.assertNotIn("TRAINING LOG SECRET", blob)
        self.assertNotIn("sk-secret-do-not-leak", blob)
        self.assertNotIn("full traceback body", blob)
        self.assertNotIn("Traceback (most recent call last)", blob)
        self.assertLessEqual(len(prompt["history"]), PLANNER_RECENT_HISTORY)
        self.assertIn("promising", prompt["memory"])
        self.assertNotIn("tried_keys", prompt["memory"])
        self.assertIn("tried_signatures", prompt["memory"])
        self.assertIn("remaining", prompt)

    def test_full_keys_stay_internal_and_compact_signatures_reach_prompt(self) -> None:
        history = example_history()
        full_keys = collect_tried_keys(history)
        self.assertEqual(len(full_keys), 3)
        self.assertTrue(all(key.startswith("{") for key in full_keys))

        prompt = build_planner_prompt(bpr_config(), history)
        prompt_blob = json.dumps(prompt, sort_keys=True)
        self.assertNotIn("tried_keys", prompt["memory"])
        self.assertNotIn("experiment_key", prompt_blob)
        for key in full_keys:
            self.assertNotIn(key, prompt_blob)
        signatures = prompt["memory"]["tried_signatures"]
        self.assertEqual(len(signatures), 3)
        self.assertEqual(signatures[1]["training_objective"], "bpr")
        self.assertEqual(signatures[1]["changes"], {"training_objective": "bpr"})
        self.assertEqual(len(signatures[1]["key_hash"]), 12)


class DuplicateTests(unittest.TestCase):
    def test_identical_config_with_different_hypothesis_is_duplicate(self) -> None:
        config = load_config()
        history = [record(0, config=config, hypothesis="First write-up.", decision="KEEP")]
        twin = copy.deepcopy(config)
        self.assertTrue(is_duplicate_config(twin, history))
        self.assertEqual(collect_tried_keys(history), [experiment_key(config)])

    def test_different_config_is_not_a_duplicate(self) -> None:
        history = [record(0, config=load_config(), decision="KEEP")]
        self.assertFalse(is_duplicate_config(bpr_config(), history))
        self.assertNotEqual(experiment_key(load_config()), experiment_key(bpr_config()))

    def test_duplicate_proposal_does_not_invoke_the_runner(self) -> None:
        runner = CountingRunner()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                runner, AlwaysBprResearcher(), load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            with patch("sys.stdout", new=io.StringIO()):
                summary = controller.run(max_iterations=3)
        self.assertEqual(runner.calls, 2)
        self.assertEqual(len(controller.history), 2)
        self.assertIn("duplicate", summary["stop_reason"])
        self.assertEqual(controller.history[1]["changes"], {"training_objective": "bpr"})
        self.assertNotEqual(controller.history[0]["hypothesis"], controller.history[1]["hypothesis"])
        self.assertEqual(
            experiment_key(apply_changes(load_config(), {"training_objective": "bpr"})),
            experiment_key(controller.history[1]["config"]),
        )


if __name__ == "__main__":
    unittest.main()
