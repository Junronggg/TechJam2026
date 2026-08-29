from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes, validate_config
from techjam_agent.controller import Controller, _write_json
from techjam_agent.proposals import DeterministicResearcher
from techjam_agent.history_features import aggregate, aggregate_pair, smoothed_rate_bucket


class FakeRunner:
    def __init__(self):
        self.calls = 0

    def run(self, config, checkpoint):
        self.calls += 1
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        score = 0.60 + (0.01 if self.calls == 2 else 0.0)
        return {"GAUC": score, "nDCG@5": score, "primary": score,
                "best_epoch": 1, "runtime_seconds": 0.01}

    def finalize(self, config, checkpoint, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
        return {"GAUC": 0.62, "nDCG@5": 0.58, "primary": 0.60}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs" / "experiment.json").read_text())
        self.project = json.loads((ROOT / "configs" / "project.json").read_text())

    def test_rejects_unsupported_change(self):
        with self.assertRaises(ValueError):
            apply_changes(self.config, {"dropout": 0.5})

    def test_history_rate_uses_train_and_leaves_current_label_out(self):
        rows = [(0, "u1", "v1", "a", "t", 1.0, 1),
                (0, "u1", "v2", "a", "t", 1.0, 0),
                (0, "u2", "v1", "a", "t", 1.0, 1)]
        stats, global_rate = aggregate(rows, 1)
        full = smoothed_rate_bucket("u1", stats, global_rate, prior=0)
        leave_positive = smoothed_rate_bucket("u1", stats, global_rate,
                                              label_to_leave_out=1, prior=0)
        self.assertEqual(full, 10)
        self.assertEqual(leave_positive, 0)

    def test_researcher_tries_features_before_hyperparameters(self):
        proposal = DeterministicResearcher().propose(self.config, [])
        self.assertEqual(proposal.changes, {"training_objective": "bpr"})

    def test_pair_sampler_keeps_pairs_within_user(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.bpr import build_pair_indices
        users = ["u1", "u1", "u2", "u2", "u3"]
        labels = np.asarray([1, 0, 0, 1, 1])
        positives, negatives = build_pair_indices(users, labels, np.random.default_rng(0))
        self.assertTrue(all(users[p] == users[n] for p, n in zip(positives, negatives)))
        self.assertTrue(all(labels[p] == 1 and labels[n] == 0 for p, n in zip(positives, negatives)))

    def test_researcher_adds_continuous_stats_after_lightgbm(self):
        lgb = apply_changes(self.config, {"model": "lightgbm"})
        proposal = DeterministicResearcher().propose(lgb, [{"config": self.config}])
        self.assertEqual(proposal.changes, {"user_tab_long_view_rate": True})

    def test_user_tab_aggregation_keeps_preferences_separate(self):
        rows = [(0, "u1", "v1", "a", "sports", 1.0, 1),
                (0, "u1", "v2", "a", "music", 1.0, 0)]
        stats, _ = aggregate_pair(rows, 1, 4)
        self.assertEqual(stats[("u1", "sports")], [1, 1])
        self.assertEqual(stats[("u1", "music")], [0, 1])

    def test_json_log_accepts_numpy_style_scalar(self):
        class Scalar:
            def item(self):
                return 0.6016

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "record.json"
            _write_json(output, {"primary": Scalar()})
            self.assertEqual(json.loads(output.read_text())["primary"], 0.6016)

    def test_controller_keeps_best_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(FakeRunner(), DeterministicResearcher(), self.config,
                self.project, base / "logs", base / "artifacts", base / "submissions")
            summary = controller.run(max_iterations=3)
            self.assertAlmostEqual(summary["best_primary"], 0.61)
            self.assertTrue((base / "artifacts" / "best_config.json").is_file())
            self.assertTrue((base / "submissions" / "final.csv").is_file())
            records = sorted((base / "logs").glob("iteration_*.json"))
            self.assertEqual(len(records), 3)
            self.assertEqual(json.loads(records[1].read_text())["decision"], "KEEP")
            self.assertNotIn("test", json.loads(records[1].read_text())["metrics"])
            self.assertTrue((base / "logs" / "experiment_history.jsonl").is_file())
            tree = json.loads((base / "logs" / "tree_snapshot.json").read_text())
            self.assertEqual(tree["nodes"][1]["parent_id"], "baseline")
            self.assertEqual(summary["final_test_metrics"]["primary"], 0.60)

    def test_official_evaluator_matches_pinned_digest(self):
        expected = self.project["official_evaluator_sha256"]
        actual = hashlib.sha256((ROOT / "kuairand-starter-kit" / "evaluate.py").read_bytes()).hexdigest()
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
