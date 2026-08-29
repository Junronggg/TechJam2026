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

    def test_researcher_reproduces_best_bpr_before_new_ablation(self):
        proposal = DeterministicResearcher().propose(self.config, [])
        self.assertEqual(
            proposal.changes,
            {"training_objective": "bpr", "learning_rate": 0.0003},
        )

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

    def test_pair_sampler_supports_multiple_negatives_per_positive(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.bpr import build_pair_indices
        users = ["u1", "u1", "u2", "u2"]
        labels = np.asarray([1, 0, 1, 0])
        positives, negatives = build_pair_indices(
            users, labels, np.random.default_rng(0), pairs_per_positive=4
        )
        self.assertEqual(len(positives), 8)
        self.assertTrue(all(users[p] == users[n] for p, n in zip(positives, negatives)))

    def test_pair_sampler_selects_highest_scored_candidate(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.bpr import build_pair_indices
        users = ["u1", "u1", "u1"]
        labels = np.asarray([1, 0, 0])
        scores = np.asarray([0.0, 0.1, 0.9])
        _, negatives = build_pair_indices(
            users,
            labels,
            np.random.default_rng(0),
            negative_scores=scores,
            hard_negative_candidates=100,
        )
        self.assertEqual(negatives.tolist(), [2])

    def test_deepfm_supports_bce_bpr_and_checkpoint_state(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.deepfm import DeepFM

        positive = np.asarray([[0, 2]], dtype=np.int32)
        negative = np.asarray([[0, 3]], dtype=np.int32)
        model = DeepFM(4, fields=2, embedding_dim=4, hidden_dim=4,
                       learning_rate=0.01, seed=0)
        model.step(np.vstack([positive, negative]), np.asarray([1.0, 0.0]))
        before = float(model.predict(positive)[0] - model.predict(negative)[0])
        for _ in range(20):
            model.bpr_step(positive, negative)
        after = float(model.predict(positive)[0] - model.predict(negative)[0])
        self.assertGreater(after, before)

        restored = DeepFM(4, fields=2, embedding_dim=4, hidden_dim=4, seed=1)
        restored.load_state_dict(model.state_dict())
        np.testing.assert_allclose(restored.predict(positive), model.predict(positive))
        hybrid_before = float(restored.predict(positive)[0] - restored.predict(negative)[0])
        restored.hybrid_step(positive, negative, bpr_weight=0.75)
        hybrid_after = float(restored.predict(positive)[0] - restored.predict(negative)[0])
        self.assertGreater(hybrid_after, hybrid_before)

    def test_multitask_deepfm_trains_shared_representation(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.deepfm import MultiTaskDeepFM

        X = np.asarray([[0, 2], [1, 3]], dtype=np.int32)
        main = np.asarray([1.0, 0.0], dtype=np.float32)
        auxiliary = np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        model = MultiTaskDeepFM(
            4, fields=2, embedding_dim=4, hidden_dim=4,
            learning_rate=0.01, seed=0,
        )
        before = model.A.copy()
        loss = model.multitask_step(X, main, auxiliary, auxiliary_weight=0.1)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(before, model.A))
        self.assertEqual(model.predict(X).shape, (2,))
        self.assertIn("A", model.state_dict())

    def test_multitask_deepfm_requires_bce(self):
        multi = apply_changes(self.config, {"model": "multitask_deepfm"})
        validate_config(multi)
        with self.assertRaises(ValueError):
            apply_changes(multi, {"training_objective": "bpr"})

    def test_auxiliary_feedback_aligns_with_starter_rows(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.feedback import LOG_FILES, align_auxiliary_labels

        header = (
            "user_id,video_id,date,is_click,is_like,long_view,duration_ms,tab\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / LOG_FILES[0]).write_text(
                header + "u1,v1,20220409,1,0,1,1000,t1\n", encoding="utf-8"
            )
            (data_dir / LOG_FILES[1]).write_text(header, encoding="utf-8")
            splits = {
                "train": [(20220409, "u1", "v1", "a1", "t1", 1000.0, 1)]
            }
            labels = align_auxiliary_labels(data_dir, splits)
            np.testing.assert_array_equal(labels["train"], [[1.0, 0.0]])

    def test_hybrid_objective_is_a_legal_fm_configuration(self):
        hybrid = apply_changes(
            self.config,
            {"training_objective": "hybrid", "hybrid_bpr_weight": 0.75},
        )
        validate_config(hybrid)

    def test_researcher_tests_ensemble_before_hybrid_loss(self):
        bpr = apply_changes(
            self.config, {"training_objective": "bpr", "learning_rate": 0.0003}
        )
        history = [{"config": self.config}, {"config": bpr}]
        proposal = DeterministicResearcher().propose(bpr, history)
        self.assertEqual(
            proposal.changes,
            {"model": "ensemble", "training_objective": "hybrid",
             "ensemble_deepfm_weight": 0.4},
        )

        for value in (0.4, 0.3, 0.5):
            history.append({"config": apply_changes(
                bpr, {"model": "ensemble", "training_objective": "hybrid",
                      "ensemble_deepfm_weight": value}
            )})
        proposal = DeterministicResearcher().propose(bpr, history)
        self.assertEqual(
            proposal.changes,
            {"training_objective": "hybrid", "hybrid_bpr_weight": 0.75},
        )

    def test_ensemble_blend_normalizes_within_user(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.ensemble import blend_scores

        users = ["u1", "u1", "u2", "u2"]
        fm = np.asarray([1.0, 2.0, 100.0, 200.0])
        deepfm = np.asarray([10.0, 20.0, -2.0, -1.0])
        blended = blend_scores(users, fm, deepfm, 0.4)
        self.assertGreater(blended[1], blended[0])
        self.assertGreater(blended[3], blended[2])

    def test_temporal_count_uses_prior_days_only(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.temporal_features import strict_past_window_counts

        splits = {
            "train": [
                (20220409, "u1", "v1", "a", "t", 1.0, 0),
                (20220409, "u1", "v2", "a", "t", 1.0, 1),
                (20220410, "u1", "v3", "a", "t", 1.0, 0),
            ],
            "valid": [(20220412, "u1", "v4", "a", "t", 1.0, 1)],
            "test": [(20220413, "u1", "v5", "a", "t", 1.0, 0)],
        }
        counts = strict_past_window_counts(splits, key_index=1, window_days=3)
        np.testing.assert_array_equal(counts["train"], [0, 0, 2])
        np.testing.assert_array_equal(counts["valid"], [3])
        np.testing.assert_array_equal(counts["test"], [2])

    def test_rolling_folds_are_expanding_and_future_only(self):
        from techjam_agent.rolling import build_rolling_splits

        rows = [(date, "u", f"v{date}", "a", "t", 1.0, 0)
                for date in range(20220408, 20220424)]
        folds = build_rolling_splits(rows)
        self.assertEqual([len(folds[name]["train"]) for name in folds], [7, 10, 13])
        self.assertEqual([len(folds[name]["valid"]) for name in folds], [3, 3, 3])
        for split in folds.values():
            self.assertLess(max(row[0] for row in split["train"]),
                            min(row[0] for row in split["valid"]))

    def test_researcher_tests_multitask_after_best_ensemble(self):
        ensemble = apply_changes(
            self.config,
            {"model": "ensemble", "training_objective": "hybrid",
             "ensemble_deepfm_weight": 0.4},
        )
        proposal = DeterministicResearcher().propose(
            ensemble, [{"config": self.config}, {"config": ensemble}]
        )
        self.assertEqual(proposal.changes, {
            "model": "multitask_deepfm",
            "training_objective": "bce",
            "learning_rate": 0.001,
        })

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

    def test_explicit_cross_encoder_uses_train_vocabulary_and_unknown(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.runner import ExperimentRunner

        train = [(0, "u1", "v1", "a1", "sports", 1.0, 1)]
        valid = [(1, "u1", "v2", "a1", "sports", 1.0, 0),
                 (1, "u2", "v3", "a2", "music", 1.0, 1)]
        base = {
            "train": (np.zeros((1, 5), dtype=np.int32), np.asarray([1]), ["u1"]),
            "valid": (np.zeros((2, 5), dtype=np.int32), np.asarray([0, 1]), ["u1", "u2"]),
        }
        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._splits = {"train": train, "valid": valid}
        runner._encoded = (base, 10)
        config = apply_changes(self.config, {"user_tab_cross": True})
        encoded, dimension = runner._encoded_for(config)
        self.assertEqual(encoded["train"][0].shape, (1, 6))
        self.assertEqual(encoded["train"][0][0, -1], encoded["valid"][0][0, -1])
        self.assertNotEqual(encoded["train"][0][0, -1], encoded["valid"][0][1, -1])
        self.assertEqual(dimension, 12)

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
