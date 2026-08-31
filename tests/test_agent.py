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

    def test_seed_and_bpr_negative_operators_are_allow_listed(self):
        seeded = apply_changes(self.config, {"seed": 4})
        self.assertEqual(seeded["hyperparameters"]["seed"], 4)
        with self.assertRaises(ValueError):
            apply_changes(self.config, {"negatives_per_positive": 2})
        bpr = apply_changes(self.config, {"training_objective": "bpr"})
        multi_negative = apply_changes(bpr, {"negatives_per_positive": 2})
        self.assertEqual(multi_negative["hyperparameters"]["negatives_per_positive"], 2)
        matched = apply_changes(bpr, {"negative_sampling_strategy": "same_tab"})
        self.assertEqual(matched["hyperparameters"]["negative_sampling_strategy"], "same_tab")
        with self.assertRaises(ValueError):
            apply_changes(self.config, {"negative_sampling_strategy": "same_tab"})

    def test_model_objective_compatibility(self):
        ranker = apply_changes(
            self.config, {"model": "lightgbm", "training_objective": "lambdarank"}
        )
        validate_config(ranker)
        with self.assertRaises(ValueError):
            apply_changes(self.config, {"training_objective": "lambdarank"})
        with self.assertRaises(ValueError):
            apply_changes(self.config, {"model": "lightgbm", "training_objective": "bpr"})
        ensemble = apply_changes(
            self.config,
            {"model": "fm_ensemble", "training_objective": "bpr", "ensemble_size": 4},
        )
        validate_config(ensemble)
        with self.assertRaises(ValueError):
            apply_changes(self.config, {"model": "fm_ensemble", "training_objective": "bpr"})
        selected = apply_changes(
            self.config,
            {"model": "fm_ensemble", "training_objective": "bpr",
             "ensemble_size": 2, "ensemble_seed_set": "3,4"},
        )
        validate_config(selected)

    def test_researcher_uses_ranked_non_duplicate_candidate_after_tuning_bpr(self):
        bpr = apply_changes(self.config, {"training_objective": "bpr"})
        tuned = apply_changes(bpr, {"learning_rate": 0.0005})
        history = [
            {"config": self.config},
            {"config": bpr},
            {"config": tuned},
        ]
        proposal = DeterministicResearcher().propose(tuned, history)
        from techjam_agent.proposals import legal_candidate_catalog
        legal = legal_candidate_catalog(tuned, history)
        self.assertIn(proposal.candidate_id, {item["candidate_id"] for item in legal})
        self.assertNotIn(proposal.changes, [item["config"] for item in history])

    def test_planner_registry_and_validator_share_values(self):
        from techjam_agent.config import ALLOWED_VALUES
        from techjam_agent.operator_registry import planner_registry

        registry = planner_registry()
        for field, values in ALLOWED_VALUES.items():
            self.assertEqual(tuple(registry[field]["values"]), tuple(values))

    def test_inactive_model_knobs_do_not_create_fake_new_experiments(self):
        from techjam_agent.config import experiment_key

        first = apply_changes(self.config, {"model": "lightgbm"})
        tuned_fm_then_tree = apply_changes(
            apply_changes(self.config, {"learning_rate": 0.0005}),
            {"model": "lightgbm"},
        )
        self.assertEqual(experiment_key(first), experiment_key(tuned_fm_then_tree))

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

    def test_multiple_bpr_negatives_are_same_user_and_deterministic(self):
        import numpy as np
        from techjam_agent.bpr import build_pair_indices

        users = ["u1", "u1", "u1", "u2", "u2"]
        labels = np.asarray([1, 0, 0, 1, 0])
        first = build_pair_indices(users, labels, np.random.default_rng(7), 4)
        second = build_pair_indices(users, labels, np.random.default_rng(7), 4)
        self.assertEqual(len(first[0]), 8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(all(users[p] == users[n] for p, n in zip(*first)))
        self.assertTrue(all(labels[p] == 1 and labels[n] == 0 for p, n in zip(*first)))

    def test_context_matched_negatives_use_match_then_fallback(self):
        import numpy as np
        from techjam_agent.bpr import build_pair_indices

        users = ["u", "u", "u", "u"]
        labels = np.asarray([1, 1, 0, 0])
        tabs = np.asarray(["sports", "music", "sports", "other"])
        positives, negatives = build_pair_indices(
            users, labels, np.random.default_rng(4), match_values=tabs
        )
        pairs = {positive: negative for positive, negative in zip(positives, negatives)}
        self.assertEqual(tabs[pairs[0]], "sports")
        self.assertEqual(labels[pairs[1]], 0)

    def test_ranking_group_builder_is_stable_and_complete(self):
        import numpy as np
        from techjam_agent.runner import _ranking_order_and_groups

        users = ["b", "a", "b", "c", "a"]
        order, groups = _ranking_order_and_groups(users)
        self.assertEqual(groups, [2, 2, 1])
        self.assertEqual(np.asarray(users)[order].tolist(), ["a", "a", "b", "b", "c"])

    def test_bpr_negative_count_is_validated(self):
        import numpy as np
        from techjam_agent.bpr import build_pair_indices

        with self.assertRaises(ValueError):
            build_pair_indices(["u", "u"], np.asarray([1, 0]), np.random.default_rng(0), 3)

    def test_researcher_selects_a_legal_candidate_after_lightgbm(self):
        lgb = apply_changes(self.config, {"model": "lightgbm"})
        proposal = DeterministicResearcher().propose(lgb, [{"config": self.config}])
        self.assertGreaterEqual(len(proposal.changes), 1)
        self.assertLessEqual(len(proposal.changes), 4)
        candidate = apply_changes(lgb, proposal.changes)
        validate_config(candidate)

    def test_user_tab_aggregation_keeps_preferences_separate(self):
        rows = [(0, "u1", "v1", "a", "sports", 1.0, 1),
                (0, "u1", "v2", "a", "music", 1.0, 0)]
        stats, _ = aggregate_pair(rows, 1, 4)
        self.assertEqual(stats[("u1", "sports")], [1, 1])
        self.assertEqual(stats[("u1", "music")], [0, 1])

    def test_author_and_affinity_train_features_leave_current_label_out(self):
        from techjam_agent.history_features import TrainHistoryStatistics

        other_rows = [
            (20220408, "u1", "v2", "a1", "sports", 10.0, 0),
            (20220408, "u2", "v3", "a1", "sports", 10.0, 1),
            (20220408, "u3", "v4", "a2", "music", 10.0, 0),
        ]
        target_zero = (20220408, "u1", "v1", "a1", "sports", 10.0, 0)
        target_one = (20220408, "u1", "v1", "a1", "sports", 10.0, 1)
        features = (
            "author_long_view_count", "author_long_view_rate",
            "user_author_long_view_count", "user_author_long_view_rate",
        )
        zero = TrainHistoryStatistics.build([target_zero, *other_rows], features)
        one = TrainHistoryStatistics.build([target_one, *other_rows], features)
        for feature in features:
            self.assertAlmostEqual(
                zero.value(feature, target_zero, leave_one_out=True),
                one.value(feature, target_one, leave_one_out=True),
            )

    def test_validation_affinity_does_not_read_validation_label(self):
        from techjam_agent.history_features import TrainHistoryStatistics

        train = [
            (20220408, "u1", "v1", "a1", "sports", 10.0, 1),
            (20220408, "u1", "v2", "a1", "sports", 10.0, 0),
        ]
        valid_zero = (20220422, "u1", "v3", "a1", "sports", 10.0, 0)
        valid_one = (20220422, "u1", "v3", "a1", "sports", 10.0, 1)
        statistics = TrainHistoryStatistics.build(train, ["user_author_long_view_rate"])
        self.assertEqual(
            statistics.numeric_value(
                "user_author_long_view_rate", valid_zero, leave_one_out=False
            ),
            statistics.numeric_value(
                "user_author_long_view_rate", valid_one, leave_one_out=False
            ),
        )

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
