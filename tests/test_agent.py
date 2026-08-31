from __future__ import annotations

import copy
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes, validate_config
from techjam_agent.causal_sequence import strict_past_sequences
from techjam_agent.controller import Controller, _write_json
from techjam_agent.proposals import DeterministicResearcher
from techjam_agent.history_features import aggregate, aggregate_pair, smoothed_rate_bucket
from techjam_agent.research_diagnostics import (
    build_slice_values,
    categorical_placebos,
    conditional_complementarity,
    evaluate_slices,
    placebo_verdict,
    strict_history_lengths,
)
from techjam_agent.sequence_model import LightweightSequenceDeepFM


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

        regression = MultiTaskDeepFM(
            4, fields=2, embedding_dim=4, hidden_dim=4,
            learning_rate=0.01, seed=1, auxiliary_tasks=1,
        )
        regression_loss = regression.multitask_step(
            X,
            main,
            np.asarray([[0.8], [0.2]], dtype=np.float32),
            auxiliary_weight=0.1,
            auxiliary_loss="mse",
        )
        self.assertTrue(np.isfinite(regression_loss))

        below = MultiTaskDeepFM(
            4, fields=2, embedding_dim=4, hidden_dim=4,
            learning_rate=0.0, seed=2, auxiliary_tasks=1,
        )
        above = MultiTaskDeepFM(
            4, fields=2, embedding_dim=4, hidden_dim=4,
            learning_rate=0.0, seed=2, auxiliary_tasks=1,
        )
        below.A.fill(0.0)
        above.A.fill(0.0)
        below.ab.fill(np.log(0.2 / 0.8))
        above.ab.fill(np.log(0.8 / 0.2))
        target = np.full((2, 1), 0.5, dtype=np.float32)
        censored = np.ones((2, 1), dtype=np.float32)
        below_loss = below.multitask_step(
            X, main, target, 0.1, auxiliary_loss="censored_mse",
            auxiliary_censored=censored,
        )
        above_loss = above.multitask_step(
            X, main, target, 0.1, auxiliary_loss="censored_mse",
            auxiliary_censored=censored,
        )
        self.assertGreater(below_loss, above_loss)

    def test_pairwise_multitask_improves_ranking_and_auxiliary_head(self):
        from techjam_agent.deepfm import MultiTaskDeepFM

        positive = np.asarray([[0, 2], [1, 2]], dtype=np.int32)
        negative = np.asarray([[0, 3], [1, 3]], dtype=np.int32)
        positive_auxiliary = np.asarray([[1.0], [1.0]], dtype=np.float32)
        negative_auxiliary = np.asarray([[0.0], [0.0]], dtype=np.float32)
        model = MultiTaskDeepFM(
            4, fields=2, embedding_dim=4, hidden_dim=4,
            learning_rate=0.01, seed=0, auxiliary_tasks=1,
        )
        before_margin = float(
            np.mean(model.predict(positive) - model.predict(negative))
        )
        before_head = model.A.copy()
        for _ in range(20):
            loss = model.pairwise_multitask_step(
                positive,
                negative,
                positive_auxiliary,
                negative_auxiliary,
                auxiliary_weight=0.1,
            )
        after_margin = float(
            np.mean(model.predict(positive) - model.predict(negative))
        )
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(after_margin, before_margin)
        self.assertFalse(np.array_equal(before_head, model.A))

    def test_dcnv2_trains_and_restores(self):
        from techjam_agent.dcnv2 import DCNv2

        X = np.asarray([[0, 2], [1, 3]], dtype=np.int32)
        labels = np.asarray([1.0, 0.0], dtype=np.float32)
        model = DCNv2(
            4, fields=2, embedding_dim=4, hidden_dim=4,
            cross_layers=2, cross_rank=2, learning_rate=0.01, seed=0,
        )
        before = model.predict(X)
        loss = model.step(X, labels)
        after = model.predict(X)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(before, after))

        restored = DCNv2(
            4, fields=2, embedding_dim=4, hidden_dim=4,
            cross_layers=2, cross_rank=2, seed=1,
        )
        restored.load_state_dict(model.state_dict())
        np.testing.assert_allclose(restored.predict(X), after)

    def test_dcnv2_requires_bce(self):
        dcn = apply_changes(self.config, {"model": "dcnv2"})
        validate_config(dcn)
        with self.assertRaises(ValueError):
            apply_changes(dcn, {"training_objective": "bpr"})

    def test_multitask_deepfm_supports_bce_and_bpr_only(self):
        multi = apply_changes(self.config, {"model": "multitask_deepfm"})
        validate_config(multi)
        pairwise = apply_changes(multi, {"training_objective": "bpr"})
        validate_config(pairwise)
        with self.assertRaisesRegex(ValueError, "not hybrid"):
            apply_changes(multi, {"training_objective": "hybrid"})

    def test_sequence_deepfm_requires_bce(self):
        sequence = apply_changes(self.config, {"model": "sequence_deepfm"})
        validate_config(sequence)
        with self.assertRaises(ValueError):
            apply_changes(sequence, {"training_objective": "bpr"})

    def test_lightweight_sequence_model_trains_and_restores(self):
        model = LightweightSequenceDeepFM(
            64, 5, embedding_dim=4, hidden_dim=6, learning_rate=0.001,
            seed=3, sequence_length=2,
        )
        X = np.asarray([[1, 10, 20, 30, 40], [2, 11, 21, 31, 41]], dtype=np.int32)
        history = {
            "video_id": np.asarray([[0, 9], [8, 10]], dtype=np.int32),
            "author_id": np.asarray([[0, 19], [18, 20]], dtype=np.int32),
            "behavior": np.asarray([[0, 2], [1, 2]], dtype=np.int8),
            "time_gap": np.asarray([[0, 1], [2, 1]], dtype=np.int8),
            "mask": np.asarray([[0, 1], [1, 1]], dtype=np.float32),
        }
        before = model.predict(X, history)
        loss = model.step(X, history, np.asarray([1, 0], dtype=np.float32))
        after = model.predict(X, history)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(after)))
        self.assertFalse(np.allclose(before, after))
        state = model.state_dict()
        restored = LightweightSequenceDeepFM(
            64, 5, embedding_dim=4, hidden_dim=6, learning_rate=0.001,
            seed=9, sequence_length=2,
        )
        restored.load_state_dict(state)
        np.testing.assert_allclose(restored.predict(X, history), after)

    def test_auxiliary_feedback_aligns_with_starter_rows(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy unavailable in this interpreter")
        from techjam_agent.feedback import (
            LOG_FILES, align_auxiliary_feedback, align_censored_watch_feedback,
        )

        header = (
            "user_id,video_id,date,is_click,is_like,long_view,play_time_ms,"
            "duration_ms,tab\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / LOG_FILES[0]).write_text(
                header + "u1,v1,20220409,1,0,1,1500,1000,t1\n", encoding="utf-8"
            )
            (data_dir / LOG_FILES[1]).write_text(header, encoding="utf-8")
            splits = {
                "train": [(20220409, "u1", "v1", "a1", "t1", 1000.0, 1)]
            }
            labels, masks = align_auxiliary_feedback(data_dir, splits)
            np.testing.assert_array_equal(labels["train"], [[1.0, 0.0, 1.0, 1.0]])
            np.testing.assert_array_equal(masks["train"], [[1.0, 1.0, 1.0, 1.0]])
            targets, watch_masks, censored = align_censored_watch_feedback(
                data_dir, splits
            )
            np.testing.assert_array_equal(watch_masks["train"], [[1.0]])
            np.testing.assert_array_equal(censored["train"], [[1.0]])
            np.testing.assert_array_equal(targets["train"], [[1.0]])

    def test_hybrid_objective_is_a_legal_fm_configuration(self):
        hybrid = apply_changes(
            self.config,
            {"training_objective": "hybrid", "hybrid_bpr_weight": 0.75},
        )
        validate_config(hybrid)

    def test_researcher_switches_mechanism_after_ensemble_weights_are_exhausted(self):
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
        self.assertEqual(proposal.changes, {
            "model": "multitask_deepfm",
            "training_objective": "bce",
            "learning_rate": 0.001,
        })

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

    def test_sequence_features_use_only_strict_past_train_labels(self):
        from techjam_agent.sequence_features import strict_sequence_categories

        splits = {
            "train": [
                (20220409, "u1", "v1", "a1", "t", 1.0, 1),
                (20220409, "u1", "v1", "a1", "t", 1.0, 0),
                (20220410, "u1", "v2", "a2", "t", 1.0, 0),
            ],
            "valid": [
                (20220411, "u1", "v1", "a1", "t", 1.0, 1),
                (20220411, "u1", "v3", "a1", "t", 1.0, 1),
            ],
            "test": [(20220412, "u1", "v1", "a1", "t", 1.0, 0)],
        }
        hour = 60 * 60 * 1000
        event_times = {
            # Deliberately not in chronological row order.
            "train": np.asarray([hour, 0, 2 * hour], dtype=np.int64),
            "valid": np.asarray([25 * hour, 4 * 24 * hour], dtype=np.int64),
            "test": np.asarray([8 * 24 * hour], dtype=np.int64),
        }
        values = strict_sequence_categories(splits, event_times)
        self.assertEqual(values["train"]["prior_video_positive"].tolist(), [0, 0, 0])
        self.assertEqual(values["train"]["author_positive_recency"].tolist(), [0, 0, 0])
        self.assertEqual(values["valid"]["prior_video_positive"].tolist(), [1, 0])
        self.assertEqual(values["valid"]["author_positive_recency"].tolist(), [2, 4])
        self.assertEqual(values["test"]["author_positive_recency"].tolist(), [5])
        self.assertEqual(values["train"]["prior_video_count"].tolist(), [1, 0, 0])
        self.assertEqual(values["train"]["previous_author_same"].tolist(), [1, 0, 0])
        self.assertEqual(values["valid"]["prior_video_count"].tolist(), [2, 0])
        self.assertEqual(values["valid"]["previous_author_same"].tolist(), [0, 1])
        self.assertEqual(values["test"]["prior_video_count"].tolist(), [2])
        self.assertEqual(values["test"]["previous_author_same"].tolist(), [0])
        self.assertEqual(values["train"]["prior_video_exposure"].tolist(), [1, 0, 0])
        self.assertEqual(values["valid"]["prior_video_exposure"].tolist(), [1, 0])
        self.assertEqual(values["test"]["prior_video_exposure"].tolist(), [1])
        self.assertEqual(values["train"]["author_recency"].tolist(), [1, 0, 0])
        self.assertEqual(values["valid"]["author_recency"].tolist(), [2, 3])
        self.assertEqual(values["test"]["author_recency"].tolist(), [5])

        changed = {name: list(rows) for name, rows in splits.items()}
        changed["valid"] = [row[:-1] + (1 - row[-1],) for row in splits["valid"]]
        changed_values = strict_sequence_categories(changed, event_times)
        self.assertEqual(
            values["valid"]["author_positive_recency"].tolist(),
            changed_values["valid"]["author_positive_recency"].tolist(),
        )

    def test_researcher_tests_multitask_after_best_ensemble(self):
        ensemble = apply_changes(
            self.config,
            {"model": "ensemble", "training_objective": "hybrid",
             "ensemble_deepfm_weight": 0.4},
        )
        calibrated = apply_changes(
            ensemble, {"ensemble_normalization": "fm_zscore_deepfm_rank"}
        )
        tuned = apply_changes(calibrated, {"ensemble_deepfm_weight": 0.65})
        proposal = DeterministicResearcher().propose(
            ensemble, [{"config": self.config}, {"config": ensemble},
                       {"config": calibrated}, {"config": tuned}]
        )
        self.assertEqual(proposal.changes, {
            "model": "multitask_deepfm",
            "training_objective": "bce",
            "learning_rate": 0.001,
        })

    def test_researcher_tests_pairwise_multitask_after_pointwise_multitask(self):
        ensemble = apply_changes(
            self.config,
            {"model": "ensemble", "training_objective": "hybrid",
             "ensemble_deepfm_weight": 0.4},
        )
        calibrated = apply_changes(
            ensemble, {"ensemble_normalization": "fm_zscore_deepfm_rank"}
        )
        tuned = apply_changes(calibrated, {"ensemble_deepfm_weight": 0.65})
        multitask = apply_changes(
            ensemble,
            {"model": "multitask_deepfm", "training_objective": "bce",
             "learning_rate": 0.001},
        )
        proposal = DeterministicResearcher().propose(
            ensemble,
            [
                {"config": self.config},
                {"config": ensemble},
                {"config": calibrated},
                {"config": tuned},
                {"config": multitask},
            ],
        )
        self.assertEqual(proposal.changes, {
            "model": "multitask_deepfm",
            "training_objective": "bpr",
            "learning_rate": 0.001,
        })

    def test_researcher_tests_new_censored_objective_before_dcnv2(self):
        ensemble = apply_changes(
            self.config,
            {"model": "ensemble", "training_objective": "hybrid",
             "ensemble_deepfm_weight": 0.4},
        )
        calibrated = apply_changes(
            ensemble, {"ensemble_normalization": "fm_zscore_deepfm_rank"}
        )
        tuned = apply_changes(calibrated, {"ensemble_deepfm_weight": 0.65})
        pointwise = apply_changes(
            ensemble,
            {"model": "multitask_deepfm", "training_objective": "bce",
             "learning_rate": 0.001},
        )
        pairwise = apply_changes(
            ensemble,
            {"model": "multitask_deepfm", "training_objective": "bpr",
             "learning_rate": 0.001},
        )
        proposal = DeterministicResearcher().propose(
            ensemble,
            [{"config": self.config}, {"config": ensemble},
             {"config": calibrated}, {"config": tuned}, {"config": pointwise},
             {"config": pairwise}],
        )
        self.assertEqual(proposal.changes, {
            "model": "multitask_deepfm",
            "training_objective": "bce",
            "auxiliary_signals": "censored_watch",
            "learning_rate": 0.001,
        })

        censored_pointwise = apply_changes(ensemble, proposal.changes)
        censored_pairwise = apply_changes(ensemble, {
            "model": "multitask_deepfm",
            "training_objective": "bpr",
            "auxiliary_signals": "censored_watch",
            "learning_rate": 0.001,
        })
        proposal = DeterministicResearcher().propose(
            ensemble,
            [{"config": self.config}, {"config": ensemble},
             {"config": calibrated}, {"config": tuned}, {"config": pointwise},
             {"config": pairwise},
             {"config": censored_pointwise}, {"config": censored_pairwise}],
        )
        self.assertEqual(proposal.changes, {
            "model": "dcnv2",
            "training_objective": "bce",
            "learning_rate": 0.001,
        })

    def test_researcher_abandons_rejected_lightgbm_family_for_new_information(self):
        lgb = apply_changes(self.config, {"model": "lightgbm"})
        proposal = DeterministicResearcher().propose(lgb, [{"config": self.config}])
        self.assertEqual(proposal.changes, {
            "model": "multitask_deepfm",
            "training_objective": "bce",
            "learning_rate": 0.001,
        })

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

    def test_test_finalization_is_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            artifacts = base / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "final_test_metrics.json").write_text("{}", encoding="utf-8")
            controller = Controller(FakeRunner(), DeterministicResearcher(), self.config,
                self.project, base / "logs", artifacts, base / "submissions")
            with self.assertRaisesRegex(RuntimeError, "already been consumed"):
                controller.run(max_iterations=1)

    def test_history_availability_is_strict_and_target_free(self):
        splits = {
            "train": [
                (1, "u1", "v1", "a1", "t", 10_000.0, 1),
                (1, "u1", "v2", "a2", "t", 20_000.0, 0),
                (1, "u1", "v3", "a3", "t", 30_000.0, 1),
            ],
            "valid": [
                (2, "u1", "v4", "a4", "t", 40_000.0, 0),
                (2, "u1", "v5", "a5", "t", 130_000.0, 1),
                (2, "u1", "v6", "a6", "t", 50_000.0, 0),
            ],
        }
        times = {
            "train": np.asarray([100, 100, 200]),
            "valid": np.asarray([300, 300, 400]),
        }
        lengths = strict_history_lengths(splits, times)
        self.assertEqual(lengths["train"].tolist(), [0, 0, 2])
        self.assertEqual(lengths["valid"].tolist(), [3, 3, 5])
        changed = {
            name: [(*row[:-1], 1 - row[-1]) for row in rows]
            for name, rows in splits.items()
        }
        changed_lengths = strict_history_lengths(changed, times)
        np.testing.assert_array_equal(lengths["valid"], changed_lengths["valid"])
        slices = build_slice_values(splits, lengths)
        self.assertEqual(slices["history"].tolist(), ["medium_3_10"] * 3)
        self.assertEqual(
            slices["duration"].tolist(),
            ["medium_30_120s", "long_120s_plus", "medium_30_120s"],
        )

    def test_last_k_sequences_block_same_time_and_validation_labels(self):
        splits = {
            "train": [
                (1, "u1", "v1", "a1", "t", 10.0, 1),
                (1, "u1", "v2", "a2", "t", 10.0, 0),
                (1, "u1", "v3", "a3", "t", 10.0, 1),
            ],
            "valid": [
                (2, "u1", "v4", "a4", "t", 10.0, 1),
                (2, "u1", "v5", "a5", "t", 10.0, 0),
            ],
        }
        times = {
            "train": np.asarray([100, 100, 200]),
            "valid": np.asarray([300, 400]),
        }
        encoded = {
            "train": (
                np.asarray([[10, 20, 30, 40, 50], [11, 21, 31, 40, 50],
                            [12, 22, 32, 40, 50]], dtype=np.int32),
                np.asarray([1, 0, 1], dtype=np.float32),
                ["u1", "u1", "u1"],
            ),
            "valid": (
                np.asarray([[13, 23, 33, 40, 50], [14, 24, 34, 40, 50]],
                           dtype=np.int32),
                np.asarray([1, 0], dtype=np.float32),
                ["u1", "u1"],
            ),
        }
        sequences = strict_past_sequences(splits, times, encoded, max_length=4)
        self.assertEqual(sequences["train"]["length"].tolist(), [0, 0, 2])
        self.assertEqual(sequences["train"]["behavior"][2].tolist(), [0, 0, 2, 1])
        self.assertEqual(sequences["valid"]["length"].tolist(), [3, 4])
        # A past validation impression is exposure-only even when its label is 1.
        self.assertEqual(sequences["valid"]["behavior"][1, -1], 1)
        flipped = copy.deepcopy(encoded)
        flipped["valid"] = (
            encoded["valid"][0], 1 - encoded["valid"][1], encoded["valid"][2]
        )
        changed = strict_past_sequences(splits, times, flipped, max_length=4)
        np.testing.assert_array_equal(
            sequences["valid"]["behavior"], changed["valid"]["behavior"]
        )

    def test_placebo_suite_requires_real_feature_to_beat_controls(self):
        real = np.asarray([0, 1, 1, 2, 2, 2], dtype=np.int32)
        variants = categorical_placebos(real, cardinality=3, seed=7)
        self.assertEqual(set(variants), {
            "real", "constant", "shuffled", "random_same_cardinality"
        })
        self.assertEqual(sorted(variants["shuffled"].tolist()), sorted(real.tolist()))
        self.assertTrue(np.all(variants["constant"] == 0))
        rejected = placebo_verdict(0.6042, {"constant": 0.6044, "shuffled": 0.6040})
        self.assertEqual(rejected["verdict"], "REINTERPRET")
        kept = placebo_verdict(0.6046, {"constant": 0.6044, "shuffled": 0.6040})
        self.assertEqual(kept["verdict"], "KEEP_CANDIDATE")

    def test_slice_complementarity_reports_error_recovery(self):
        def evaluator(users, labels, scores):
            labels = np.asarray(labels)
            scores = np.asarray(scores)
            positive = float(scores[labels == 1].mean()) if np.any(labels == 1) else 0.0
            negative = float(scores[labels == 0].mean()) if np.any(labels == 0) else 0.0
            primary = positive - negative
            return {"GAUC": primary, "nDCG@5": primary, "primary": primary,
                    "users": len(set(users)), "rows": len(labels)}

        users = ["u1", "u1", "u2", "u2"]
        labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
        scores_a = np.asarray([0.0, 1.0, 0.8, 0.2], dtype=np.float32)
        scores_b = np.asarray([1.0, 0.0, 0.7, 0.3], dtype=np.float32)
        slices = {"history": np.asarray(["cold", "cold", "rich", "rich"], dtype=object)}
        metrics = evaluate_slices(evaluator, users, labels, scores_b, slices)
        self.assertIn("history=cold", metrics)
        comparison = conditional_complementarity(
            evaluator, users, labels, scores_a, scores_b, slices
        )
        self.assertGreater(comparison["overall"]["primary_delta_b_minus_a"], 0)
        self.assertEqual(comparison["overall"]["model_b_recovered_a_errors"], 1)
        self.assertEqual(comparison["overall"]["model_b_new_pair_errors"], 0)

    def test_official_evaluator_matches_pinned_digest(self):
        expected = self.project["official_evaluator_sha256"]
        actual = hashlib.sha256((ROOT / "kuairand-starter-kit" / "evaluate.py").read_bytes()).hexdigest()
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
