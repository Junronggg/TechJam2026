from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from techjam_agent.config import FEATURE_SCHEMA_VERSION, apply_changes
from techjam_agent.controller import Controller
from techjam_agent.experiment_planner import AutonomousExperimentPlanner, rank_candidates
from techjam_agent.memory import build_structured_research_memory, distill_research_patterns
from techjam_agent.proposals import Proposal


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text())


def load_project() -> dict:
    return json.loads((ROOT / "configs" / "project.json").read_text())


class ScoredRunner:
    def run(self, config, checkpoint):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        control = config["hyperparameters"]["feature_control"]
        if not config["features"]["prior_video_positive"]:
            score = 0.6000
        else:
            score = {
                "real": 0.6002,
                "constant": 0.6003,
                "shuffled": 0.6001,
                "random_same_cardinality": 0.60015,
            }[control]
        return {"GAUC": score, "nDCG@5": score, "primary": score,
                "runtime_seconds": 0.01}

    def finalize(self, config, checkpoint, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("row_id,user_id,video_id,score\n", encoding="utf-8")
        return {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5}


class OneFeatureResearcher:
    def __init__(self):
        self.last_selection = {
            "selected_family": "candidate_history",
            "selected_score": 0.5,
            "criteria": "test",
            "ranked_candidates": [],
        }

    def propose(self, best, history):
        return Proposal(
            "Test prior-video history.",
            "Candidate-specific history needs attribution controls.",
            {"prior_video_positive": True},
            "deterministic",
        )


class CandidatePlanningTests(unittest.TestCase):
    def test_initial_ranking_prefers_bpr_and_explains_score(self):
        ranked = rank_candidates(load_config(), [])
        self.assertEqual(ranked[0].candidate.family, "ranking_objective")
        self.assertEqual(ranked[0].candidate.changes, {
            "training_objective": "bpr", "learning_rate": 0.0003,
        })
        payload = ranked[0].as_dict()
        for key in ("expected_gain", "evidence_strength", "novelty",
                    "compute_cost", "redundancy", "score", "action_type",
                    "hard_blocked", "soft_stopped", "evidence_reasons"):
            self.assertIn(key, payload)
        self.assertEqual(ranked[0].candidate.action_type, "TRY_MODEL")

    def test_repeated_noise_stops_a_direction_without_slice_or_diversity_gain(self):
        history = []
        for iteration in (1, 2):
            history.append({
                "iteration": iteration,
                "config": copy.deepcopy(load_config()),
                "changes": {"learning_rate": 0.002},
                "delta_from_parent": 0.00001,
                "candidate_selection": {"selected_family": "global_context"},
                "critique": {"verdict": "noise"},
                "diagnostics": {},
            })
        global_context = next(
            row for row in rank_candidates(load_config(), history)
            if row.candidate.family == "global_context"
        )
        self.assertTrue(global_context.direction_stopped)
        self.assertFalse(global_context.hard_blocked)
        self.assertTrue(global_context.soft_stopped)
        no_memory = next(
            row for row in rank_candidates(
                load_config(), history, memory_mode="no_memory"
            ) if row.candidate.family == "global_context"
        )
        self.assertFalse(no_memory.hard_blocked)
        self.assertEqual(no_memory.evidence_reasons, ("missing_leakage_evidence",))


class AutonomousControlTests(unittest.TestCase):
    def test_small_categorical_gain_triggers_controls_and_rolls_back_fake_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                ScoredRunner(), OneFeatureResearcher(), load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            summary = controller.run(max_iterations=5)
            self.assertAlmostEqual(summary["best_primary"], 0.6000)
            self.assertEqual(
                [row["decision"] for row in controller.history],
                ["KEEP", "REINTERPRET", "CONTROL", "CONTROL", "CONTROL"],
            )
            real = controller.history[1]
            self.assertEqual(real["diagnostics"]["placebo_verdict"], "REINTERPRET")
            memory = json.loads((base / "logs" / "research_memory.json").read_text())
            self.assertEqual(memory["hypotheses"][0]["status"], "reinterpreted")

    def test_manual_intervention_count_is_audited_not_hardcoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            controller = Controller(
                ScoredRunner(), OneFeatureResearcher(), load_config(), load_project(),
                base / "logs", base / "artifacts", base / "submissions",
            )
            controller.record_intervention("GPU failed", "restarted worker", False)
            summary = controller.run(max_iterations=1)
            self.assertEqual(summary["manual_interventions"], 1)
            self.assertEqual(summary["avoidable_manual_interventions"], 0)
            event = json.loads(
                (base / "logs" / "manual_interventions.jsonl").read_text()
            )
            self.assertEqual(event["intervention_id"], "manual_001")


class StructuredMemoryTests(unittest.TestCase):
    def test_memory_records_reinterpreted_evidence_without_test_metrics(self):
        item = {
            "iteration": 1,
            "hypothesis": "candidate history helps",
            "changes": {"prior_video_positive": True},
            "status": "success",
            "decision": "REINTERPRET",
            "metrics": {"primary": 0.6002, "test_primary": 0.999},
            "delta_from_parent": 0.0002,
            "delta_from_best": 0.0002,
            "candidate_selection": {"selected_family": "candidate_history"},
            "critique": {"verdict": "noise", "confidence": "low",
                           "interpretation": "small gain", "next_test": "control"},
            "diagnostics": {"placebo_verdict": "REINTERPRET"},
            "config": apply_changes(load_config(), {"prior_video_positive": True}),
        }
        memory = build_structured_research_memory([item])
        self.assertEqual(memory["hypotheses"][0]["status"], "reinterpreted")
        self.assertEqual(memory["version"], 2)
        self.assertEqual(memory["research_patterns"][0]["policy"], "gather_evidence")
        self.assertNotIn("0.999", json.dumps(memory))

    def test_distilled_pattern_stops_repeated_unsupported_family(self):
        history = []
        for iteration, verdict in ((1, "reject"), (2, "noise")):
            history.append({
                "iteration": iteration,
                "status": "success",
                "decision": "REJECT",
                "delta_from_parent": -0.0001,
                "candidate_selection": {"selected_family": "candidate_history"},
                "critique": {"verdict": verdict},
                "diagnostics": {},
            })
        pattern = distill_research_patterns(history)[0]
        self.assertEqual(pattern["family"], "candidate_history")
        self.assertEqual(pattern["policy"], "stop_direction")

    def test_suspicious_gain_becomes_retest_with_control_pattern(self):
        pattern = distill_research_patterns([{
            "iteration": 1,
            "status": "success",
            "decision": "KEEP",
            "delta_from_parent": 0.0002,
            "candidate_selection": {"selected_family": "candidate_history"},
            "critique": {"verdict": "promote"},
            "diagnostics": {"placebo_status": "scheduled"},
        }])[0]
        self.assertEqual(pattern["policy"], "retest_with_control")
        self.assertEqual(pattern["evidence"]["control_pending"], 1)

    def test_planner_records_matching_research_pattern(self):
        history = [{
            "iteration": 1,
            "status": "success",
            "decision": "KEEP",
            "delta_from_parent": 0.002,
            "candidate_selection": {"selected_family": "ranking_objective"},
            "critique": {"verdict": "promote"},
            "diagnostics": {},
        }]
        planner = AutonomousExperimentPlanner()
        planner.select(load_config(), history)
        self.assertIsNotNone(planner.last_selection)
        self.assertEqual(
            planner.last_selection["retrieved_pattern"]["policy"],
            "exploit_with_confirmation",
        )

    def test_memory_modes_are_recorded_and_patterns_add_distilled_guidance(self):
        history = [{
            "iteration": 1,
            "status": "success",
            "decision": "KEEP",
            "delta_from_parent": 0.002,
            "candidate_selection": {"selected_family": "ranking_objective"},
            "critique": {"verdict": "promote"},
            "diagnostics": {},
        }]
        raw_score = next(
            row.score for row in rank_candidates(
                load_config(), history, memory_mode="raw_history"
            ) if row.candidate.family == "ranking_objective"
        )
        distilled_score = next(
            row.score for row in rank_candidates(
                load_config(), history, memory_mode="distilled_patterns"
            ) if row.candidate.family == "ranking_objective"
        )
        self.assertGreater(distilled_score, raw_score)
        planner = AutonomousExperimentPlanner(memory_mode="no_memory")
        planner.select(load_config(), history)
        self.assertEqual(planner.last_selection["memory_mode"], "no_memory")
        self.assertIsNone(planner.last_selection["retrieved_pattern"])
        self.assertEqual(
            set(planner.last_selection["counterfactual_choices"]),
            {"no_memory", "raw_history", "distilled_patterns"},
        )
        self.assertIsInstance(planner.last_selection["memory_changed_choice"], bool)

    def test_persistent_stop_policy_only_affects_distilled_mode(self):
        evidence = {"family_policies": [{
            "family": "temporal_counts",
            "policy": "stop_direction",
            "confidence": "high",
            "evidence": "failed rolling validation",
        }]}
        distilled = next(
            row for row in rank_candidates(
                load_config(), [], memory_mode="distilled_patterns",
                prior_evidence=evidence,
            ) if row.candidate.family == "temporal_counts"
        )
        raw = next(
            row for row in rank_candidates(
                load_config(), [], memory_mode="raw_history",
                prior_evidence=evidence,
            ) if row.candidate.family == "temporal_counts"
        )
        self.assertTrue(distilled.direction_stopped)
        self.assertFalse(raw.hard_blocked)
        self.assertEqual(raw.evidence_reasons, ("missing_leakage_evidence",))

    def test_scoped_policy_only_stops_matching_model(self):
        evidence = {"family_policies": [{
            "family": "temporal_counts",
            "policy": "stop_direction",
            "confidence": 0.9,
            "applies_to": {
                "task": "long_view",
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "models": ["ensemble"],
            },
        }]}
        fm_temporal = next(
            row for row in rank_candidates(
                load_config(), [], prior_evidence=evidence,
            ) if row.candidate.family == "temporal_counts"
        )
        ensemble = apply_changes(load_config(), {
            "model": "ensemble", "training_objective": "hybrid",
        })
        ensemble_temporal = next(
            row for row in rank_candidates(
                ensemble, [], prior_evidence=evidence,
            ) if row.candidate.family == "temporal_counts"
        )
        self.assertFalse(fm_temporal.hard_blocked)
        self.assertEqual(fm_temporal.evidence_reasons, ("missing_leakage_evidence",))
        self.assertTrue(ensemble_temporal.direction_stopped)

    def test_policy_expires_when_feature_schema_changes(self):
        evidence = {"family_policies": [{
            "family": "temporal_counts",
            "policy": "stop_direction",
            "confidence": 0.9,
            "applies_to": {
                "task": "long_view",
                "feature_schema": "obsolete-schema",
                "models": ["fm"],
            },
        }]}
        temporal = next(
            row for row in rank_candidates(
                load_config(), [], prior_evidence=evidence,
            ) if row.candidate.family == "temporal_counts"
        )
        self.assertFalse(temporal.hard_blocked)
        self.assertEqual(temporal.evidence_reasons, ("missing_leakage_evidence",))

    def test_policy_only_applies_when_scoped_features_are_present(self):
        evidence = {"family_policies": [{
            "family": "temporal_counts",
            "policy": "stop_direction",
            "confidence": 0.9,
            "applies_to": {
                "task": "long_view",
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "models": ["ensemble"],
                "features": {
                    "user_recent_3d_activity": True,
                    "item_recent_3d_exposure": True,
                },
                "hyperparameters": {"ensemble_deepfm_weight": 0.4},
            },
        }]}
        ensemble = apply_changes(load_config(), {
            "model": "ensemble", "training_objective": "hybrid",
        })
        user_only = next(
            row for row in rank_candidates(
                ensemble, [], prior_evidence=evidence,
            ) if row.candidate.changes == {"user_recent_3d_activity": True}
        )
        ensemble_with_user = apply_changes(
            ensemble, {"user_recent_3d_activity": True}
        )
        combined = next(
            row for row in rank_candidates(
                ensemble_with_user, [], prior_evidence=evidence,
            ) if row.candidate.changes == {"item_recent_3d_exposure": True}
        )
        self.assertFalse(user_only.hard_blocked)
        self.assertEqual(user_only.evidence_reasons, ("missing_leakage_evidence",))
        self.assertTrue(combined.direction_stopped)

    def test_generated_evidence_stops_exact_rejected_pairwise_multitask(self):
        evidence = json.loads(
            (ROOT / "configs" / "generated_family_policies.json").read_text(
                encoding="utf-8"
            )
        )
        pairwise = next(
            row for row in rank_candidates(
                load_config(), [], prior_evidence=evidence,
            ) if row.candidate.family == "pairwise_multitask"
        )
        self.assertEqual(pairwise.candidate.changes, {
            "model": "multitask_deepfm",
            "training_objective": "bpr",
            "learning_rate": 0.001,
        })
        self.assertTrue(pairwise.direction_stopped)


if __name__ == "__main__":
    unittest.main()
