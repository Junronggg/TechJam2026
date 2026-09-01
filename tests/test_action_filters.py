"""Milestone B: cheap feasibility filters on the hierarchical planner.

Fixtures only. No training, no API, no markdown parsing, no test-split access.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import FEATURE_SCHEMA_VERSION, apply_changes, experiment_key
from techjam_agent.evidence import DEFAULT_FEASIBILITY_THRESHOLDS, FEASIBILITY_SCHEMA_VERSION
from techjam_agent.experiment_planner import (
    ActionType,
    AutonomousExperimentPlanner,
    admissible_candidates,
    choose_ranked,
    generate_candidates,
    rank_candidates,
)
from techjam_agent.proposals import DeterministicResearcher


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def bpr_config() -> dict:
    return apply_changes(load_config(), {
        "training_objective": "bpr", "learning_rate": 0.0003,
    })


def tried_row(config: dict, changes: dict | None = None) -> dict:
    return {
        "config": copy.deepcopy(config),
        "changes": {} if changes is None else changes,
        "status": "success",
        "critique": {"verdict": "noise"},
        "diagnostics": {},
    }


def exhaust(best: dict, rows: list[dict] | None = None, *,
            keep: set[tuple[str, ...]] | None = None) -> list[dict]:
    history = list(rows or [])
    keep = keep or set()
    for candidate in generate_candidates(best):
        if tuple(sorted(candidate.changes)) in keep:
            continue
        history.append(tried_row(apply_changes(best, candidate.changes), candidate.changes))
    return history


def feasibility_prior(records: list[dict], thresholds: dict | None = None) -> dict:
    return {
        "feasibility": {
            "version": FEASIBILITY_SCHEMA_VERSION,
            "thresholds": {**DEFAULT_FEASIBILITY_THRESHOLDS, **(thresholds or {})},
            "records": records,
        }
    }


def coverage_record(feature: str, coverage: float, *,
                    eligible_rows: int | None = None,
                    total_rows: int | None = None) -> dict:
    return {
        "source_id": f"coverage_{feature}",
        "family": "global_context" if feature == "global_context" else feature,
        "kind": "feature_coverage",
        "sha256": "a" * 64,
        "applies_to": {
            "task": "long_view",
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "features": {feature: True},
        },
        "result": {
            "kind": "feature_coverage",
            "coverage": coverage,
            "eligible_rows": eligible_rows,
            "total_rows": total_rows,
        },
    }


def correlation_record(models: list[str], correlation: float) -> dict:
    return {
        "source_id": "corr_v1",
        "family": "heterogeneous_ensemble",
        "kind": "prediction_correlation",
        "sha256": "b" * 64,
        "applies_to": {"task": "long_view", "feature_schema": FEATURE_SCHEMA_VERSION},
        "result": {
            "kind": "prediction_correlation",
            "correlation": correlation,
            "models": sorted(models),
            "split": "validation",
        },
    }


def runtime_record(family: str, median: float, *, models: list[str] | None = None) -> dict:
    return {
        "source_id": f"runtime_{family}",
        "family": family,
        "kind": "family_runtime",
        "sha256": "c" * 64,
        "applies_to": {
            "task": "long_view",
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "models": models or [],
        },
        "result": {
            "kind": "family_runtime",
            "median_runtime_seconds": median,
            "runtime_seconds": [median],
            "observations": 1,
        },
    }


def safe_leakage(*features: str) -> list[dict]:
    return [
        leakage_record(feature, "safe", leakage_safe=True, strict_past=True)
        for feature in features
    ]


def robust_ensemble_stop() -> dict:
    return {
        "family_policies": [{
            "family": "heterogeneous_ensemble",
            "policy": "stop_direction",
            "scientific_verdict": "REJECTED",
            "confidence": 0.9,
            "applies_to": {
                "task": "long_view",
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "models": ["ensemble"],
                "training_objectives": ["hybrid"],
                "hyperparameters": {"ensemble_deepfm_weight": 0.4},
            },
            "created_from": [{
                "source_id": "ensemble_rolling_v1",
                "kind": "rolling_aggregate",
                "result": {
                    "signal": "negative",
                    "mean_delta": -0.0003,
                    "wins": 1,
                    "folds": 3,
                    "robust": True,
                },
            }],
        }]
    }


def live_replay(best: dict, history: list[dict], evidence: dict | None):
    live = DeterministicResearcher(prior_evidence=evidence).propose(best, history)
    selected = AutonomousExperimentPlanner(prior_evidence=evidence).select(best, history)
    replayed = choose_ranked(rank_candidates(best, history, prior_evidence=evidence))
    return live, selected, replayed


def leakage_record(feature: str, status: str, *,
                   leakage_safe: bool | None = None,
                   strict_past: bool | None = None) -> dict:
    return {
        "source_id": f"leakage_{feature}",
        "family": feature,
        "kind": "leakage_status",
        "sha256": "d" * 64,
        "applies_to": {
            "task": "long_view",
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "features": {feature: True},
        },
        "result": {
            "kind": "leakage_status",
            "status": status,
            "leakage_safe": leakage_safe,
            "strict_past": strict_past,
        },
    }


def row_for(ranked, **changes) -> object:
    return next(row for row in ranked if row.candidate.changes == changes)


class CoverageFilterTests(unittest.TestCase):
    def test_low_nonzero_coverage_is_soft_and_relaxable(self) -> None:
        evidence = feasibility_prior([
            coverage_record("global_context", 0.003, eligible_rows=4),
            *safe_leakage("global_context"),
        ])
        ranked = rank_candidates(load_config(), [], prior_evidence=evidence)
        soft = row_for(ranked, global_context=True)
        self.assertFalse(soft.hard_blocked)
        self.assertTrue(soft.soft_stopped)
        self.assertNotIn(soft, admissible_candidates(ranked, relax_soft=False))
        self.assertIn(soft, admissible_candidates(ranked, relax_soft=True))
        proposal = DeterministicResearcher(prior_evidence=evidence).propose(
            load_config(), [],
        )
        self.assertNotEqual(proposal.changes, {"global_context": True})
        history = exhaust(load_config(), keep={("global_context",)})
        recovered = AutonomousExperimentPlanner(prior_evidence=evidence).select(
            load_config(), history,
        )
        self.assertEqual(recovered.changes, {"global_context": True})

    def test_zero_coverage_exact_scope_can_be_hard(self) -> None:
        evidence = feasibility_prior([
            coverage_record("prior_video_positive", 0.0, eligible_rows=0, total_rows=100),
            *safe_leakage("prior_video_positive", "author_positive_recency"),
        ])
        ranked = rank_candidates(load_config(), [], prior_evidence=evidence)
        blocked = row_for(ranked, prior_video_positive=True)
        other = row_for(ranked, author_positive_recency=True)
        self.assertTrue(blocked.hard_blocked)
        self.assertFalse(blocked.soft_stopped)
        self.assertFalse(other.hard_blocked)
        self.assertFalse(other.soft_stopped)
        self.assertNotIn(blocked, admissible_candidates(ranked, relax_soft=True))
        history = exhaust(load_config(), keep={("prior_video_positive",)})
        with self.assertRaises(StopIteration):
            AutonomousExperimentPlanner(prior_evidence=evidence).select(
                load_config(), history,
            )

    def test_zero_coverage_without_eligible_row_proof_stays_soft(self) -> None:
        evidence = feasibility_prior([
            coverage_record("global_context", 0.0),
            *safe_leakage("global_context"),
        ])
        ranked = rank_candidates(load_config(), [], prior_evidence=evidence)
        row = row_for(ranked, global_context=True)
        self.assertFalse(row.hard_blocked)
        self.assertTrue(row.soft_stopped)

    def test_coverage_threshold_is_configurable(self) -> None:
        record = coverage_record("global_context", 0.02, eligible_rows=20)
        safe = safe_leakage("global_context")
        default = row_for(
            rank_candidates(
                load_config(), [],
                prior_evidence=feasibility_prior([record, *safe]),
            ),
            global_context=True,
        )
        raised = row_for(
            rank_candidates(
                load_config(), [],
                prior_evidence=feasibility_prior(
                    [record, *safe], {"low_coverage": 0.05},
                ),
            ),
            global_context=True,
        )
        self.assertFalse(default.soft_stopped)
        self.assertTrue(raised.soft_stopped)


class CorrelationFilterTests(unittest.TestCase):
    def test_high_correlation_alone_is_preferred_pass_soft(self) -> None:
        evidence = feasibility_prior([correlation_record(["fm", "deepfm"], 0.995)])
        ranked = rank_candidates(bpr_config(), [], prior_evidence=evidence)
        ensemble = next(
            row for row in ranked if row.candidate.action_type == ActionType.TRY_ENSEMBLE
        )
        self.assertFalse(ensemble.hard_blocked)
        self.assertTrue(ensemble.soft_stopped)
        self.assertEqual(ensemble.evidence_reasons, ("high_prediction_correlation",))
        self.assertNotIn(ensemble, admissible_candidates(ranked, relax_soft=False))
        self.assertIn(ensemble, admissible_candidates(ranked, relax_soft=True))
        for row in ranked:
            if row.candidate.action_type == ActionType.TRY_MODEL:
                self.assertFalse(row.hard_blocked)
                self.assertFalse(row.soft_stopped)
                self.assertNotIn("high_prediction_correlation", row.evidence_reasons)

    def test_high_correlation_is_recoverable_in_the_relaxed_pass(self) -> None:
        evidence = feasibility_prior([correlation_record(["fm", "deepfm"], 0.995)])
        history = exhaust(bpr_config(), keep={
            ("ensemble_deepfm_weight", "model", "training_objective"),
        })
        live, selected, replayed = live_replay(bpr_config(), history, evidence)
        self.assertEqual(selected.changes, live.changes)
        self.assertEqual(replayed[0].candidate.changes, live.changes)
        self.assertEqual(live.changes.get("model"), "ensemble")
        planner = AutonomousExperimentPlanner(prior_evidence=evidence)
        planner.select(bpr_config(), history)
        self.assertEqual(planner.last_selection["selection_pass"], "relaxed")
        self.assertIn(
            "high_prediction_correlation",
            planner.last_selection["evidence_reasons"],
        )

    def test_standalone_models_remain_executable(self) -> None:
        evidence = feasibility_prior([correlation_record(["fm", "deepfm"], 0.995)])
        ranked = rank_candidates(load_config(), [], prior_evidence=evidence)
        self.assertFalse(any(row.candidate.action_type == ActionType.TRY_ENSEMBLE
                             for row in ranked))
        for family in ("ranking_objective", "multitask", "cross_network", "tree_model"):
            row = next(item for item in ranked if item.candidate.family == family)
            self.assertFalse(row.hard_blocked)
            self.assertFalse(row.soft_stopped)
            self.assertEqual(row.candidate.action_type, ActionType.TRY_MODEL)
        live, selected, replayed = live_replay(load_config(), [], evidence)
        self.assertEqual(selected.changes, live.changes)
        self.assertEqual(replayed[0].candidate.changes, live.changes)
        self.assertEqual(live.changes.get("training_objective"), "bpr")

    def test_high_correlation_plus_robust_ensemble_stop_is_hard(self) -> None:
        evidence = {
            **robust_ensemble_stop(),
            **feasibility_prior([correlation_record(["fm", "deepfm"], 0.995)]),
        }
        ranked = rank_candidates(bpr_config(), [], prior_evidence=evidence)
        ensemble = next(
            row for row in ranked if row.candidate.action_type == ActionType.TRY_ENSEMBLE
        )
        self.assertTrue(ensemble.hard_blocked)
        self.assertFalse(ensemble.soft_stopped)
        self.assertIn("high_prediction_correlation", ensemble.evidence_reasons)
        self.assertNotIn(ensemble, admissible_candidates(ranked, relax_soft=True))
        history = exhaust(bpr_config(), keep={
            ("ensemble_deepfm_weight", "model", "training_objective"),
        })
        with self.assertRaises(StopIteration):
            DeterministicResearcher(prior_evidence=evidence).propose(bpr_config(), history)
        with self.assertRaises(StopIteration):
            AutonomousExperimentPlanner(prior_evidence=evidence).select(
                bpr_config(), history,
            )
        self.assertEqual(
            choose_ranked(rank_candidates(bpr_config(), history, prior_evidence=evidence)),
            [],
        )

    def test_unrelated_model_pair_does_not_block_fm_deepfm_ensemble(self) -> None:
        evidence = feasibility_prior([correlation_record(["deepfm", "dcnv2"], 0.996)])
        ranked = rank_candidates(bpr_config(), [], prior_evidence=evidence)
        ensemble = next(
            row for row in ranked if row.candidate.family == "heterogeneous_ensemble"
        )
        self.assertFalse(ensemble.hard_blocked)
        self.assertNotIn("high_prediction_correlation", ensemble.evidence_reasons)


class RuntimeFilterTests(unittest.TestCase):
    def test_measured_runtime_changes_ranking_deterministically(self) -> None:
        baseline = {
            row.candidate.family: row.score
            for row in rank_candidates(load_config(), [])
        }
        evidence = feasibility_prior([runtime_record("sequence_model", 813.4,
                                                    models=["sequence_deepfm"])])
        first = rank_candidates(load_config(), [], prior_evidence=evidence)
        second = rank_candidates(load_config(), [], prior_evidence=evidence)
        self.assertEqual([row.as_dict() for row in first], [row.as_dict() for row in second])
        measured = next(row for row in first if row.candidate.family == "sequence_model")
        self.assertLess(measured.score, baseline["sequence_model"])
        self.assertEqual(
            next(row for row in first if row.candidate.family == "ranking_objective").score,
            baseline["ranking_objective"],
        )

    def test_missing_runtime_falls_back_to_prior(self) -> None:
        empty = feasibility_prior([])
        with_empty = {
            row.candidate.family: row.score
            for row in rank_candidates(load_config(), [], prior_evidence=empty)
        }
        without = {
            row.candidate.family: row.score
            for row in rank_candidates(load_config(), [])
        }
        self.assertEqual(with_empty, without)


class LeakageFilterTests(unittest.TestCase):
    def test_unsafe_leakage_is_hard_in_both_passes(self) -> None:
        evidence = feasibility_prior([
            leakage_record("user_tab_cross", "unsafe", leakage_safe=False, strict_past=False),
        ])
        ranked = rank_candidates(load_config(), [], prior_evidence=evidence)
        blocked = row_for(ranked, user_tab_cross=True)
        neighbor = row_for(ranked, user_author_cross=True)
        self.assertTrue(blocked.hard_blocked)
        self.assertFalse(blocked.soft_stopped)
        self.assertIn("unsafe_leakage", blocked.evidence_reasons)
        self.assertNotIn(blocked, admissible_candidates(ranked, relax_soft=False))
        self.assertNotIn(blocked, admissible_candidates(ranked, relax_soft=True))
        self.assertFalse(neighbor.hard_blocked)
        self.assertTrue(neighbor.soft_stopped)
        self.assertEqual(neighbor.evidence_reasons, ("missing_leakage_evidence",))
        history = exhaust(load_config(), keep={("user_tab_cross",)})
        with self.assertRaises(StopIteration):
            DeterministicResearcher(prior_evidence=evidence).propose(load_config(), history)
        with self.assertRaises(StopIteration):
            AutonomousExperimentPlanner(prior_evidence=evidence).select(
                load_config(), history,
            )
        self.assertEqual(
            choose_ranked(rank_candidates(load_config(), history, prior_evidence=evidence)),
            [],
        )

    def test_no_leakage_record_is_soft(self) -> None:
        ranked = rank_candidates(load_config(), [])
        row = row_for(ranked, global_context=True)
        self.assertFalse(row.hard_blocked)
        self.assertTrue(row.soft_stopped)
        self.assertEqual(row.evidence_reasons, ("missing_leakage_evidence",))
        self.assertNotIn(row, admissible_candidates(ranked, relax_soft=False))
        self.assertIn(row, admissible_candidates(ranked, relax_soft=True))
        for item in ranked:
            if item.candidate.action_type in {ActionType.TRY_MODEL, ActionType.TUNE}:
                self.assertNotIn("missing_leakage_evidence", item.evidence_reasons)
        history = exhaust(load_config(), keep={("global_context",)})
        live, selected, replayed = live_replay(load_config(), history, None)
        self.assertEqual(selected.changes, live.changes)
        self.assertEqual(replayed[0].candidate.changes, live.changes)
        self.assertEqual(live.changes, {"global_context": True})

    def test_uncertain_leakage_is_soft_and_relaxable(self) -> None:
        evidence = feasibility_prior([
            leakage_record("global_context", "uncertain"),
        ])
        ranked = rank_candidates(load_config(), [], prior_evidence=evidence)
        row = row_for(ranked, global_context=True)
        self.assertFalse(row.hard_blocked)
        self.assertTrue(row.soft_stopped)
        self.assertEqual(row.evidence_reasons, ("uncertain_leakage_evidence",))
        history = exhaust(load_config(), keep={("global_context",)})
        live, selected, replayed = live_replay(load_config(), history, evidence)
        self.assertEqual(selected.changes, live.changes)
        self.assertEqual(replayed[0].candidate.changes, live.changes)
        self.assertEqual(live.changes, {"global_context": True})

    def test_explicit_safe_strict_past_is_executable_in_preferred_pass(self) -> None:
        evidence = feasibility_prior(safe_leakage("global_context"))
        ranked = rank_candidates(load_config(), [], prior_evidence=evidence)
        cleared = row_for(ranked, global_context=True)
        self.assertFalse(cleared.hard_blocked)
        self.assertFalse(cleared.soft_stopped)
        self.assertEqual(cleared.evidence_reasons, ())
        self.assertIn(cleared, admissible_candidates(ranked, relax_soft=False))
        live, selected, replayed = live_replay(load_config(), [], evidence)
        self.assertEqual(selected.changes, live.changes)
        self.assertEqual(replayed[0].candidate.changes, live.changes)
        planner = AutonomousExperimentPlanner(prior_evidence=evidence)
        planner.select(load_config(), [])
        self.assertEqual(planner.last_selection["selection_pass"], "preferred")

    def test_safe_without_strict_past_stays_soft(self) -> None:
        evidence = feasibility_prior([
            leakage_record("global_context", "safe", leakage_safe=True, strict_past=None),
        ])
        row = row_for(
            rank_candidates(load_config(), [], prior_evidence=evidence),
            global_context=True,
        )
        self.assertFalse(row.hard_blocked)
        self.assertTrue(row.soft_stopped)
        self.assertEqual(row.evidence_reasons, ("uncertain_leakage_evidence",))


class IsolationAndParityTests(unittest.TestCase):
    def test_duplicates_remain_blocked_with_feasibility_present(self) -> None:
        bpr = bpr_config()
        evidence = feasibility_prior([
            coverage_record("global_context", 0.003, eligible_rows=4),
            runtime_record("sequence_model", 813.4, models=["sequence_deepfm"]),
        ])
        history = [tried_row(bpr, {"training_objective": "bpr", "learning_rate": 0.0003})]
        ranked = rank_candidates(load_config(), history, prior_evidence=evidence)
        keys = {
            experiment_key(apply_changes(load_config(), row.candidate.changes))
            for row in ranked
        }
        self.assertNotIn(experiment_key(bpr), keys)
        for pass_rows in (
            admissible_candidates(ranked, relax_soft=False),
            admissible_candidates(ranked, relax_soft=True),
        ):
            for row in pass_rows:
                self.assertNotEqual(
                    experiment_key(apply_changes(load_config(), row.candidate.changes)),
                    experiment_key(bpr),
                )

    def test_live_and_replay_agree_on_soft_coverage_fallback(self) -> None:
        evidence = feasibility_prior([coverage_record("global_context", 0.003, eligible_rows=4)])
        history = exhaust(load_config(), keep={("global_context",)})
        live = DeterministicResearcher(prior_evidence=evidence).propose(load_config(), history)
        selected = AutonomousExperimentPlanner(prior_evidence=evidence).select(
            load_config(), history,
        )
        replayed = choose_ranked(rank_candidates(
            load_config(), history, prior_evidence=evidence,
        ))
        self.assertEqual(selected.changes, live.changes)
        self.assertEqual(replayed[0].candidate.changes, live.changes)
        self.assertEqual(live.changes, {"global_context": True})

    def test_markdown_is_never_consumed(self) -> None:
        opened: list[str] = []
        original = Path.read_text

        def guarded(self, *args, **kwargs):
            opened.append(self.name)
            if self.name in {"TRY.md", "AGENT-TRY.md"}:
                raise AssertionError("planner must not read markdown research logs")
            return original(self, *args, **kwargs)

        evidence = feasibility_prior([coverage_record("global_context", 0.003, eligible_rows=4)])
        with patch.object(Path, "read_text", guarded):
            DeterministicResearcher(prior_evidence=evidence).propose(load_config(), [])
        self.assertNotIn("TRY.md", opened)
        self.assertNotIn("AGENT-TRY.md", opened)


if __name__ == "__main__":
    unittest.main()
