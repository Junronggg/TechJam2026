"""Milestone A1: closed action types and hard versus soft evidence.

Fake history and structured policies only. No training, no API, no markdown parsing.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import (
    FEATURE_KEYS,
    FEATURE_SCHEMA_VERSION,
    MODELS,
    apply_changes,
    experiment_key,
    validate_config,
)
from techjam_agent.experiment_planner import (
    ACTION_TYPES,
    ActionType,
    AutonomousExperimentPlanner,
    admissible_candidates,
    generate_candidates,
    rank_candidates,
)


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def generated_policies() -> dict:
    return json.loads(
        (ROOT / "configs" / "generated_family_policies.json").read_text(encoding="utf-8")
    )


def bpr_config() -> dict:
    return apply_changes(load_config(), {
        "training_objective": "bpr", "learning_rate": 0.0003,
    })


def ensemble_config() -> dict:
    return apply_changes(bpr_config(), {
        "model": "ensemble", "training_objective": "hybrid",
        "ensemble_deepfm_weight": 0.4,
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
    """Mark every legal candidate as tried except those whose sorted change keys match keep."""
    history = list(rows or [])
    keep = keep or set()
    for candidate in generate_candidates(best):
        key = tuple(sorted(candidate.changes))
        if key in keep:
            continue
        history.append(tried_row(apply_changes(best, candidate.changes), candidate.changes))
    return history


def weak_family_history(family: str, count: int = 2) -> list[dict]:
    rows = []
    for iteration in range(1, count + 1):
        rows.append({
            "iteration": iteration,
            "config": copy.deepcopy(load_config()),
            "changes": {"learning_rate": 0.002},
            "delta_from_parent": 0.00001,
            "candidate_selection": {"selected_family": family},
            "critique": {"verdict": "noise"},
            "diagnostics": {},
        })
    return rows


def robust_stop(family: str, applies_to: dict) -> dict:
    return {
        "family_policies": [{
            "family": family,
            "policy": "stop_direction",
            "scientific_verdict": "REJECTED",
            "confidence": 0.9,
            "applies_to": applies_to,
            "created_from": [{
                "source_id": "rolling_v1",
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


class ActionTypeContractTests(unittest.TestCase):
    def test_every_candidate_has_a_valid_action_type(self) -> None:
        parents = [
            load_config(),
            bpr_config(),
            ensemble_config(),
            apply_changes(load_config(), {"model": "lightgbm", "training_objective": "bce"}),
            apply_changes(load_config(), {"model": "dcnv2", "training_objective": "bce"}),
        ]
        seen: set[str] = set()
        for parent in parents:
            for candidate in generate_candidates(parent):
                self.assertIn(candidate.action_type, ACTION_TYPES)
                seen.add(str(candidate.action_type))
        self.assertEqual(seen, {item.value for item in ActionType})

    def test_existing_generators_map_to_the_declared_verbs(self) -> None:
        by_family = {row.family: row.action_type for row in generate_candidates(load_config())}
        self.assertEqual(by_family["ranking_objective"], ActionType.TRY_MODEL)
        self.assertEqual(by_family["multitask"], ActionType.TRY_MODEL)
        self.assertEqual(by_family["pairwise_multitask"], ActionType.TRY_MODEL)
        self.assertEqual(by_family["cross_network"], ActionType.TRY_MODEL)
        self.assertEqual(by_family["tree_model"], ActionType.TRY_MODEL)
        self.assertEqual(by_family["sequence_model"], ActionType.TRY_SEQUENCE)
        self.assertEqual(by_family["global_context"], ActionType.TRY_FEATURE)
        self.assertEqual(by_family["optimization"], ActionType.TUNE)
        ensemble = next(
            row for row in generate_candidates(bpr_config())
            if row.family == "heterogeneous_ensemble"
        )
        self.assertEqual(ensemble.action_type, ActionType.TRY_ENSEMBLE)

    def test_every_emitted_change_passes_apply_changes(self) -> None:
        for parent in (load_config(), bpr_config(), ensemble_config()):
            for candidate in generate_candidates(parent):
                validate_config(apply_changes(parent, candidate.changes))

    def test_no_unknown_model_or_feature_can_be_emitted(self) -> None:
        for candidate in generate_candidates(load_config()):
            if "model" in candidate.changes:
                self.assertIn(candidate.changes["model"], MODELS)
            for key in candidate.changes:
                if key in FEATURE_KEYS or key in {"model", "training_objective"}:
                    continue
                self.assertNotIn(key, {"bst", "transformer", "invented_feature"})
            self.assertTrue(
                set(candidate.changes).issubset(
                    set(FEATURE_KEYS) | {"model", "training_objective",
                                         "learning_rate", "embedding_dim", "l2",
                                         "batch_size", "ensemble_deepfm_weight"}
                )
            )

    def test_closed_action_set_excludes_multi_training_verbs(self) -> None:
        self.assertNotIn("RUN_PLACEBO", ACTION_TYPES)
        self.assertNotIn("RUN_ROLLING", ACTION_TYPES)
        self.assertNotIn("RUN_MULTI_SEED", ACTION_TYPES)


class HardSoftSelectionTests(unittest.TestCase):
    def test_robust_negative_evidence_stays_blocked_in_both_passes(self) -> None:
        evidence = robust_stop("temporal_counts", {
            "task": "long_view",
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "models": ["ensemble"],
            "features": {
                "user_recent_3d_activity": True,
                "item_recent_3d_exposure": True,
            },
            "hyperparameters": {"ensemble_deepfm_weight": 0.4},
        })
        parent = apply_changes(ensemble_config(), {"user_recent_3d_activity": True})
        ranked = rank_candidates(parent, [], prior_evidence=evidence)
        blocked = next(
            row for row in ranked
            if row.candidate.changes == {"item_recent_3d_exposure": True}
        )
        self.assertTrue(blocked.hard_blocked)
        self.assertFalse(blocked.soft_stopped)
        preferred = admissible_candidates(ranked, relax_soft=False)
        relaxed = admissible_candidates(ranked, relax_soft=True)
        self.assertNotIn(blocked, preferred)
        self.assertNotIn(blocked, relaxed)
        self.assertTrue(all(
            not row.hard_blocked and not row.soft_stopped for row in preferred
        ))

    def test_soft_evidence_is_skipped_in_pass_one_and_recoverable_in_pass_two(self) -> None:
        history = exhaust(
            load_config(),
            weak_family_history("global_context"),
            keep={("global_context",)},
        )
        ranked = rank_candidates(load_config(), history)
        soft = next(row for row in ranked if row.candidate.family == "global_context")
        self.assertFalse(soft.hard_blocked)
        self.assertTrue(soft.soft_stopped)
        self.assertEqual(admissible_candidates(ranked, relax_soft=False), [])
        recovered = admissible_candidates(ranked, relax_soft=True)
        self.assertEqual(recovered[0].candidate.family, "global_context")
        planner = AutonomousExperimentPlanner()
        selected = planner.select(load_config(), history)
        self.assertEqual(selected.changes, {"global_context": True})
        self.assertEqual(planner.last_selection["selection_pass"], "relaxed")

    def test_duplicate_stays_blocked_in_both_passes(self) -> None:
        bpr = apply_changes(load_config(), {
            "training_objective": "bpr", "learning_rate": 0.0003,
        })
        history = [tried_row(bpr, {"training_objective": "bpr", "learning_rate": 0.0003})]
        ranked = rank_candidates(load_config(), history)
        keys = {experiment_key(apply_changes(load_config(), row.candidate.changes))
                for row in ranked}
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

    def test_stop_iteration_only_when_remaining_candidates_are_hard_blocked(self) -> None:
        evidence = robust_stop("temporal_counts", {
            "task": "long_view",
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "models": ["ensemble"],
            "features": {
                "user_recent_3d_activity": True,
                "item_recent_3d_exposure": True,
            },
            "hyperparameters": {"ensemble_deepfm_weight": 0.4},
        })
        parent = apply_changes(ensemble_config(), {"user_recent_3d_activity": True})
        history = exhaust(parent, keep={("item_recent_3d_exposure",)})
        ranked = rank_candidates(parent, history, prior_evidence=evidence)
        self.assertTrue(all(row.hard_blocked for row in ranked))
        self.assertEqual(choose_or_empty(ranked), [])
        planner = AutonomousExperimentPlanner(prior_evidence=evidence)
        with self.assertRaises(StopIteration):
            planner.select(parent, history)

    def test_soft_stop_does_not_raise_when_it_is_the_only_remaining_direction(self) -> None:
        history = exhaust(
            load_config(),
            weak_family_history("optimization"),
            keep={("learning_rate",), ("embedding_dim",), ("l2",), ("batch_size",)},
        )
        planner = AutonomousExperimentPlanner()
        selected = planner.select(load_config(), history)
        self.assertEqual(selected.action_type, ActionType.TUNE)
        self.assertEqual(planner.last_selection["selection_pass"], "relaxed")

    def test_scoped_temporal_policy_does_not_over_block_individual_features(self) -> None:
        evidence = generated_policies()
        fm_user = next(
            row for row in rank_candidates(load_config(), [], prior_evidence=evidence)
            if row.candidate.changes == {"user_recent_3d_activity": True}
        )
        self.assertFalse(fm_user.hard_blocked)
        self.assertIn("missing_leakage_evidence", fm_user.evidence_reasons)
        ensemble_user = next(
            row for row in rank_candidates(
                ensemble_config(), [], prior_evidence=evidence,
            ) if row.candidate.changes == {"user_recent_3d_activity": True}
        )
        self.assertFalse(ensemble_user.hard_blocked)
        self.assertIn("missing_leakage_evidence", ensemble_user.evidence_reasons)
        parent = apply_changes(ensemble_config(), {"user_recent_3d_activity": True})
        combined = next(
            row for row in rank_candidates(parent, [], prior_evidence=evidence)
            if row.candidate.changes == {"item_recent_3d_exposure": True}
        )
        self.assertTrue(combined.hard_blocked)

    def test_identical_inputs_return_identical_ranking_and_selection(self) -> None:
        history = weak_family_history("global_context")
        evidence = generated_policies()
        rankings = [
            [row.as_dict() for row in rank_candidates(
                load_config(), history, prior_evidence=evidence,
            )]
            for _ in range(3)
        ]
        self.assertEqual(rankings[0], rankings[1])
        self.assertEqual(rankings[1], rankings[2])
        planner = AutonomousExperimentPlanner(prior_evidence=evidence)
        selected = [planner.select(load_config(), history).changes for _ in range(3)]
        self.assertEqual(selected[0], selected[1])
        self.assertEqual(selected[1], selected[2])

    def test_memory_changed_choice_and_counterfactuals_remain_intact(self) -> None:
        history = [{
            "iteration": 1,
            "status": "success",
            "decision": "KEEP",
            "delta_from_parent": 0.002,
            "candidate_selection": {"selected_family": "ranking_objective"},
            "critique": {"verdict": "promote"},
            "diagnostics": {},
        }]
        planner = AutonomousExperimentPlanner(memory_mode="no_memory")
        planner.select(load_config(), history)
        selection = planner.last_selection
        self.assertIsNotNone(selection)
        self.assertEqual(
            set(selection["counterfactual_choices"]),
            {"no_memory", "raw_history", "distilled_patterns"},
        )
        self.assertIsInstance(selection["memory_changed_choice"], bool)
        self.assertEqual(selection["memory_mode"], "no_memory")
        self.assertIsNone(selection["retrieved_pattern"])
        for choice in selection["counterfactual_choices"].values():
            if choice is None:
                continue
            self.assertIn("family", choice)
            self.assertIn("changes", choice)
            self.assertIn("differs_from_selected", choice)

    def test_non_robust_stop_direction_is_soft_not_hard(self) -> None:
        evidence = {"family_policies": [{
            "family": "pairwise_multitask",
            "policy": "stop_direction",
            "scientific_verdict": "REJECTED",
            "applies_to": {"models": ["multitask_deepfm"]},
            "created_from": [{
                "source_id": "single_v1",
                "kind": "single_delta",
                "result": {"signal": "negative", "delta": -0.001, "robust": False},
            }],
        }]}
        row = next(
            item for item in rank_candidates(
                load_config(), [], prior_evidence=evidence,
            ) if item.candidate.family == "pairwise_multitask"
        )
        self.assertFalse(row.hard_blocked)
        self.assertTrue(row.soft_stopped)


def choose_or_empty(ranked):
    from techjam_agent.experiment_planner import choose_ranked
    return choose_ranked(ranked)


if __name__ == "__main__":
    unittest.main()
