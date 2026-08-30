"""Planner-backed deterministic fallback. Fake history only: no network, no LLM, no training."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes, experiment_key, validate_config
from techjam_agent.experiment_planner import (
    admissible_candidates,
    generate_candidates,
    rank_candidates,
)
from techjam_agent.memory import evidence_directions
from techjam_agent.proposals import DeterministicResearcher

TIMEOUT_ERROR = {"type": "TimeoutError", "message": "experiment exceeded 900s timeout"}
MISSING_LIGHTGBM = {
    "type": "RuntimeError",
    "message": "LightGBM is required: python -m pip install -r requirements.txt",
}


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def bpr_config() -> dict:
    return apply_changes(load_config(), {"training_objective": "bpr",
                                         "learning_rate": 0.0003})


def record(iteration: int, config: dict, changes: dict, *, verdict: str,
           primary: float | None = 0.6015, error: dict | None = None) -> dict:
    """One history row in the shape the Controller writes."""
    metrics = None if primary is None else {"GAUC": primary, "nDCG@5": primary,
                                            "primary": primary}
    return {
        "iteration": iteration,
        "hypothesis": f"row {iteration}",
        "changes": changes,
        "decision": "REJECT" if verdict in ("reject", "failed") else "KEEP",
        "status": "error" if verdict == "failed" else "success",
        "metrics": metrics,
        "config": copy.deepcopy(config),
        "critique": {"verdict": verdict, "delta": 0.0, "next_test": "n/a",
                     "reasons": [verdict], "meaningful_improvement": False},
        "error": error,
    }


def baseline_row() -> dict:
    return record(0, load_config(), {}, verdict="noise")


def exhaust(best: dict, rows: list[dict], *, keep_model: str | None = None,
            keep_families: set[str] | None = None) -> list[dict]:
    """Mark every legal planner candidate as tried, except a kept model or family."""
    history = list(rows)
    iteration = 100
    for candidate in generate_candidates(best):
        if keep_model is not None and candidate.changes.get("model") == keep_model:
            continue
        if keep_families is not None and candidate.family in keep_families:
            continue
        try:
            config = apply_changes(best, candidate.changes)
        except ValueError:
            continue
        history.append(record(iteration, config, candidate.changes, verdict="noise"))
        iteration += 1
    return history


def lightgbm_sibling(feature: str) -> tuple[dict, dict]:
    """A legal LightGBM config that is not the planner's default tree-model action."""
    changes = {"model": "lightgbm", "training_objective": "bce", feature: True}
    config = apply_changes(
        apply_changes(load_config(), {"model": "lightgbm", "training_objective": "bce"}),
        {feature: True},
    )
    return config, changes


def lightgbm_failures(count: int, error: dict) -> list[dict]:
    # Record evidence on sibling LightGBM configs so the default generated
    # tree-model action remains untried and can be ranked as hard or soft.
    variants = ("continuous_history_stats", "user_long_view_rate")
    rows = [baseline_row()]
    for index, feature in enumerate(variants[:count], start=1):
        config, changes = lightgbm_sibling(feature)
        rows.append(record(index, config, changes, verdict="failed",
                           primary=None, error=error))
    return rows


class HardBlockTests(unittest.TestCase):
    def test_structural_failure_skips_further_variants_of_that_model(self) -> None:
        history = lightgbm_failures(1, MISSING_LIGHTGBM)
        self.assertEqual(evidence_directions(history).blocked_models,
                         frozenset({"lightgbm"}))
        proposal = DeterministicResearcher().propose(load_config(), history)
        self.assertNotEqual(proposal.changes.get("model"), "lightgbm")
        validate_config(apply_changes(load_config(), proposal.changes))
        ranked = rank_candidates(load_config(), history)
        for row in ranked:
            if row.candidate.changes.get("model") == "lightgbm":
                self.assertTrue(row.hard_blocked)

    def test_structurally_failed_model_is_never_retried_in_the_second_pass(self) -> None:
        best = load_config()
        history = exhaust(best, lightgbm_failures(1, MISSING_LIGHTGBM),
                          keep_model="lightgbm")
        self.assertEqual(evidence_directions(history).blocked_models,
                         frozenset({"lightgbm"}))
        ranked = rank_candidates(best, history)
        self.assertTrue(any(row.hard_blocked for row in ranked))
        self.assertEqual(admissible_candidates(ranked, relax_soft=True), [])
        with self.assertRaises(StopIteration):
            DeterministicResearcher().propose(best, history)

    def test_only_hard_blocked_or_duplicate_candidates_raise_stop_iteration(self) -> None:
        best = load_config()
        clean = exhaust(best, [baseline_row()])
        with self.assertRaises(StopIteration):
            DeterministicResearcher().propose(best, clean)

    def test_one_generic_timeout_does_not_block_the_whole_model(self) -> None:
        history = lightgbm_failures(1, TIMEOUT_ERROR)
        self.assertFalse(evidence_directions(history))
        ranked = rank_candidates(load_config(), history)
        lightgbm = next(row for row in ranked
                        if row.candidate.changes.get("model") == "lightgbm")
        self.assertFalse(lightgbm.hard_blocked)
        self.assertFalse(lightgbm.soft_stopped)
        proposal = DeterministicResearcher().propose(load_config(), history)
        validate_config(apply_changes(load_config(), proposal.changes))

    def test_two_consistent_timeouts_softly_disfavor_the_model(self) -> None:
        history = lightgbm_failures(2, TIMEOUT_ERROR)
        directions = evidence_directions(history)
        self.assertEqual(directions.soft_models, frozenset({"lightgbm"}))
        self.assertEqual(directions.blocked_models, frozenset())
        ranked = rank_candidates(load_config(), history)
        lightgbm = next(row for row in ranked
                        if row.candidate.changes.get("model") == "lightgbm")
        self.assertFalse(lightgbm.hard_blocked)
        self.assertTrue(lightgbm.soft_stopped)
        self.assertTrue(admissible_candidates(ranked, relax_soft=False))
        proposal = DeterministicResearcher().propose(load_config(), history)
        self.assertNotEqual(proposal.changes.get("model"), "lightgbm")


class SoftEvidenceTests(unittest.TestCase):
    def rejected_ensemble_history(self) -> list[dict]:
        # Reject a sibling ensemble+hybrid weight so the generated 0.4 action
        # stays untried and can be ranked as soft-stopped, not omitted as a duplicate.
        changes = {"model": "ensemble", "training_objective": "hybrid",
                   "ensemble_deepfm_weight": 0.5}
        return [
            baseline_row(),
            record(1, bpr_config(), {"training_objective": "bpr",
                                     "learning_rate": 0.0003}, verdict="promote"),
            record(2, apply_changes(bpr_config(), changes), changes,
                   verdict="reject", primary=0.55),
        ]

    def test_rejected_mechanism_is_skipped_while_alternatives_exist(self) -> None:
        history = self.rejected_ensemble_history()
        proposal = DeterministicResearcher().propose(bpr_config(), history)
        self.assertNotEqual(proposal.changes.get("model"), "ensemble")
        ensemble = next(
            row for row in rank_candidates(bpr_config(), history)
            if row.candidate.family == "heterogeneous_ensemble"
        )
        self.assertTrue(ensemble.soft_stopped)
        self.assertFalse(ensemble.hard_blocked)

    def test_rejected_mechanism_is_reconsidered_when_nothing_else_remains(self) -> None:
        best = bpr_config()
        history = exhaust(best, self.rejected_ensemble_history(),
                          keep_families={"heterogeneous_ensemble"})
        proposal = DeterministicResearcher().propose(best, history)
        self.assertEqual(proposal.changes.get("model"), "ensemble")
        self.assertEqual(proposal.changes.get("training_objective"), "hybrid")

    def test_softly_disfavored_model_is_reconsidered_when_nothing_else_remains(self) -> None:
        best = load_config()
        history = exhaust(best, lightgbm_failures(2, TIMEOUT_ERROR),
                          keep_model="lightgbm")
        self.assertEqual(evidence_directions(history).soft_models,
                         frozenset({"lightgbm"}))
        proposal = DeterministicResearcher().propose(best, history)
        self.assertEqual(proposal.changes.get("model"), "lightgbm")
        validate_config(apply_changes(best, proposal.changes))

    def test_unrelated_directions_remain_available(self) -> None:
        directions = evidence_directions(self.rejected_ensemble_history())
        for config in (load_config(), bpr_config(),
                       apply_changes(load_config(), {"model": "lightgbm"})):
            self.assertIsNone(directions.hard_block_for(config))
            self.assertIsNone(directions.soft_reason_for(config))
        ranked = rank_candidates(load_config(), self.rejected_ensemble_history())
        self.assertTrue(any(
            not row.hard_blocked and not row.soft_stopped
            and row.candidate.family != "heterogeneous_ensemble"
            for row in ranked
        ))

    def test_exact_duplicates_stay_blocked_through_both_passes(self) -> None:
        best = bpr_config()
        history = exhaust(best, self.rejected_ensemble_history(),
                          keep_families={"heterogeneous_ensemble"})
        tried = {experiment_key(row["config"]) for row in history
                 if isinstance(row.get("config"), dict)}
        proposal = DeterministicResearcher().propose(best, history)
        candidate = apply_changes(best, proposal.changes)
        self.assertNotIn(experiment_key(candidate), tried)
        ranked = rank_candidates(best, history)
        for row in admissible_candidates(ranked, relax_soft=False) + \
                admissible_candidates(ranked, relax_soft=True):
            self.assertNotIn(
                experiment_key(apply_changes(best, row.candidate.changes)), tried,
            )


class LegalCandidateTests(unittest.TestCase):
    def test_planner_does_not_emit_illegal_lightgbm_feature_toggles(self) -> None:
        best = apply_changes(load_config(), {"model": "lightgbm"})
        for candidate in generate_candidates(best):
            validate_config(apply_changes(best, candidate.changes))
            self.assertNotIn("user_tab_cross", candidate.changes)

    def test_planner_does_not_switch_to_lightgbm_when_cross_features_are_on(self) -> None:
        best = apply_changes(load_config(), {"user_tab_cross": True})
        for candidate in generate_candidates(best):
            validate_config(apply_changes(best, candidate.changes))
            self.assertNotEqual(candidate.changes.get("model"), "lightgbm")

    def test_every_returned_proposal_validates(self) -> None:
        bests = [
            load_config(),
            bpr_config(),
            apply_changes(load_config(), {"model": "lightgbm"}),
            apply_changes(load_config(), {"user_tab_cross": True}),
            apply_changes(bpr_config(), {"model": "ensemble",
                                         "training_objective": "hybrid"}),
            apply_changes(load_config(), {"model": "deepfm"}),
        ]
        for best in bests:
            with self.subTest(model=best["model"], objective=best["training_objective"]):
                proposal = DeterministicResearcher().propose(best, [baseline_row()])
                validate_config(apply_changes(best, proposal.changes))

    def test_robust_scoped_negative_evidence_stays_hard_blocked(self) -> None:
        evidence = json.loads(
            (ROOT / "configs" / "generated_family_policies.json").read_text(
                encoding="utf-8"
            )
        )
        parent = apply_changes(
            apply_changes(bpr_config(), {
                "model": "ensemble", "training_objective": "hybrid",
            }),
            {"user_recent_3d_activity": True},
        )
        ranked = rank_candidates(parent, [baseline_row()], prior_evidence=evidence)
        combined = next(
            row for row in ranked
            if row.candidate.changes == {"item_recent_3d_exposure": True}
        )
        self.assertTrue(combined.hard_blocked)
        self.assertNotIn(combined, admissible_candidates(ranked, relax_soft=True))


class TwoPassTests(unittest.TestCase):
    def mixed_evidence_history(self) -> list[dict]:
        """fm is softly disfavored, lightgbm is hard blocked."""
        rows = [baseline_row()]
        for index in (1, 2):
            rows.append(record(index, load_config(), {"learning_rate": 0.002},
                               verdict="failed", primary=None, error=TIMEOUT_ERROR))
        rows.append(record(3, apply_changes(load_config(), {"model": "lightgbm"}),
                           {"model": "lightgbm"}, verdict="failed", primary=None,
                           error=MISSING_LIGHTGBM))
        return rows

    def test_soft_evidence_relaxes_but_hard_blocks_do_not(self) -> None:
        history = self.mixed_evidence_history()
        directions = evidence_directions(history)
        self.assertEqual(directions.soft_models, frozenset({"fm"}))
        self.assertEqual(directions.blocked_models, frozenset({"lightgbm"}))
        ranked = rank_candidates(load_config(), history)
        preferred = admissible_candidates(ranked, relax_soft=False)
        self.assertTrue(preferred)
        self.assertFalse(any(
            row.candidate.changes.get("model") == "lightgbm" for row in preferred
        ))
        proposal = DeterministicResearcher().propose(load_config(), history)
        self.assertNotEqual(proposal.changes.get("model"), "lightgbm")
        validate_config(apply_changes(load_config(), proposal.changes))

    def test_clean_history_matches_the_pre_evidence_proposal(self) -> None:
        proposal = DeterministicResearcher().propose(load_config(), [baseline_row()])
        self.assertEqual(proposal.changes,
                         {"training_objective": "bpr", "learning_rate": 0.0003})

    def test_empty_legal_space_still_raises_stop_iteration(self) -> None:
        with patch(
            "techjam_agent.experiment_planner.generate_candidates",
            return_value=[],
        ):
            with self.assertRaises(StopIteration):
                DeterministicResearcher().propose(load_config(), [baseline_row()])


class DeterminismAndIsolationTests(unittest.TestCase):
    def history(self) -> list[dict]:
        changes = {"model": "lightgbm"}
        return [
            baseline_row(),
            record(1, apply_changes(load_config(), changes), changes, verdict="failed",
                   primary=None, error=MISSING_LIGHTGBM),
        ]

    def test_identical_history_produces_identical_proposals(self) -> None:
        history = self.history()
        results = [DeterministicResearcher().propose(load_config(), history).as_dict()
                   for _ in range(3)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_relaxed_pass_is_also_deterministic(self) -> None:
        best = bpr_config()
        history = exhaust(best, self.rejected_soft_history(),
                          keep_families={"heterogeneous_ensemble"})
        results = [DeterministicResearcher().propose(best, history).as_dict()
                   for _ in range(3)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def rejected_soft_history(self) -> list[dict]:
        changes = {"model": "ensemble", "training_objective": "hybrid",
                   "ensemble_deepfm_weight": 0.5}
        return [
            baseline_row(),
            record(1, apply_changes(bpr_config(), changes), changes,
                   verdict="reject", primary=0.55),
        ]

    def test_dirty_history_rows_do_not_crash_the_planner(self) -> None:
        history = [None, "junk", {}, {"config": None}, {"config": {"model": 1}},
                   {"config": load_config(), "critique": "junk", "error": "junk"},
                   *self.history()]
        proposal = DeterministicResearcher().propose(load_config(), history)
        self.assertTrue(proposal.changes)
        self.assertEqual(proposal.source, "deterministic")

    def test_no_network_or_llm_is_used(self) -> None:
        def explode(*args, **kwargs):
            raise AssertionError("the deterministic fallback must not open a connection")

        with patch("urllib.request.urlopen", explode):
            proposal = DeterministicResearcher().propose(load_config(), self.history())
        self.assertEqual(proposal.source, "deterministic")
        self.assertEqual(proposal.token_usage,
                         {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.assertEqual(proposal.llm_attempts, 0)


if __name__ == "__main__":
    unittest.main()
