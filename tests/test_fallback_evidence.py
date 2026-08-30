"""Evidence-driven deterministic fallback. Fake history only: no network, no LLM, no training."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import FEATURE_KEYS, apply_changes, experiment_key, validate_config
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


def first_free_feature() -> str:
    return next(key for key in FEATURE_KEYS
                if key not in ("continuous_history_stats", "user_tab_long_view_rate"))


def exhaust(best: dict, rows: list[dict], *, keep: str | None = None) -> list[dict]:
    """Mark every legal candidate as already tried, except those of one model."""
    history = list(rows)
    iteration = 100
    for changes, _, _ in DeterministicResearcher()._ordered_candidates(best):
        try:
            candidate = apply_changes(best, changes)
        except ValueError:
            continue
        if keep is not None and candidate["model"] == keep:
            continue
        history.append(record(iteration, candidate, changes, verdict="noise"))
        iteration += 1
    return history


class HardBlockTests(unittest.TestCase):
    def lightgbm_failures(self, count: int, error: dict) -> list[dict]:
        variants = ({"model": "lightgbm"},
                    {"model": "lightgbm", "user_tab_long_view_rate": True},
                    {"model": "lightgbm", "continuous_history_stats": True})
        rows = [baseline_row(),
                record(1, bpr_config(), {"training_objective": "bpr",
                                         "learning_rate": 0.0003}, verdict="noise")]
        for index, changes in enumerate(variants[:count], start=2):
            rows.append(record(index, apply_changes(load_config(), changes), changes,
                               verdict="failed", primary=None, error=error))
        return rows

    def test_structural_failure_skips_further_variants_of_that_model(self) -> None:
        history = self.lightgbm_failures(1, MISSING_LIGHTGBM)
        proposal = DeterministicResearcher().propose(load_config(), history)
        self.assertNotEqual(proposal.changes.get("model"), "lightgbm")
        self.assertEqual(proposal.changes, {first_free_feature(): True})
        self.assertIn("lightgbm cannot run here", proposal.reason)

    def test_structurally_failed_model_is_never_retried_in_the_second_pass(self) -> None:
        """Relaxing soft evidence must never resurrect a model that cannot run."""
        best = load_config()
        history = exhaust(best, self.lightgbm_failures(1, MISSING_LIGHTGBM),
                          keep="lightgbm")
        self.assertEqual(evidence_directions(history).blocked_models,
                         frozenset({"lightgbm"}))
        with self.assertRaises(StopIteration):
            DeterministicResearcher().propose(best, history)

    def test_only_hard_blocked_or_duplicate_candidates_raise_stop_iteration(self) -> None:
        best = load_config()
        clean = exhaust(best, [baseline_row()])
        with self.assertRaises(StopIteration):
            DeterministicResearcher().propose(best, clean)

    def test_one_generic_timeout_does_not_block_the_whole_model(self) -> None:
        history = self.lightgbm_failures(1, TIMEOUT_ERROR)
        self.assertFalse(evidence_directions(history))
        proposal = DeterministicResearcher().propose(load_config(), history)
        self.assertEqual(proposal.changes,
                         {"model": "lightgbm", "user_tab_long_view_rate": True})
        self.assertNotIn("Evidence skipped", proposal.reason)

    def test_two_consistent_timeouts_softly_disfavor_the_model(self) -> None:
        history = self.lightgbm_failures(2, TIMEOUT_ERROR)
        directions = evidence_directions(history)
        self.assertEqual(directions.soft_models, frozenset({"lightgbm"}))
        self.assertEqual(directions.blocked_models, frozenset())
        proposal = DeterministicResearcher().propose(load_config(), history)
        self.assertNotEqual(proposal.changes.get("model"), "lightgbm")
        self.assertEqual(proposal.changes, {first_free_feature(): True})
        self.assertIn("failed repeatedly", proposal.reason)


class SoftEvidenceTests(unittest.TestCase):
    def rejected_ensemble_history(self) -> list[dict]:
        changes = {"model": "ensemble", "training_objective": "hybrid",
                   "ensemble_deepfm_weight": 0.4}
        return [
            baseline_row(),
            record(1, bpr_config(), {"training_objective": "bpr",
                                     "learning_rate": 0.0003}, verdict="promote"),
            record(2, apply_changes(bpr_config(), changes), changes,
                   verdict="reject", primary=0.55),
        ]

    def test_rejected_mechanism_is_skipped_while_alternatives_exist(self) -> None:
        proposal = DeterministicResearcher().propose(bpr_config(),
                                                     self.rejected_ensemble_history())
        self.assertNotEqual(proposal.changes.get("model"), "ensemble")
        self.assertEqual(proposal.changes,
                         {"training_objective": "hybrid", "hybrid_bpr_weight": 0.75})
        self.assertIn("ensemble+hybrid", proposal.reason)

    def test_rejected_mechanism_is_reconsidered_when_nothing_else_remains(self) -> None:
        best = bpr_config()
        history = exhaust(best, self.rejected_ensemble_history(), keep="ensemble")
        proposal = DeterministicResearcher().propose(best, history)
        self.assertEqual(proposal.changes.get("model"), "ensemble")
        self.assertEqual(proposal.changes.get("ensemble_deepfm_weight"), 0.3)

    def test_softly_disfavored_model_is_reconsidered_when_nothing_else_remains(self) -> None:
        best = load_config()
        failure = record(1, apply_changes(best, {"model": "lightgbm"}),
                         {"model": "lightgbm"}, verdict="failed", primary=None,
                         error=TIMEOUT_ERROR)
        second = record(2, apply_changes(best, {"model": "lightgbm",
                                                "user_tab_long_view_rate": True}),
                        {"model": "lightgbm", "user_tab_long_view_rate": True},
                        verdict="failed", primary=None, error=TIMEOUT_ERROR)
        history = exhaust(best, [baseline_row(), failure, second], keep="lightgbm")
        self.assertEqual(evidence_directions(history).soft_models,
                         frozenset({"lightgbm"}))
        proposal = DeterministicResearcher().propose(best, history)
        self.assertEqual(proposal.changes,
                         {"model": "lightgbm", "continuous_history_stats": True})

    def test_unrelated_directions_remain_available(self) -> None:
        directions = evidence_directions(self.rejected_ensemble_history())
        for config in (load_config(), bpr_config(),
                       apply_changes(load_config(), {"model": "lightgbm"})):
            self.assertIsNone(directions.hard_block_for(config))
            self.assertIsNone(directions.soft_reason_for(config))

    def test_exact_duplicates_stay_blocked_through_both_passes(self) -> None:
        best = bpr_config()
        history = exhaust(best, self.rejected_ensemble_history(), keep="ensemble")
        tried = {experiment_key(row["config"]) for row in history}
        proposal = DeterministicResearcher().propose(best, history)
        candidate = apply_changes(best, proposal.changes)
        self.assertNotIn(experiment_key(candidate), tried)


class IllegalCandidateTests(unittest.TestCase):
    """The candidate order is model-agnostic, so some combinations cannot validate."""

    def test_lightgbm_best_generates_fm_only_feature_toggles(self) -> None:
        best = apply_changes(load_config(), {"model": "lightgbm"})
        illegal = []
        for changes, _, _ in DeterministicResearcher()._ordered_candidates(best):
            try:
                apply_changes(best, changes)
            except ValueError:
                illegal.append(changes)
        self.assertTrue(illegal, "expected the shared order to offer FM-only features")
        self.assertIn({"user_tab_cross": True}, illegal)

    def test_fm_with_a_cross_feature_generates_illegal_lightgbm_candidates(self) -> None:
        best = apply_changes(load_config(), {"user_tab_cross": True})
        candidate, note = DeterministicResearcher._legal_candidate(best,
                                                                   {"model": "lightgbm"})
        self.assertIsNone(candidate)
        self.assertIn("not a legal configuration", note)
        self.assertIn("FM-family", note)

    def test_illegal_candidates_are_skipped_and_reported(self) -> None:
        best = apply_changes(load_config(), {"user_tab_cross": True})
        bpr_switch = {"training_objective": "bpr", "learning_rate": 0.0003}
        history = [baseline_row(),
                   record(1, apply_changes(best, bpr_switch), bpr_switch,
                          verdict="noise")]
        proposal = DeterministicResearcher().propose(best, history)
        self.assertNotEqual(proposal.changes.get("model"), "lightgbm")
        self.assertEqual(proposal.changes, {first_free_feature(): True})
        self.assertIn("not a legal configuration", proposal.reason)

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

    def test_unexpected_errors_are_not_swallowed(self) -> None:
        def explode(*args, **kwargs):
            raise KeyError("hyperparameters")

        with patch("techjam_agent.proposals.apply_changes", explode):
            with self.assertRaises(KeyError):
                DeterministicResearcher().propose(load_config(), [baseline_row()])


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
        proposal = DeterministicResearcher().propose(load_config(), history)
        self.assertEqual(proposal.changes,
                         {"training_objective": "bpr", "learning_rate": 0.0003})
        self.assertNotIn("Evidence skipped", proposal.reason)

    def test_clean_history_matches_the_pre_evidence_proposal(self) -> None:
        proposal = DeterministicResearcher().propose(load_config(), [baseline_row()])
        self.assertEqual(proposal.changes,
                         {"training_objective": "bpr", "learning_rate": 0.0003})
        self.assertNotIn("Evidence skipped", proposal.reason)

    def test_empty_candidate_order_still_raises_stop_iteration(self) -> None:
        with patch.object(DeterministicResearcher, "_ordered_candidates",
                          return_value=iter(())):
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
        changes = {"model": "ensemble", "training_objective": "hybrid",
                   "ensemble_deepfm_weight": 0.4}
        rejected = record(1, apply_changes(best, changes), changes, verdict="reject",
                          primary=0.55)
        history = exhaust(best, [baseline_row(), rejected], keep="ensemble")
        results = [DeterministicResearcher().propose(best, history).as_dict()
                   for _ in range(3)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

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
