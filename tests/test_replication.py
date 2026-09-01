from __future__ import annotations

import unittest

from src.techjam_agent.replication import critique_replications, summarize_objective_comparison


class ReplicationSummaryTests(unittest.TestCase):
    def test_paired_validation_summary(self) -> None:
        summary = summarize_objective_comparison([
            {"seed": 0, "bce": {"primary": 0.60}, "bpr": {"primary": 0.61}},
            {"seed": 1, "bce": {"primary": 0.59}, "bpr": {"primary": 0.605}},
        ])
        self.assertEqual(summary["split"], "validation")
        self.assertEqual(summary["seeds_improved"], 2)
        self.assertEqual(summary["seeds_total"], 2)
        self.assertAlmostEqual(summary["paired_delta"]["mean"], 0.0125)
        critique = critique_replications(summary)
        self.assertEqual(critique["verdict"], "consistent_improvement")
        self.assertIn("2/2", critique["observation"])

    def test_all_seed_small_gain_is_consistent_not_inconsistent(self) -> None:
        summary = summarize_objective_comparison([
            {"seed": 0, "bce": {"primary": 0.601}, "bpr": {"primary": 0.602}},
            {"seed": 1, "bce": {"primary": 0.600}, "bpr": {"primary": 0.6015}},
        ])
        self.assertEqual(
            critique_replications(summary)["verdict"], "consistent_small_improvement"
        )

    def test_single_seed_is_not_claimed_as_stable(self) -> None:
        summary = summarize_objective_comparison([
            {"seed": 0, "bce": {"primary": 0.601}, "bpr": {"primary": 0.604}},
        ])
        self.assertEqual(critique_replications(summary)["verdict"], "single_seed")

    def test_failed_pair_is_excluded(self) -> None:
        summary = summarize_objective_comparison([
            {"seed": 0, "bce": {"primary": 0.601}, "bpr": None, "error": "failed"},
        ])
        self.assertEqual(summary["seeds_total"], 0)
        self.assertEqual(len(summary["failures"]), 1)


if __name__ == "__main__":
    unittest.main()
