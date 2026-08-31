from __future__ import annotations

import json
import unittest
from pathlib import Path

from techjam_agent.experiment_planner import rank_candidates
from techjam_agent.proposals import build_planner_prompt
from techjam_agent.skills import default_skill_registry


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads(
        (ROOT / "configs" / "experiment.json").read_text(encoding="utf-8")
    )


class SkillRegistryTests(unittest.TestCase):
    def test_minimal_registry_contains_ten_available_safe_skills(self) -> None:
        rows = default_skill_registry().catalog()
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {row["category"] for row in rows},
            {"discovery", "training", "evidence", "research_memory"},
        )
        self.assertTrue(all(row["status"] == "available" for row in rows))
        self.assertTrue(all(row["test_labels_allowed"] is False for row in rows))

    def test_candidates_bind_to_registered_primary_and_evidence_skills(self) -> None:
        registry = default_skill_registry()
        rows = rank_candidates(load_config(), [])
        self.assertTrue(rows)
        for row in rows:
            candidate = row.candidate
            registry.require(candidate.skill_id)
            for skill_id in candidate.required_confirmation:
                registry.require(skill_id)
        bpr = next(
            row.candidate for row in rows
            if row.candidate.family == "ranking_objective"
        )
        self.assertEqual(bpr.skill_id, "train_model")
        self.assertEqual(
            bpr.required_confirmation, ("run_rolling", "run_paired_seeds")
        )

    def test_prompt_separates_principles_skills_and_controller_guards(self) -> None:
        prompt = build_planner_prompt(load_config(), [])
        self.assertGreaterEqual(len(prompt["research_principles"]), 8)
        self.assertEqual(len(prompt["skill_catalog"]), 10)
        self.assertFalse(
            prompt["controller_guards"]["test_labels_allowed_during_research"]
        )
        self.assertFalse(prompt["capability_policy"]["capability_builder_enabled"])
        self.assertIn("train_graph", prompt["capability_policy"]["known_gaps"])
        for row in prompt["candidate_ranking"]:
            self.assertIn("skill_id", row)
            self.assertIn("required_confirmation", row)
            self.assertIn("risk", row)

    def test_deterministic_selection_writes_audited_decision_record(self) -> None:
        from techjam_agent.proposals import DeterministicResearcher

        researcher = DeterministicResearcher()
        researcher.propose(load_config(), [])
        record = researcher.last_selection["decision_record"]
        self.assertEqual(
            set(record),
            {
                "hypothesis", "mechanism_basis", "family", "proposed_action",
                "expected_gain", "novelty", "risk", "required_confirmation",
            },
        )
        default_skill_registry().require(record["proposed_action"])


if __name__ == "__main__":
    unittest.main()
