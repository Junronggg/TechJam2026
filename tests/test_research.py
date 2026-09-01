from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes
from techjam_agent.proposals import legal_candidate_catalog
from techjam_agent.research import (
    build_research_context,
    diagnose_history,
    research_phase,
    retrieve_experiences,
)


def base_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))


class ProgressiveResearchTests(unittest.TestCase):
    def test_exploration_decays_into_confirmation(self) -> None:
        early = research_phase(1, 10)
        middle = research_phase(6, 10)
        late = research_phase(9, 10)
        self.assertEqual(early[0], "explore")
        self.assertEqual(middle[0], "focus")
        self.assertEqual(late[0], "confirm")
        self.assertGreater(early[2], middle[2])
        self.assertGreater(middle[2], late[2])
        self.assertEqual(research_phase(1, 2)[0], "explore")

    def test_diagnosis_detects_runtime_and_metric_tradeoff(self) -> None:
        history = [{
            "iteration": 1,
            "status": "error",
            "error": {"type": "TimeoutError", "message": "worker exceeded 300s"},
            "changes": {"model": "sasrec"},
        }, {
            "iteration": 2,
            "status": "success",
            "changes": {"tag": True},
            "critique": {"metric_deltas": {"GAUC": 0.002, "nDCG@5": -0.001}},
        }]
        codes = {item["code"] for item in diagnose_history(history)}
        self.assertIn("runtime_bottleneck", codes)
        self.assertIn("metric_tradeoff", codes)

    def test_noise_keep_is_retrieved_as_failure_not_success(self) -> None:
        history = [{
            "iteration": 1,
            "status": "success",
            "decision": "KEEP",
            "changes": {"tag": True},
            "metrics": {"GAUC": 0.6, "nDCG@5": 0.6, "primary": 0.6},
            "critique": {"verdict": "noise"},
        }]
        retrieved = retrieve_experiences(history, {"features": "tag"})
        self.assertEqual(retrieved["successes"], [])
        self.assertEqual(retrieved["failures"][0]["outcome"], "valid_nonimproving")

    def test_ranked_shortlist_is_legal_and_penalizes_timed_out_family(self) -> None:
        base = base_config()
        history = [{
            "iteration": 1,
            "status": "error",
            "decision": "REJECT",
            "changes": {"model": "sasrec", "training_objective": "bpr"},
            "config": apply_changes(base, {"model": "sasrec", "training_objective": "bpr"}),
            "error": {"type": "TimeoutError", "message": "worker exceeded timeout"},
            "critique": {"verdict": "failed"},
        }]
        catalog = legal_candidate_catalog(base, history)
        context = build_research_context(
            base,
            history,
            catalog,
            iteration=1,
            total_iterations=10,
            parent_id="baseline",
            remaining_seconds=3600,
            shortlist_size=5,
        )
        legal_ids = {item["candidate_id"] for item in catalog}
        self.assertTrue(context["ranked_candidates"])
        self.assertLessEqual(len(context["ranked_candidates"]), 5)
        self.assertTrue(all(
            item["candidate_id"] in legal_ids for item in context["ranked_candidates"]
        ))
        self.assertTrue(all(
            not (item["changes"].get("model") == "sasrec")
            for item in context["ranked_candidates"]
        ))


if __name__ == "__main__":
    unittest.main()
