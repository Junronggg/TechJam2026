from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from techjam_agent.config import apply_changes, experiment_key
from techjam_agent.experiment_planner import (
    AutonomousExperimentPlanner,
    choose_ranked,
    rank_candidates,
)
from techjam_agent.proposals import DeterministicResearcher

from scripts.replay_planner_memory import (
    build_validation_archive,
    normalize_logged_config,
    replay_mode,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "experiment.json").read_text())


class PlannerReplayTests(unittest.TestCase):
    def test_normalize_logged_config_fills_new_schema_fields(self):
        current = load_config()
        old = {
            "model": "fm",
            "training_objective": "bpr",
            "hyperparameters": {"learning_rate": 0.0003},
            "features": {"global_context": True},
        }
        normalized = normalize_logged_config(old, current)
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["hyperparameters"]["learning_rate"], 0.0003)
        self.assertEqual(
            set(normalized["hyperparameters"]),
            set(current["hyperparameters"]),
        )
        self.assertTrue(normalized["features"]["global_context"])

    def test_archive_whitelists_validation_metrics_and_uses_median(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_example"
            run.mkdir()
            rows = []
            for primary in (0.60, 0.62, 0.61):
                rows.append({
                    "status": "success",
                    "config": config,
                    "metrics": {
                        "GAUC": primary,
                        "nDCG@5": primary,
                        "primary": primary,
                        "test_primary": 0.99,
                    },
                })
            (run / "experiment_history.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            archive, audit = build_validation_archive(Path(tmp), config)
        result = archive[experiment_key(config)]
        self.assertAlmostEqual(result["metrics"]["primary"], 0.61)
        self.assertNotIn("test_primary", result["metrics"])
        self.assertEqual(audit["successful_validation_rows"], 3)

    def test_replay_uses_logged_outcome_after_supported_selection(self):
        config = load_config()
        bpr = apply_changes(config, {
            "training_objective": "bpr",
            "learning_rate": 0.0003,
        })
        archive = {
            experiment_key(config): {
                "metrics": {"GAUC": 0.60, "nDCG@5": 0.60, "primary": 0.60},
                "observations": 1,
                "primary_min": 0.60,
                "primary_max": 0.60,
                "sources": ["baseline"],
            },
            experiment_key(bpr): {
                "metrics": {"GAUC": 0.61, "nDCG@5": 0.61, "primary": 0.61},
                "observations": 1,
                "primary_min": 0.61,
                "primary_max": 0.61,
                "sources": ["bpr"],
            },
        }
        result = replay_mode("distilled_patterns", config, archive, 2, 0.002)
        self.assertEqual(result["experiments_replayed"], 1)
        self.assertEqual(result["trajectory"][0]["family"], "ranking_objective")
        self.assertAlmostEqual(result["best_primary"], 0.61)

    def test_live_and_replay_agree_when_only_a_soft_stopped_candidate_remains(self) -> None:
        from techjam_agent.experiment_planner import generate_candidates
        from techjam_agent.config import apply_changes as apply

        config = load_config()
        timeout = {"type": "TimeoutError", "message": "experiment exceeded 900s timeout"}
        lightgbm_base = apply(config, {"model": "lightgbm", "training_objective": "bce"})
        history = [{
            "iteration": 1,
            "config": apply(lightgbm_base, {"continuous_history_stats": True}),
            "changes": {"model": "lightgbm", "continuous_history_stats": True},
            "status": "error",
            "critique": {"verdict": "failed"},
            "error": timeout,
        }, {
            "iteration": 2,
            "config": apply(lightgbm_base, {"user_long_view_rate": True}),
            "changes": {"model": "lightgbm", "user_long_view_rate": True},
            "status": "error",
            "critique": {"verdict": "failed"},
            "error": timeout,
        }]
        iteration = 100
        for candidate in generate_candidates(config):
            if candidate.changes.get("model") == "lightgbm":
                continue
            history.append({
                "iteration": iteration,
                "config": apply(config, candidate.changes),
                "changes": candidate.changes,
                "status": "success",
                "critique": {"verdict": "noise"},
            })
            iteration += 1

        live = DeterministicResearcher().propose(config, history)
        planner = AutonomousExperimentPlanner()
        selected = planner.select(config, history)
        replayed = choose_ranked(rank_candidates(config, history))
        self.assertEqual(selected.changes, live.changes)
        self.assertEqual(replayed[0].candidate.changes, live.changes)
        self.assertEqual(live.changes.get("model"), "lightgbm")
        self.assertEqual(planner.last_selection["selection_pass"], "relaxed")

    def test_legacy_direction_stopped_row_is_not_silently_relaxed(self) -> None:
        class LegacyRow:
            def __init__(self, stopped: bool) -> None:
                self.direction_stopped = stopped
                self.candidate = type("C", (), {"changes": {"model": "lightgbm"}})()

        from techjam_agent.experiment_planner import admissible_candidates
        rows = [LegacyRow(True)]
        self.assertEqual(admissible_candidates(rows, relax_soft=False), [])
        self.assertEqual(admissible_candidates(rows, relax_soft=True), [])


if __name__ == "__main__":
    unittest.main()
