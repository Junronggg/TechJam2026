from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from techjam_agent.config import apply_changes, experiment_key

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


if __name__ == "__main__":
    unittest.main()
