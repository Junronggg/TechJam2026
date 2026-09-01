from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes, validate_config
from techjam_agent.field_aware import FieldAwareFM
from techjam_agent.research_state import load_research_state


class FieldAwareModelTests(unittest.TestCase):
    def test_bce_and_bpr_updates_are_finite_and_change_scores(self) -> None:
        X = np.asarray([[0, 2, 4], [1, 3, 5], [0, 3, 4]], dtype=np.int32)
        labels = np.asarray([1, 0, 1], dtype=np.float32)
        model = FieldAwareFM(dim=6, fields=3, k=4, lr=0.01, seed=7)
        before = model.predict(X)
        loss = model.step(X, labels)
        self.assertTrue(math.isfinite(loss))
        self.assertFalse(np.allclose(before, model.predict(X)))
        pair_loss = model.bpr_step(X[[0, 2]], X[[1, 1]])
        self.assertTrue(math.isfinite(pair_loss))

    def test_linear_and_ffm_are_legal_executable_families(self) -> None:
        baseline = json.loads((ROOT / "configs" / "experiment.json").read_text())
        validate_config(apply_changes(baseline, {"model": "linear"}))
        validate_config(apply_changes(baseline, {"model": "ffm"}))

    def test_incumbent_can_atomically_switch_to_every_model_family(self) -> None:
        incumbent = load_research_state(
            ROOT / "configs" / "research_state.json"
        )["incumbent"]["config"]
        validate_config(apply_changes(incumbent, {
            "model": "ffm", "ensemble_size": 1,
            "ensemble_seed_set": "sequential",
        }))
        validate_config(apply_changes(incumbent, {
            "model": "linear", "ensemble_size": 1,
            "ensemble_seed_set": "sequential",
        }))
        validate_config(apply_changes(incumbent, {
            "model": "lightgbm", "training_objective": "bce",
            "ensemble_size": 1, "ensemble_seed_set": "sequential",
        }))


if __name__ == "__main__":
    unittest.main()
