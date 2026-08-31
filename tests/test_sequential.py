from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes, validate_config
from techjam_agent.model_interface import registered_model_ids
from techjam_agent.operator_registry import MODEL_SPECS
from techjam_agent.sequence import build_previous_positive_context
from techjam_agent.sequential import FPMC, SequentialFM


def row(user: str, item: str, label: int, timestamp: int) -> tuple:
    return (20220408, user, item, "author", "tab", 1000.0, label, "tag", timestamp)


def encoded(split_rows: dict[str, list[tuple]]) -> dict[str, tuple]:
    user_ids = {"u1": 0, "u2": 1, "new": 2}
    item_ids = {"v1": 10, "v2": 11, "v3": 12, "new_item": 13}
    result = {}
    for split, rows in split_rows.items():
        matrix = np.zeros((len(rows), 5), dtype=np.int32)
        matrix[:, 0] = [user_ids[value[1]] for value in rows]
        matrix[:, 1] = [item_ids[value[2]] for value in rows]
        matrix[:, 2:] = np.asarray([20, 30, 40], dtype=np.int32)
        labels = np.asarray([value[6] for value in rows], dtype=np.float32)
        result[split] = (matrix, labels, [value[1] for value in rows])
    return result


class SequenceBuilderTests(unittest.TestCase):
    def test_context_is_strictly_earlier_and_same_timestamp_safe(self) -> None:
        splits = {
            "train": [
                row("u1", "v1", 1, 100),
                row("u1", "v2", 0, 100),
                row("u1", "v3", 0, 200),
                row("u1", "v2", 1, 300),
            ],
            "valid": [
                row("u1", "v3", 1, 400),
                row("u1", "v1", 0, 500),
                row("new", "new_item", 1, 500),
            ],
            "test": [row("u1", "v3", 0, 600)],
        }
        context = build_previous_positive_context(splits, encoded(splits))
        pad = context.padding_item
        self.assertEqual(context.previous_items["train"].tolist(), [pad, pad, 0, 0])
        # Validation labels never update the offline feature state: both u1
        # rows and the later test row see the final training positive v2.
        self.assertEqual(context.previous_items["valid"].tolist(), [1, 1, pad])
        self.assertEqual(context.previous_items["test"].tolist(), [1])


class FPMCTests(unittest.TestCase):
    def test_bpr_update_learns_positive_order_and_predictions_are_finite(self) -> None:
        model = FPMC(2, 3, embedding_dim=4, learning_rate=0.05, l2=0.0, seed=3)
        users = np.asarray([0, 0], dtype=np.int32)
        positive = np.asarray([0, 0], dtype=np.int32)
        negative = np.asarray([1, 1], dtype=np.int32)
        previous = np.asarray([2, 2], dtype=np.int32)
        for _ in range(40):
            loss = model.bpr_step(users, positive, negative, previous)
        self.assertTrue(np.isfinite(loss))
        scores = model.predict(
            np.asarray([0, 0]), np.asarray([0, 1]), np.asarray([2, 2])
        )
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(scores[0], scores[1])

    def test_checkpoint_round_trip(self) -> None:
        model = FPMC(2, 3, embedding_dim=4, seed=1)
        expected = model.predict(
            np.asarray([0, 1]), np.asarray([1, 2]), np.asarray([0, 3])
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path, best_epoch=7)
            restored = FPMC(2, 3, embedding_dim=4, seed=99)
            self.assertEqual(restored.load(path), 7)
            actual = restored.predict(
                np.asarray([0, 1]), np.asarray([1, 2]), np.asarray([0, 3])
            )
        np.testing.assert_allclose(actual, expected)

    def test_registry_and_config_expose_sequence_capability_safely(self) -> None:
        self.assertEqual(set(registered_model_ids()), set(MODEL_SPECS))
        self.assertTrue(MODEL_SPECS["fpmc"].supports_sequence)
        config = json.loads(
            (ROOT / "configs" / "experiment.json").read_text(encoding="utf-8")
        )
        sequence_config = apply_changes(
            config, {
                "model": "fpmc", "training_objective": "bpr", "embedding_dim": 8,
            }
        )
        self.assertEqual(sequence_config["hyperparameters"]["embedding_dim"], 8)
        validate_config(sequence_config)
        invalid = copy.deepcopy(sequence_config)
        invalid["features"]["tag"] = True
        with self.assertRaisesRegex(ValueError, "base fields only"):
            validate_config(invalid)


class SequentialFMTests(unittest.TestCase):
    def test_hybrid_bpr_update_and_checkpoint(self) -> None:
        model = SequentialFM(20, 3, embedding_dim=4, learning_rate=0.05, l2=0.0, seed=4)
        positive_x = np.asarray([[0, 10, 13], [0, 10, 13]], dtype=np.int32)
        negative_x = np.asarray([[0, 11, 13], [0, 11, 13]], dtype=np.int32)
        positive_items = np.asarray([0, 0], dtype=np.int32)
        negative_items = np.asarray([1, 1], dtype=np.int32)
        previous = np.asarray([2, 2], dtype=np.int32)
        for _ in range(40):
            loss = model.bpr_step(
                positive_x, negative_x, positive_items, negative_items, previous
            )
        self.assertTrue(np.isfinite(loss))
        scores = model.predict(
            np.vstack((positive_x[0], negative_x[0])),
            np.asarray([0, 1]),
            np.asarray([2, 2]),
        )
        self.assertGreater(scores[0], scores[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seq_fm.npz"
            model.save(path, best_epoch=5)
            restored = SequentialFM(20, 3, embedding_dim=4, seed=9)
            self.assertEqual(restored.load(path), 5)
            restored_scores = restored.predict(
                np.vstack((positive_x[0], negative_x[0])),
                np.asarray([0, 1]),
                np.asarray([2, 2]),
            )
        np.testing.assert_allclose(restored_scores, scores)


if __name__ == "__main__":
    unittest.main()
