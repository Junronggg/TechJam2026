from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import apply_changes, validate_config
from techjam_agent.error_slices import build_error_slice_report
from techjam_agent.history_models import CandidateAwareDIN, MetadataSASRec
from techjam_agent.lightgcn import LightGCN


class AdvancedModelForwardTests(unittest.TestCase):
    def _inputs(self):
        field_dims = [4, 5, 3, 3, 4]
        metadata_dims = {"item": 5, "author": 3, "tag": 4, "duration": 4}
        candidate_x = torch.tensor([[0, 1, 1, 0, 2], [1, 2, 0, 1, 3]])
        history = {
            "item": torch.tensor([[1, 2, 0], [0, 0, 0]]),
            "author": torch.tensor([[1, 2, 0], [0, 0, 0]]),
            "tag": torch.tensor([[2, 1, 0], [0, 0, 0]]),
            "duration": torch.tensor([[1, 3, 0], [0, 0, 0]]),
        }
        metadata = {
            "item": torch.tensor([2, 3]), "author": torch.tensor([2, 1]),
            "tag": torch.tensor([1, 3]), "duration": torch.tensor([3, 2]),
        }
        mask = history["item"] != 0
        return field_dims, metadata_dims, candidate_x, history, mask, metadata

    def test_din_is_candidate_aware_and_handles_empty_history(self) -> None:
        args = self._inputs()
        model = CandidateAwareDIN(args[0], args[1], embedding_dim=4, hidden_dim=8)
        output = model(*args[2:])
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(model.scorer.network[0].weight.grad)

    def test_metadata_sasrec_is_finite_with_empty_history(self) -> None:
        args = self._inputs()
        model = MetadataSASRec(
            args[0], args[1], embedding_dim=4, hidden_dim=8, max_seq_len=3,
            num_heads=2, num_layers=1,
        )
        output = model(*args[2:])
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.isfinite(output).all())

    def test_lightgcn_propagates_and_trains(self) -> None:
        model = LightGCN(
            2, 3, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2]),
            embedding_dim=4, num_layers=2,
        )
        propagated = model.propagate()
        positive = model.score(torch.tensor([0, 1]), torch.tensor([0, 2]), propagated)
        negative = model.score(torch.tensor([0, 1]), torch.tensor([2, 0]), propagated)
        loss = -torch.nn.functional.logsigmoid(positive - negative).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.user_embedding.weight.grad)


class AdvancedSearchSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = json.loads((ROOT / "configs" / "experiment.json").read_text())

    def test_new_model_objective_and_hyperparameter_combinations_are_legal(self) -> None:
        for changes in (
            {"model": "din", "training_objective": "group_softmax",
             "negatives_per_positive": 8, "hard_negative_pool_size": 16,
             "sequence_length": 50},
            {"model": "sasrec_meta", "training_objective": "bpr",
             "sequence_length": 10, "dropout": 0.2},
            {"model": "multitask", "training_objective": "group_softmax",
             "auxiliary_weight": 0.1},
            {"model": "lightgcn", "training_objective": "bpr",
             "graph_layers": 3},
            {"model": "lightgcn_hybrid", "training_objective": "bpr",
             "embedding_dim": 32},
        ):
            validate_config(apply_changes(self.base, changes))


class ErrorSliceTests(unittest.TestCase):
    @staticmethod
    def _row(user: str, item: str, label: int, hour: int) -> tuple:
        return (20220422, user, item, "a", "tab", 1000.0, label, "tag",
                1000 + hour, hour, 4, 3, "NORMAL", "HIGH")

    @staticmethod
    def _evaluate(users, labels, scores):
        user_count = len(set(users))
        positive_user_count = len({user for user, label in zip(users, labels) if label > 0})
        ndcg = positive_user_count / max(1, user_count)
        return {"GAUC": 0.5, "nDCG@5": ndcg, "primary": (0.5 + ndcg) / 2,
                "users": user_count, "rows": len(labels)}

    def test_report_exposes_structural_ceiling_and_slices(self) -> None:
        train = [self._row("u1", "a", 1, 1), self._row("u2", "b", 0, 2)]
        valid = [
            self._row("u1", "a", 1, 1), self._row("u1", "b", 0, 1),
            self._row("u2", "a", 0, 2), self._row("u2", "b", 0, 2),
        ]
        report = build_error_slice_report(
            train, valid, np.asarray([0.9, 0.1, 0.2, 0.3]), self._evaluate,
            min_rows=1,
        )
        self.assertEqual(report["structural"]["zero_positive_users"], 1)
        self.assertEqual(report["structural"]["maximum_dataset_ndcg"], 0.5)
        self.assertTrue(any(row["dimension"] == "hour" for row in report["slices"]))


if __name__ == "__main__":
    unittest.main()
