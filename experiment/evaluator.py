"""Read-only wrapper around the organizer-provided evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Sequence

from experiment.schemas import MetricBundle


class EvaluatorIntegrityError(RuntimeError):
    pass


class OfficialEvaluator:
    def __init__(self, evaluator_path: Path, expected_sha256: str | None = None) -> None:
        self.evaluator_path = evaluator_path.resolve()
        self.expected_sha256 = expected_sha256

    def evaluate(
        self,
        user_ids: Sequence[object],
        labels: Sequence[float],
        scores: Sequence[float],
    ) -> MetricBundle:
        self.verify_integrity()
        output = self._load_module().evaluate(user_ids, labels, scores)
        return MetricBundle(gauc=float(output["GAUC"]), ndcg_at_5=float(output["nDCG@5"]))

    def verify_integrity(self) -> str:
        digest = hashlib.sha256(self.evaluator_path.read_bytes()).hexdigest()
        if self.expected_sha256 and digest != self.expected_sha256:
            raise EvaluatorIntegrityError(
                f"Official evaluator hash mismatch: expected {self.expected_sha256}, got {digest}"
            )
        return digest

    def _load_module(self) -> ModuleType:
        spec = importlib.util.spec_from_file_location("techjam_official_evaluate", self.evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import evaluator from {self.evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

