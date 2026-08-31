from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .research_diagnostics import conditional_complementarity, evaluate_slices
from .evidence_escalator import ConfirmationAction


def _error_summary(value: Any, limit: int = 1000) -> str:
    """Return the useful final exception line while the result file keeps the traceback."""
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    message = lines[-1] if lines else "isolated worker failed without an error message"
    return message[-limit:]


class IsolatedExperimentRunner:
    """Run training in a child process while keeping final test evaluation local and gated."""

    def __init__(self, runner, timeout_seconds: float = 900.0) -> None:
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def run(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        request = checkpoint.with_suffix(".request.json")
        result = checkpoint.with_suffix(".result.json")
        request.parent.mkdir(parents=True, exist_ok=True)
        result.unlink(missing_ok=True)
        request.write_text(json.dumps({
            "root": str(self.runner.root), "data_dir": str(self.runner.data_dir),
            "starter_dir": str(self.runner.starter_dir),
            "evaluator_sha256": self.runner.evaluator_sha256,
            "config": config, "checkpoint": str(checkpoint), "result": str(result),
        }), encoding="utf-8")
        environment = os.environ.copy()
        source = str(self.runner.root / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "techjam_agent.worker", "--request", str(request)],
                cwd=self.runner.root, env=environment, text=True, capture_output=True,
                timeout=self.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"experiment exceeded {self.timeout_seconds:.0f}s timeout") from exc
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        payload = None
        if result.is_file():
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError("isolated worker wrote an invalid result file") from exc
        if completed.returncode != 0:
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(_error_summary(payload["error"]))
            message = completed.stderr[-4000:] or f"worker exited with code {completed.returncode}"
            raise RuntimeError(_error_summary(message))
        if not isinstance(payload, dict):
            raise RuntimeError("isolated worker exited without a result file")
        if payload["status"] != "success":
            raise RuntimeError(_error_summary(payload.get("error")))
        return payload["metrics"]

    def confirm(
        self,
        action: ConfirmationAction,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Execute a multi-fit confirmation action in a separate process."""
        request = output_dir.with_suffix(".request.json")
        result = output_dir.with_suffix(".result.json")
        request.parent.mkdir(parents=True, exist_ok=True)
        result.unlink(missing_ok=True)
        request.write_text(json.dumps({
            "root": str(self.runner.root),
            "data_dir": str(self.runner.data_dir),
            "starter_dir": str(self.runner.starter_dir),
            "evaluator_sha256": self.runner.evaluator_sha256,
            "action": action.as_dict(),
            "output_dir": str(output_dir),
            "result": str(result),
        }), encoding="utf-8")
        environment = os.environ.copy()
        source = str(self.runner.root / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        timeout = self.timeout_seconds * max(1, action.estimated_training_runs)
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "techjam_agent.confirmation_worker",
                 "--request", str(request)],
                cwd=self.runner.root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"confirmation exceeded {timeout:.0f}s timeout"
            ) from exc
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        payload = None
        if result.is_file():
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError("confirmation worker wrote an invalid result file") from exc
        if completed.returncode != 0:
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(_error_summary(payload["error"]))
            message = completed.stderr[-4000:] or (
                f"confirmation worker exited with code {completed.returncode}"
            )
            raise RuntimeError(_error_summary(message))
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise RuntimeError("confirmation worker exited without a successful result")
        return payload["result"]

    def finalize(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self.runner.finalize(config, checkpoint, output)

    @staticmethod
    def _validation_artifact(checkpoint: Path) -> Path:
        return checkpoint.with_name(checkpoint.stem + "_validation.npz")

    @staticmethod
    def _load_validation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        with np.load(path, allow_pickle=False) as values:
            users = values["users"].copy()
            labels = values["labels"].copy()
            scores = values["scores"].copy()
            slices = {
                name.removeprefix("slice_"): values[name].copy()
                for name in values.files if name.startswith("slice_")
            }
        return users, labels, scores, slices

    def diagnose(
        self,
        checkpoint: Path,
        champion_checkpoint: Path | None = None,
    ) -> dict[str, Any] | None:
        """Compute fixed slices and candidate-vs-champion error complementarity."""
        path = self._validation_artifact(checkpoint)
        if not path.is_file():
            return None
        users, labels, scores, slices = self._load_validation(path)
        evaluate = self.runner.evaluate_mod.evaluate
        report: dict[str, Any] = {
            "slice_metrics": evaluate_slices(
                evaluate, users, labels, scores, slices, min_rows=100
            ),
            "strong_slice_gain": False,
            "diversity_advantage": False,
        }
        if champion_checkpoint is None:
            return report
        champion_path = self._validation_artifact(champion_checkpoint)
        if not champion_path.is_file():
            return report
        base_users, base_labels, base_scores, base_slices = self._load_validation(
            champion_path
        )
        if not np.array_equal(users, base_users) or not np.array_equal(labels, base_labels):
            raise ValueError("candidate and champion validation artifacts are not aligned")
        comparison = conditional_complementarity(
            evaluate, users, labels, base_scores, scores,
            base_slices or slices, min_rows=100,
        )
        report["complementarity"] = comparison
        overall = comparison.get("overall", {})
        correlation = overall.get("within_user_score_correlation")
        report["within_user_score_correlation"] = correlation
        report["pair_error_recovery_rate"] = overall.get("pair_error_recovery_rate")
        slice_deltas = {
            name: float(values["primary_delta_b_minus_a"])
            for name, values in comparison.items() if name != "overall"
        }
        strongest = max(slice_deltas.items(), key=lambda item: item[1], default=(None, 0.0))
        report["strongest_slice"] = strongest[0]
        report["strongest_slice_gain"] = float(strongest[1])
        report["diversity_advantage"] = bool(
            correlation is not None
            and float(correlation) < 0.95
            and float(overall.get("pair_error_recovery_rate", 0.0)) > 0.05
        )
        return report
