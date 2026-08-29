"""Isolated subprocess backend for real KuaiRand-Pure FM validation runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from experiment.executor import SubprocessExecutor
from experiment.schemas import (
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    MetricBundle,
    ModelConfig,
    write_json,
)


class OfficialFMSubprocessBackend:
    def __init__(
        self,
        project_root: Path,
        starter_dir: Path,
        data_dir: Path,
        evaluator_sha256: str,
        timeout_seconds: float = 15 * 60,
    ) -> None:
        self.project_root = project_root.resolve()
        self.starter_dir = starter_dir.resolve()
        self.data_dir = data_dir.resolve()
        self.evaluator_sha256 = evaluator_sha256
        self.timeout_seconds = timeout_seconds
        self.executor = SubprocessExecutor()

    def execute(
        self,
        spec: ExperimentSpec,
        config: ModelConfig,
        parent_metrics: MetricBundle,
        run_dir: Path,
    ) -> ExperimentResult:
        del parent_metrics
        return self.run_config(spec.experiment_id, config, run_dir)

    def run_config(
        self, experiment_id: str, config: ModelConfig, run_dir: Path
    ) -> ExperimentResult:
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = run_dir / "backend_request.json"
        write_json(
            request_path,
            {
                "experiment_id": experiment_id,
                "config": config.to_dict(),
                "starter_dir": str(self.starter_dir),
                "data_dir": str(self.data_dir),
                "run_dir": str(run_dir),
                "evaluator_sha256": self.evaluator_sha256,
            },
        )
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "recommender.official_fm_worker",
            "--request",
            str(request_path),
        ]
        command_result = self.executor.run(
            command=command,
            cwd=self.project_root,
            run_dir=run_dir,
            timeout_seconds=self.timeout_seconds,
            environment={"PYTHONUTF8": "1"},
        )
        if command_result.timed_out:
            return ExperimentResult(
                experiment_id=experiment_id,
                status=ExperimentStatus.TIMEOUT,
                runtime_seconds=command_result.runtime_seconds,
                error=f"Training exceeded {self.timeout_seconds:.0f}s timeout",
            )

        result_path = run_dir / "backend_result.json"
        if not result_path.is_file():
            stderr = command_result.stderr_path.read_text(encoding="utf-8", errors="replace")
            return ExperimentResult(
                experiment_id=experiment_id,
                status=ExperimentStatus.FAILED,
                runtime_seconds=command_result.runtime_seconds,
                error=stderr[-8000:] or f"Worker exited with code {command_result.return_code}",
            )
        return self._read_result(result_path)

    @staticmethod
    def _read_result(path: Path) -> ExperimentResult:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_metrics = payload.get("metrics")
        metrics = None
        if raw_metrics is not None:
            metrics = MetricBundle(
                gauc=float(raw_metrics["GAUC"]),
                ndcg_at_5=float(raw_metrics["nDCG@5"]),
            )
        return ExperimentResult(
            experiment_id=payload["experiment_id"],
            status=ExperimentStatus(payload["status"]),
            metrics=metrics,
            runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
            checkpoint=payload.get("checkpoint"),
            prediction_path=payload.get("prediction_path"),
            error=payload.get("error"),
            recovery_attempts=int(payload.get("recovery_attempts", 0)),
        )

