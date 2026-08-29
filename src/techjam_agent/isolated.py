from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


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

    def finalize(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self.runner.finalize(config, checkpoint, output)
