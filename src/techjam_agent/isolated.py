from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class IsolatedExperimentRunner:
    """Run training in a child process while keeping final test evaluation local and gated."""

    def __init__(self, runner, timeout_seconds: float = 900.0) -> None:
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def run(self, config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
        request = checkpoint.with_suffix(".request.json")
        result = checkpoint.with_suffix(".result.json")
        request.parent.mkdir(parents=True, exist_ok=True)
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
        if completed.returncode != 0 or not result.is_file():
            message = completed.stderr[-4000:] or f"worker exited with code {completed.returncode}"
            raise RuntimeError(message)
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload["status"] != "success":
            raise RuntimeError(payload["error"])
        return payload["metrics"]

    def finalize(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self.runner.finalize(config, checkpoint, output)
