from __future__ import annotations

import json
import os
import signal
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

    def __init__(self, runner, timeout_seconds: float = 900.0,
                 model_timeouts: dict[str, float] | None = None) -> None:
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self.model_timeouts = dict(model_timeouts or {})

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
        timeout = float(self.model_timeouts.get(config.get("model"), self.timeout_seconds))
        stdout_path = checkpoint.with_suffix(".stdout.log")
        stderr_path = checkpoint.with_suffix(".stderr.log")
        command = [sys.executable, "-m", "techjam_agent.worker", "--request", str(request)]
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
                    stderr_path.open("w", encoding="utf-8") as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=self.runner.root,
                    env=environment,
                    text=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=(os.name != "nt"),
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                    ),
                )
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    self._terminate_tree(process)
                    raise TimeoutError(f"experiment exceeded {timeout:.0f}s timeout") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"experiment exceeded {timeout:.0f}s timeout") from exc
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        if stdout:
            print(stdout, end="", flush=True)
        payload = None
        if result.is_file():
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError("isolated worker wrote an invalid result file") from exc
        if returncode != 0:
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(_error_summary(payload["error"]))
            message = stderr[-4000:] or f"worker exited with code {returncode}"
            raise RuntimeError(_error_summary(message))
        if not isinstance(payload, dict):
            raise RuntimeError("isolated worker exited without a result file")
        if payload["status"] != "success":
            raise RuntimeError(_error_summary(payload.get("error")))
        return payload["metrics"]

    @staticmethod
    def _terminate_tree(process: subprocess.Popen) -> None:
        """Terminate the exact isolated worker and descendants after its deadline."""
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                text=True,
                capture_output=True,
                check=False,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def finalize(self, config: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
        return self.runner.finalize(config, checkpoint, output)
