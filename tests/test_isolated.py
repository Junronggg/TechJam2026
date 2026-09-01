"""Error propagation tests for isolated experiments. No training is launched."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.isolated import IsolatedExperimentRunner


class FakeRunner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = root / "data"
        self.starter_dir = root / "starter"
        self.evaluator_sha256 = "digest"


class ErrorPropagationTests(unittest.TestCase):
    def test_worker_result_exposes_the_actual_final_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints" / "iteration_002.npz"
            result = checkpoint.with_suffix(".result.json")

            class FailedWorker:
                pid = 123
                returncode = 1

                def __init__(self, *args, **kwargs):
                    del args, kwargs
                    result.parent.mkdir(parents=True, exist_ok=True)
                    result.write_text(json.dumps({
                        "status": "error",
                        "error": "Traceback (most recent call last):\n"
                                 "ModuleNotFoundError: No module named 'lightgbm'",
                    }), encoding="utf-8")

                def wait(self, timeout=None):
                    return self.returncode

            isolated = IsolatedExperimentRunner(FakeRunner(root))
            with patch("techjam_agent.isolated.subprocess.Popen", FailedWorker):
                with self.assertRaises(RuntimeError) as raised:
                    isolated.run({"model": "lightgbm"}, checkpoint)
        self.assertEqual(
            str(raised.exception), "ModuleNotFoundError: No module named 'lightgbm'"
        )

    def test_stale_result_is_removed_before_worker_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints" / "iteration_002.npz"
            result = checkpoint.with_suffix(".result.json")
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text('{"status":"success","metrics":{"primary":1}}', encoding="utf-8")

            test_case = self

            class FailedWithoutResult:
                pid = 123
                returncode = 1

                def __init__(self, *args, **kwargs):
                    del args
                    test_case.assertFalse(result.exists())
                    kwargs["stderr"].write("worker crashed")
                    kwargs["stderr"].flush()

                def wait(self, timeout=None):
                    return self.returncode

            isolated = IsolatedExperimentRunner(FakeRunner(root))
            with patch("techjam_agent.isolated.subprocess.Popen", FailedWithoutResult):
                with self.assertRaisesRegex(RuntimeError, "worker crashed"):
                    isolated.run({"model": "lightgbm"}, checkpoint)

    def test_model_specific_timeout_terminates_worker_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints" / "iteration_002.npz"

            class TimedOutWorker:
                pid = 456
                returncode = None

                def __init__(self, *args, **kwargs):
                    del args, kwargs

                def wait(self, timeout=None):
                    raise subprocess.TimeoutExpired("worker", timeout)

            isolated = IsolatedExperimentRunner(
                FakeRunner(root), timeout_seconds=900, model_timeouts={"multitask": 12}
            )
            with patch("techjam_agent.isolated.subprocess.Popen", TimedOutWorker), \
                    patch.object(isolated, "_terminate_tree") as terminate:
                with self.assertRaisesRegex(TimeoutError, "12s timeout"):
                    isolated.run({"model": "multitask"}, checkpoint)
            terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
