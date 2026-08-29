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

            def failed_worker(*args, **kwargs):
                result.parent.mkdir(parents=True, exist_ok=True)
                result.write_text(json.dumps({
                    "status": "error",
                    "error": "Traceback (most recent call last):\n"
                             "ModuleNotFoundError: No module named 'lightgbm'",
                }), encoding="utf-8")
                return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="")

            isolated = IsolatedExperimentRunner(FakeRunner(root))
            with patch("techjam_agent.isolated.subprocess.run", side_effect=failed_worker):
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

            def failed_without_result(*args, **kwargs):
                self.assertFalse(result.exists())
                return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="worker crashed")

            isolated = IsolatedExperimentRunner(FakeRunner(root))
            with patch("techjam_agent.isolated.subprocess.run", side_effect=failed_without_result):
                with self.assertRaisesRegex(RuntimeError, "worker crashed"):
                    isolated.run({"model": "lightgbm"}, checkpoint)


if __name__ == "__main__":
    unittest.main()
