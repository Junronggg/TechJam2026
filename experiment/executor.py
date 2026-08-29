"""Subprocess boundary with timeout and captured output."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    return_code: int | None
    runtime_seconds: float
    timed_out: bool
    stdout_path: Path
    stderr_path: Path


class SubprocessExecutor:
    def run(
        self,
        command: Sequence[str],
        cwd: Path,
        run_dir: Path,
        timeout_seconds: float,
        environment: Mapping[str, str] | None = None,
    ) -> CommandResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        env = os.environ.copy()
        if environment:
            env.update(environment)
        started = time.monotonic()
        timed_out = False
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                list(command), cwd=cwd, stdout=stdout, stderr=stderr, text=True, env=env
            )
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
                return_code = None
        return CommandResult(
            return_code=return_code,
            runtime_seconds=time.monotonic() - started,
            timed_out=timed_out,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

