"""Append-only experiment evidence logging."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Mapping, Any


class ExperimentLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

