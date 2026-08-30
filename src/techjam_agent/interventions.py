from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InterventionLogger:
    """Append-only audit trail for human changes made during an autonomous run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []

    def record(self, reason: str, action: str, avoidable: bool) -> dict[str, Any]:
        if not reason.strip() or not action.strip():
            raise ValueError("intervention reason and action must be non-empty")
        event = {
            "intervention_id": f"manual_{len(self.events) + 1:03d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason.strip(),
            "action": action.strip(),
            "avoidable": bool(avoidable),
        }
        self.events.append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def avoidable_count(self) -> int:
        return sum(bool(event["avoidable"]) for event in self.events)
