from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import apply_changes, experiment_key, validate_config
from .proposals import DeterministicResearcher, Proposal


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars and paths without importing NumPy in the controller."""
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class Controller:
    def __init__(self, runner, researcher, initial_config: dict[str, Any], project: dict[str, Any],
                 run_dir: Path, artifacts_dir: Path, submissions_dir: Path) -> None:
        validate_config(initial_config)
        self.runner, self.researcher = runner, researcher
        self.best_config = initial_config
        self.project = project
        self.run_dir, self.artifacts_dir, self.submissions_dir = run_dir, artifacts_dir, submissions_dir
        self.history: list[dict[str, Any]] = []
        self.best_score = float("-inf")
        self.best_checkpoint: Path | None = None
        self.started = time.monotonic()

    def _record(self, item: dict[str, Any]) -> None:
        self.history.append(item)
        _write_json(self.run_dir / f"iteration_{item['iteration']:03d}.json", item)

    def _execute(self, iteration: int, config: dict[str, Any], proposal: Proposal) -> None:
        checkpoint = self.run_dir / "checkpoints" / f"iteration_{iteration:03d}.npz"
        parent_score = None if self.best_score == float("-inf") else self.best_score
        item = {"iteration": iteration, "timestamp": datetime.now(timezone.utc).isoformat(),
                **proposal.as_dict(), "parent_score": parent_score, "config": config,
                "manual_intervention": False}
        changes = "baseline" if not proposal.changes else ", ".join(
            f"{key}={value}" for key, value in proposal.changes.items())
        print(f"\nIteration {iteration}: {changes}", flush=True)
        print(f"  Hypothesis: {proposal.hypothesis}", flush=True)
        try:
            metrics = self.runner.run(config, checkpoint)
            score = metrics["primary"]
            decision = "KEEP" if score > self.best_score else "REJECT"
            item.update({"status": "success", "metrics": metrics,
                         "delta_from_best": None if parent_score is None else score - parent_score,
                         "decision": decision, "error": None})
            if decision == "KEEP":
                self.best_score, self.best_config, self.best_checkpoint = score, config, checkpoint
                _write_json(self.artifacts_dir / "best_config.json", config)
                _write_json(self.artifacts_dir / "best_metrics.json", metrics)
                shutil.copy2(checkpoint, self.artifacts_dir / "best_model.npz")
            print(f"  Result: primary={score:.6f} | {decision}", flush=True)
        except Exception as exc:
            item.update({"status": "error", "metrics": None, "delta_from_best": None,
                         "decision": "REJECT", "error": {"type": type(exc).__name__, "message": str(exc)}})
            print(f"  Error: {type(exc).__name__}: {exc} | REJECT", flush=True)
        self._record(item)

    def _converged(self) -> bool:
        limits = self.project["run_limits"]
        successful = [x for x in self.history[1:] if x["status"] == "success"]
        recent = successful[-limits["convergence_rounds"]:]
        return len(recent) == limits["convergence_rounds"] and all(
            (x["delta_from_best"] or 0.0) <= limits["convergence_epsilon"] for x in recent)

    def run(self, max_iterations: int | None = None) -> dict[str, Any]:
        limits = self.project["run_limits"]
        cap = min(max_iterations or limits["max_iterations"], limits["max_iterations"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run log: {self.run_dir}", flush=True)
        _write_json(self.run_dir / "run_meta.json", {"started_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": self.project["benchmark"], "limits": limits})
        baseline = Proposal("Reproduce the official FM baseline.",
                            "A verified baseline anchors every subsequent comparison.", {}, "system")
        self._execute(0, self.best_config, baseline)
        stop_reason = "max_iterations"
        for iteration in range(1, cap + 1):
            if self.best_checkpoint is None:
                stop_reason = "baseline_failed"
                break
            if time.monotonic() - self.started >= limits["max_wall_clock_hours"] * 3600:
                stop_reason = "wall_clock_limit"
                break
            if self._converged():
                stop_reason = "converged"
                break
            try:
                proposal = self.researcher.propose(self.best_config, self.history)
                candidate = apply_changes(self.best_config, proposal.changes)
                if experiment_key(candidate) in {experiment_key(x["config"]) for x in self.history}:
                    raise ValueError("researcher proposed a duplicate configuration")
            except (StopIteration, ValueError, RuntimeError) as exc:
                if not isinstance(self.researcher, DeterministicResearcher):
                    proposal = DeterministicResearcher().propose(self.best_config, self.history)
                    candidate = apply_changes(self.best_config, proposal.changes)
                else:
                    stop_reason = f"proposal_stopped: {exc}"
                    break
            self._execute(iteration, candidate, proposal)
        if self.best_checkpoint is not None:
            self.runner.write_submission(self.best_checkpoint, self.submissions_dir / "final.csv")
        summary = {"stop_reason": stop_reason, "iterations": len(self.history),
                   "best_primary": None if self.best_score == float("-inf") else self.best_score,
                   "best_iteration": next((x["iteration"] for x in self.history
                       if x.get("metrics") and x["metrics"]["primary"] == self.best_score), None),
                   "manual_interventions": 0,
                   "wall_clock_seconds": time.monotonic() - self.started}
        _write_json(self.run_dir / "summary.json", summary)
        print(f"\nStopped: {stop_reason} | best_primary={summary['best_primary']}", flush=True)
        return summary
