from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import apply_changes, validate_config
from .critic import review
from .memory import is_duplicate_config
from .proposals import DeterministicResearcher, Proposal
from .tree import ExperimentTree


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


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class Controller:
    def __init__(self, runner, researcher, initial_config: dict[str, Any], project: dict[str, Any],
                 run_dir: Path, artifacts_dir: Path, submissions_dir: Path,
                 clock: Callable[[], float] = time.monotonic) -> None:
        validate_config(initial_config)
        self.runner, self.researcher = runner, researcher
        self.best_config = initial_config
        self.project = project
        self.run_dir, self.artifacts_dir, self.submissions_dir = run_dir, artifacts_dir, submissions_dir
        self.history: list[dict[str, Any]] = []
        self.best_score = float("-inf")
        self.best_checkpoint: Path | None = None
        self.best_iteration: int | None = None
        self.tree = ExperimentTree()
        self.clock = clock
        self.convergence_streak = 0
        self.started = self.clock()

    def _record(self, item: dict[str, Any], parent_iteration: int | None) -> None:
        self.history.append(item)
        self.tree.add(item["iteration"], parent_iteration, item)
        _write_json(self.run_dir / f"iteration_{item['iteration']:03d}.json", item)
        _write_json(self.run_dir / "tree_snapshot.json", self.tree.snapshot())
        with (self.run_dir / "experiment_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")

    def _execute(self, iteration: int, config: dict[str, Any], proposal: Proposal) -> None:
        checkpoint = self.run_dir / "checkpoints" / f"iteration_{iteration:03d}.npz"
        parent_score = None if self.best_score == float("-inf") else self.best_score
        parent_iteration = self.best_iteration
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
            item["critique"] = review(
                metrics, parent_score,
                self.project["run_limits"]["convergence_epsilon"], "success",
                history=self.history, changes=proposal.changes,
            )
            if iteration > 0:
                self._update_convergence_streak(score, parent_score)
            if decision == "KEEP":
                self.best_score, self.best_config, self.best_checkpoint = score, config, checkpoint
                self.best_iteration = iteration
                _write_json(self.artifacts_dir / "best_config.json", config)
                _write_json(self.artifacts_dir / "best_metrics.json", metrics)
                shutil.copy2(checkpoint, self.artifacts_dir / "best_model.npz")
            print(f"  Result: primary={score:.6f} | {decision}", flush=True)
        except Exception as exc:
            item.update({"status": "error", "metrics": None, "delta_from_best": None,
                         "decision": "REJECT", "error": {"type": type(exc).__name__, "message": str(exc)}})
            item["critique"] = review(
                None, parent_score,
                self.project["run_limits"]["convergence_epsilon"],
                "error", item["error"],
                history=self.history, changes=proposal.changes,
            )
            print(f"  Error: {type(exc).__name__}: {exc} | REJECT", flush=True)
        self._record(item, parent_iteration)

    def _update_convergence_streak(self, score: Any, parent_score: Any) -> None:
        """Only a finite candidate comparison is convergence evidence; failures leave it as is."""
        candidate, parent = _as_float(score), _as_float(parent_score)
        if candidate is None or parent is None:
            return
        epsilon = float(self.project["run_limits"]["convergence_epsilon"])
        if candidate - parent > epsilon:
            self.convergence_streak = 0
        else:
            self.convergence_streak += 1

    def _converged(self) -> bool:
        return self.convergence_streak >= int(self.project["run_limits"]["convergence_rounds"])

    def _iteration_cap(self, max_iterations: int | None) -> int:
        """Total executed experiments, baseline included, clamped to the official maximum."""
        official = int(self.project["run_limits"]["max_iterations"])
        if max_iterations is None:
            return official
        return max(1, min(int(max_iterations), official))

    def _experiment_cost_seconds(self) -> float:
        """Conservative per-experiment reservation: the configured hard timeout."""
        timeout = _as_float(self.project.get("experiment_timeout_seconds"))
        return timeout if timeout is not None and timeout > 0 else 0.0

    def _elapsed(self) -> float:
        return self.clock() - self.started

    def _propose(self, researcher) -> tuple[Any, Any, tuple[str, str] | None]:
        """Resolve one legal, non-duplicate candidate, or report why the search stopped."""
        try:
            proposal = researcher.propose(self.best_config, self.history)
            candidate = apply_changes(self.best_config, proposal.changes)
        except (StopIteration, ValueError, RuntimeError) as exc:
            return None, None, ("search_exhausted", f"{type(exc).__name__}: {exc}")
        if is_duplicate_config(candidate, self.history):
            return None, None, ("duplicate_configuration", "candidate configuration already executed")
        return proposal, candidate, None

    def _budget_block(self, budget_seconds: float, reserve: float) -> str | None:
        remaining = budget_seconds - self._elapsed()
        if remaining <= 0:
            return "wall_clock_exhausted"
        if remaining < reserve:
            return "insufficient_time_for_next_experiment"
        return None

    def run(self, max_iterations: int | None = None) -> dict[str, Any]:
        limits = self.project["run_limits"]
        cap = self._iteration_cap(max_iterations)
        budget_seconds = float(limits["max_wall_clock_hours"]) * 3600.0
        reserve = self._experiment_cost_seconds()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run log: {self.run_dir}", flush=True)
        _write_json(self.run_dir / "run_meta.json", {"started_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": self.project["benchmark"], "limits": limits,
            "max_total_experiments": cap})
        baseline = Proposal("Reproduce the official FM baseline.",
                            "A verified baseline anchors every subsequent comparison.", {}, "system")
        stop_reason, stop_detail = "max_iterations", None
        for iteration in range(cap):
            blocked = self._budget_block(budget_seconds, reserve)
            if blocked is not None:
                stop_reason = blocked
                break
            if iteration == 0:
                self._execute(0, self.best_config, baseline)
                if self.best_checkpoint is None:
                    stop_reason = "baseline_failed"
                    break
                continue
            if self._converged():
                stop_reason = "converged"
                break
            proposal, candidate, failure = self._propose(self.researcher)
            if failure is not None and not isinstance(self.researcher, DeterministicResearcher):
                proposal, candidate, failure = self._propose(DeterministicResearcher())
            if failure is not None:
                stop_reason, stop_detail = failure
                break
            self._execute(iteration, candidate, proposal)
        final_test = None
        if self.best_checkpoint is not None:
            final_test = self.runner.finalize(self.best_config, self.best_checkpoint,
                                              self.submissions_dir / "final.csv")
            _write_json(self.artifacts_dir / "final_test_metrics.json", final_test)
        executed = len(self.history)
        elapsed = self._elapsed()
        summary = {"stop_reason": stop_reason, "stop_detail": stop_detail,
                   "iterations": executed,
                   "total_experiments": executed,
                   "candidate_experiments": max(0, executed - 1),
                   "best_primary": None if self.best_score == float("-inf") else self.best_score,
                   "best_iteration": self.best_iteration,
                   "manual_interventions": 0,
                   "final_test_metrics": final_test,
                   "convergence_streak": self.convergence_streak,
                   "elapsed_seconds": elapsed,
                   "remaining_seconds": max(0.0, budget_seconds - elapsed),
                   "wall_clock_seconds": elapsed,
                   "limits": {"max_total_experiments": cap,
                              "official_max_iterations": int(limits["max_iterations"]),
                              "max_wall_clock_hours": limits["max_wall_clock_hours"],
                              "wall_clock_budget_seconds": budget_seconds,
                              "convergence_epsilon": float(limits["convergence_epsilon"]),
                              "convergence_rounds": int(limits["convergence_rounds"]),
                              "experiment_cost_seconds": reserve}}
        _write_json(self.run_dir / "summary.json", summary)
        print(f"\nStopped: {stop_reason} | best_primary={summary['best_primary']}", flush=True)
        return summary
