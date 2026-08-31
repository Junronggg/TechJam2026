from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import apply_changes, validate_config
from .autonomous_branch import materialize_code_branch
from .critic import review
from .memory import is_duplicate_config
from .proposals import (
    DeterministicResearcher,
    Proposal,
    legal_candidate_catalog,
    standardize_proposal,
)
from .research import build_research_context
from .tree import (
    ExperimentParent,
    ExperimentTree,
    TreePolicyConfig,
    TreeSearchPolicy,
)


MAX_PROPOSAL_RESOLUTION_ATTEMPTS = 5


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


def _validation_metrics_only(metrics: Any) -> dict[str, float] | None:
    if not isinstance(metrics, dict):
        return None
    cleaned: dict[str, float] = {}
    for key in ("GAUC", "nDCG@5", "primary"):
        value = _as_float(metrics.get(key))
        if value is not None:
            cleaned[key] = value
    return cleaned or None


class Controller:
    def __init__(self, runner, researcher, initial_config: dict[str, Any], project: dict[str, Any],
                 run_dir: Path, artifacts_dir: Path, submissions_dir: Path,
                 clock: Callable[[], float] = time.monotonic,
                 prior_history: list[dict[str, Any]] | None = None,
                 shared_incumbent: dict[str, Any] | None = None,
                 initial_checkpoint: Path | None = None,
                 initial_metrics: dict[str, Any] | None = None) -> None:
        validate_config(initial_config)
        self.runner, self.researcher = runner, researcher
        # Planner autonomy expands the legal candidate catalog; it does not
        # bypass the existing model, setup, validation, or safety boundaries.
        self.autonomous_mode = bool(getattr(researcher, "autonomous_mode", False))
        self.best_config = initial_config
        self.project = project
        self.run_dir, self.artifacts_dir, self.submissions_dir = run_dir, artifacts_dir, submissions_dir
        self.history: list[dict[str, Any]] = []
        self.prior_history = [
            item for item in (prior_history or []) if isinstance(item, dict)
        ]
        self.best_score = float("-inf")
        self.best_checkpoint: Path | None = None
        self.best_iteration: int | None = None
        self.tree = ExperimentTree()
        self.clock = clock
        self.convergence_streak = 0
        self.started = self.clock()
        self.tree_policy = TreeSearchPolicy(TreePolicyConfig(**project.get("tree_search", {})))
        self._pending_parent_selection: dict[str, Any] | None = None
        self.llm_token_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                                "total_tokens": 0}
        self.llm_requests = 0
        self.llm_failures = 0
        self.llm_http_requests = 0
        self.llm_http_failures = 0
        self.llm_proposal_failures = 0
        self.llm_fallbacks = 0
        self.llm_error_categories: dict[str, int] = {}
        self._last_llm_call_ids: list[str] = []
        self._research_context: dict[str, Any] = {}
        self.initial_checkpoint = initial_checkpoint
        self.initial_metrics = _validation_metrics_only(initial_metrics)
        incumbent_metrics = (
            shared_incumbent.get("validation_metrics")
            if isinstance(shared_incumbent, dict) else None
        )
        self.shared_best_score = _as_float(
            incumbent_metrics.get("primary") if isinstance(incumbent_metrics, dict) else None
        )
        if self.shared_best_score is None:
            self.shared_best_score = float("-inf")
        existing_score = self._existing_shared_score()
        self.shared_best_ready = existing_score is not None and (
            existing_score + 1e-12 >= self.shared_best_score
        )
        if existing_score is not None:
            self.shared_best_score = max(self.shared_best_score, existing_score)

    def _planning_history(self) -> list[dict[str, Any]]:
        return [*self.prior_history, *self.history]

    def _existing_shared_score(self) -> float | None:
        path = self.artifacts_dir / "best_metrics.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        score = _as_float(payload.get("primary") if isinstance(payload, dict) else None)
        required = (
            self.artifacts_dir / "best_config.json",
            self.artifacts_dir / "best_model.npz",
        )
        return score if score is not None and all(path.is_file() for path in required) else None

    def _persist_run_best(
        self, config: dict[str, Any], metrics: dict[str, Any], checkpoint: Path
    ) -> None:
        best_dir = self.run_dir / "best"
        _write_json(best_dir / "config.json", config)
        _write_json(best_dir / "metrics.json", metrics)
        shutil.copy2(checkpoint, best_dir / "model.npz")
        text_checkpoint = checkpoint.with_suffix(".txt")
        if text_checkpoint.is_file():
            shutil.copy2(text_checkpoint, best_dir / "model.txt")

    def _maybe_promote_shared_best(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any],
        checkpoint: Path,
        iteration: int,
    ) -> bool:
        """Promote only if this run meets or beats the durable validation incumbent."""
        score = float(metrics["primary"])
        improves = score > self.shared_best_score + 1e-12
        restores_missing_incumbent = (
            not self.shared_best_ready and score + 1e-12 >= self.shared_best_score
        )
        if not improves and not restores_missing_incumbent:
            return False
        _write_json(self.artifacts_dir / "best_config.json", config)
        _write_json(self.artifacts_dir / "best_metrics.json", metrics)
        shutil.copy2(checkpoint, self.artifacts_dir / "best_model.npz")
        text_checkpoint = checkpoint.with_suffix(".txt")
        if text_checkpoint.is_file():
            shutil.copy2(text_checkpoint, self.artifacts_dir / "best_model.txt")
        _write_json(self.artifacts_dir / "best_manifest.json", {
            "validation_primary": score,
            "source_run": self.run_dir.name,
            "source_iteration": iteration,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        self.shared_best_score = score
        self.shared_best_ready = True
        return True

    def _record(self, item: dict[str, Any], parent_id: str | None) -> None:
        self.history.append(item)
        self.tree.add(item["iteration"], parent_id, item)
        _write_json(self.run_dir / f"iteration_{item['iteration']:03d}.json", item)
        _write_json(self.run_dir / "tree_snapshot.json", self.tree.snapshot())
        with (self.run_dir / "experiment_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")

    def _execute(self, iteration: int, config: dict[str, Any], proposal: Proposal,
                 parent: ExperimentParent | None = None, *,
                 cached_checkpoint: Path | None = None,
                 cached_metrics: dict[str, Any] | None = None) -> None:
        """Run one experiment against an explicit parent; global best stays a separate concept."""
        checkpoint = self.run_dir / "checkpoints" / f"iteration_{iteration:03d}.npz"
        global_best_before = None if self.best_score == float("-inf") else self.best_score
        parent_id = None if parent is None else parent.node_id
        parent_primary = None if parent is None else parent.primary
        parent_metrics = self._parent_metrics(parent)
        item = {"iteration": iteration, "timestamp": datetime.now(timezone.utc).isoformat(),
                **proposal.as_dict(),
                "autonomous_mode": self.autonomous_mode,
                "parent_id": parent_id,
                "parent_primary": parent_primary,
                "parent_metrics": parent_metrics,
                # Pre-P2.6 name for parent_primary. Kept so old readers stay valid.
                "parent_score": parent_primary,
                "global_best_primary_before": global_best_before,
                "config": config, "manual_intervention": False}
        research = self._research_context.get("research")
        if isinstance(research, dict):
            ranked = research.get("ranked_candidates")
            ranked = ranked if isinstance(ranked, list) else []
            selected = next(
                (row for row in ranked if row.get("candidate_id") == proposal.candidate_id),
                {},
            )
            item.update({
                "research_phase": research.get("phase"),
                "expansion_mode": research.get("expansion_mode"),
                "reference_ids": research.get("reference_ids", []),
                "strategy_id": selected.get("strategy_id"),
                "evolution_recipe": selected.get("evolution_recipe"),
                "candidate_rank": (
                    next((index for index, row in enumerate(ranked, start=1)
                          if row.get("candidate_id") == proposal.candidate_id), None)
                ),
                "predicted_utility": selected.get("predicted_utility"),
                "estimated_runtime_seconds": selected.get("estimated_runtime_seconds"),
                "diagnosis_codes": [
                    row.get("code") for row in research.get("diagnoses", [])
                    if isinstance(row, dict) and row.get("code")
                ],
            })
        if (self._pending_parent_selection is not None and
                self._pending_parent_selection.get("parent_id") == parent_id):
            item["parent_selection"] = self._pending_parent_selection
        else:
            item["parent_selection"] = None
        changes = "baseline" if not proposal.changes else ", ".join(
            f"{key}={value}" for key, value in proposal.changes.items())
        print(f"\nIteration {iteration}: {changes}", flush=True)
        print(f"  Researcher: {proposal.source}", flush=True)
        if parent is not None:
            parent_features = [
                key for key, enabled in parent.config.get("features", {}).items() if enabled
            ]
            print(
                f"  Parent: {parent.node_id} | model={parent.config['model']} "
                f"objective={parent.config['training_objective']} "
                f"primary={parent.primary:.6f} | features={parent_features or ['base_only']}",
                flush=True,
            )
        print(f"  Hypothesis: {proposal.hypothesis}", flush=True)
        if proposal.proposal_type == "code_branch" and proposal.code_branch:
            print(
                f"  Generated code branch: {proposal.code_branch.get('branch_name', 'unnamed')} "
                f"({len(str(proposal.code_branch.get('source', ''))):,} chars; static gate passed)",
                flush=True,
            )
        execution_started = self.clock()
        try:
            if cached_checkpoint is not None and cached_metrics is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_checkpoint, checkpoint)
                cached_text = cached_checkpoint.with_suffix(".txt")
                if cached_text.is_file():
                    shutil.copy2(cached_text, checkpoint.with_suffix(".txt"))
                metrics = dict(cached_metrics)
                item["cache_hit"] = True
                item["cache_source"] = str(cached_checkpoint)
                print("  Reusing exact validated checkpoint; no retraining.", flush=True)
            else:
                metrics = self.runner.run(config, checkpoint)
                item["cache_hit"] = False
            item["execution_seconds"] = max(0.0, self.clock() - execution_started)
            slice_path = checkpoint.with_suffix(".slices.json")
            if slice_path.is_file():
                try:
                    slice_report = json.loads(slice_path.read_text(encoding="utf-8"))
                    item["error_slices"] = {
                        "structural": slice_report.get("structural", {}),
                        "worst_slices": list(slice_report.get("worst_slices") or [])[:5],
                    }
                except (OSError, json.JSONDecodeError):
                    item["error_slices"] = None
            score = metrics["primary"]
            decision = "KEEP" if score > self.best_score else "REJECT"
            item.update({"status": "success", "metrics": metrics,
                         "delta_from_parent":
                             None if parent_primary is None else score - parent_primary,
                         "delta_from_best":
                             None if global_best_before is None else score - global_best_before,
                         "delta_from_incumbent":
                             None if global_best_before is None else score - global_best_before,
                         "delta_from_official_baseline": score - float(
                             self.project["baseline"]["validation"]["primary"]
                         ),
                         "decision": decision, "error": None})
            # Search reward is always measured against the global incumbent.
            # A branch-local improvement is useful negative/neutral evidence,
            # but it is never a success and must not attract more budget.
            reward_epsilon = float(self.project["run_limits"]["convergence_epsilon"])
            genuine_gain = (
                decision == "KEEP" and iteration > 0
                and global_best_before is not None
                and score - global_best_before > reward_epsilon
            )
            if genuine_gain:
                item.update({"search_outcome": "global_best", "search_reward": 2})
            else:
                item.update({"search_outcome": "valid_nonimproving", "search_reward": 0})
            item["critique"] = review(
                metrics, parent_primary,
                self.project["run_limits"]["convergence_epsilon"], "success",
                history=self.history, changes=proposal.changes,
                parent_metrics=parent_metrics,
            )
            item["metric_deltas"] = item["critique"].get("metric_deltas", {})
            if iteration > 0:
                self._update_convergence_streak(score, global_best_before)
            if decision == "KEEP":
                self.best_score, self.best_config, self.best_checkpoint = score, config, checkpoint
                self.best_iteration = iteration
                self._persist_run_best(config, metrics, checkpoint)
                item["shared_best_promoted"] = self._maybe_promote_shared_best(
                    config, metrics, checkpoint, iteration
                )
            else:
                item["shared_best_promoted"] = False
            print(f"  Result: primary={score:.6f} | {decision}", flush=True)
        except Exception as exc:
            item["execution_seconds"] = max(0.0, self.clock() - execution_started)
            item.update({"status": "error", "metrics": None,
                         "delta_from_parent": None, "delta_from_best": None,
                          "decision": "REJECT", "error": {"type": type(exc).__name__, "message": str(exc)}})
            item.update({"search_outcome": "execution_failure", "search_reward": -1})
            item["critique"] = review(
                None, parent_primary,
                self.project["run_limits"]["convergence_epsilon"],
                "error", item["error"],
                history=self.history, changes=proposal.changes,
                parent_metrics=parent_metrics,
            )
            item["metric_deltas"] = item["critique"].get("metric_deltas", {})
            print(f"  Error: {type(exc).__name__}: {exc} | REJECT", flush=True)
        self._record(item, parent_id)
        self._pending_parent_selection = None

    def _parent_metrics(self, parent: ExperimentParent | None) -> dict[str, float] | None:
        if parent is None:
            return None
        for item in reversed(self.history):
            if item.get("iteration") == parent.iteration:
                return _validation_metrics_only(item.get("metrics"))
        return None

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
        minimum = int(self.project["run_limits"].get("min_candidate_experiments", 0))
        completed_candidates = sum(
            item.get("iteration", 0) > 0 and item.get("status") == "success"
            for item in self.history
        )
        return (
            completed_candidates >= minimum
            and self.convergence_streak >= int(
                self.project["run_limits"]["convergence_rounds"]
            )
        )

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

    def _select_parent(self) -> ExperimentParent | None:
        """Choose a parent from the branch-preserving search frontier."""
        budget_seconds = float(self.project["run_limits"]["max_wall_clock_hours"]) * 3600.0
        remaining = max(0.0, budget_seconds - self._elapsed())
        try:
            selection = self.tree_policy.select(
                self.history,
                remaining,
                getattr(self, "_search_progress", 0.0),
            )
        except RuntimeError:
            self._pending_parent_selection = None
            return None
        self._pending_parent_selection = selection.as_dict()
        return selection.parent

    def _capture_researcher_accounting(self, researcher: Any, *, failed: bool) -> None:
        records = getattr(researcher, "last_call_records", [])
        if not isinstance(records, list):
            records = []
        records = [item for item in records if isinstance(item, dict)]
        self._last_llm_call_ids = [
            str(item["call_id"]) for item in records if item.get("call_id")
        ]
        if records:
            self._persist_llm_calls(records)
        attempts = getattr(researcher, "last_attempts", 0)
        try:
            attempts = max(0, int(attempts))
        except (TypeError, ValueError):
            attempts = 0
        if records:
            attempts = len(records)
        is_llm = bool(getattr(researcher, "is_llm", False) or records or attempts)
        usage = getattr(researcher, "last_token_usage", {})
        if isinstance(usage, dict):
            for key in self.llm_token_usage:
                try:
                    self.llm_token_usage[key] += max(0, int(usage.get(key, 0) or 0))
                except (TypeError, ValueError):
                    continue
        self.llm_requests += attempts
        self.llm_http_requests += attempts
        for item in records:
            error = item.get("error") if isinstance(item.get("error"), dict) else {}
            status = item.get("http_status")
            try:
                failed_status = status is not None and int(status) >= 400
            except (TypeError, ValueError):
                failed_status = False
            unavailable_transport = (
                status is None and error.get("category") in {"network", "timeout"}
            )
            if failed_status or unavailable_transport:
                self.llm_http_failures += 1
        for item in records:
            error = item.get("error") if isinstance(item.get("error"), dict) else {}
            category = error.get("category")
            if isinstance(category, str) and category:
                self.llm_error_categories[category] = (
                    self.llm_error_categories.get(category, 0) + 1
                )
        if failed and is_llm:
            self.llm_failures += 1
            self.llm_proposal_failures += 1

    def _persist_llm_calls(self, records: list[dict[str, Any]]) -> None:
        path = self.run_dir / "llm_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                call_id = str(record.get("call_id") or "unknown_call")
                _write_json(self.run_dir / "llm_calls" / f"{call_id}.json", record)
                handle.write(
                    json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
                )

    def _propose(self, researcher, parent: ExperimentParent) -> tuple[Any, Any, tuple[str, str] | None]:
        """Resolve one legal, non-duplicate candidate, or report why the search stopped."""
        set_context = getattr(researcher, "set_run_context", None)
        if callable(set_context):
            set_context(dict(self._research_context))
        last_problem: tuple[str, str] | None = None
        for _ in range(MAX_PROPOSAL_RESOLUTION_ATTEMPTS):
            failed = False
            try:
                proposal = researcher.propose(parent.config, self._planning_history())
            except StopIteration as exc:
                failed = True
                self._capture_researcher_accounting(researcher, failed=failed)
                return None, None, ("search_exhausted", f"{type(exc).__name__}: {exc}")
            except (ValueError, RuntimeError) as exc:
                failed = True
                self._capture_researcher_accounting(researcher, failed=failed)
                return None, None, ("search_exhausted", f"{type(exc).__name__}: {exc}")
            self._capture_researcher_accounting(researcher, failed=failed)
            try:
                proposal = standardize_proposal(
                    proposal,
                    parent.config,
                    self._planning_history(),
                    autonomous=self.autonomous_mode,
                )
                candidate = apply_changes(parent.config, proposal.changes)
                if proposal.proposal_type == "code_branch":
                    if candidate.get("model") != "custom" or not proposal.code_branch:
                        raise ValueError("generated branch must resolve to model='custom'")
                    branch = materialize_code_branch(
                        self.artifacts_dir.parent, proposal.code_branch
                    )
                    candidate.update({
                        "code_branch": branch["source_path"],
                        "code_branch_sha256": branch["sha256"],
                        "code_branch_name": branch["branch_name"],
                    })
                    validate_config(candidate)
            except (KeyError, TypeError, ValueError) as exc:
                last_problem = ("search_exhausted", f"invalid proposal: {type(exc).__name__}: {exc}")
                continue
            if is_duplicate_config(candidate, self._planning_history()):
                last_problem = ("duplicate_configuration",
                                "candidate configuration already executed")
                continue
            return proposal, candidate, None
        return None, None, last_problem or (
            "search_exhausted", "planner could not produce a legal configuration"
        )

    def _budget_block(self, budget_seconds: float, reserve: float) -> str | None:
        remaining = budget_seconds - self._elapsed()
        if remaining <= 0:
            return "wall_clock_exhausted"
        if remaining < reserve:
            return "insufficient_time_for_next_experiment"
        return None

    def run(
        self,
        max_iterations: int | None = None,
        *,
        final_evaluation: bool = True,
    ) -> dict[str, Any]:
        limits = self.project["run_limits"]
        cap = self._iteration_cap(max_iterations)
        budget_seconds = float(limits["max_wall_clock_hours"]) * 3600.0
        reserve = self._experiment_cost_seconds()
        research_settings = self.project.get("research_search", {})
        minimum_reserve = min(
            reserve,
            float(research_settings.get("minimum_reserve_seconds", 60.0)),
        ) if reserve > 0 else 0.0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run log: {self.run_dir}", flush=True)
        _write_json(self.run_dir / "run_meta.json", {"started_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": self.project["benchmark"], "limits": limits,
            "max_total_experiments": cap,
            "prior_experiments_loaded": len(self.prior_history),
            "researcher": {
                "type": type(self.researcher).__name__,
                "model": getattr(self.researcher, "model", None),
                "base_url": getattr(self.researcher, "base_url", None),
                "allow_code_branches": bool(getattr(self.researcher, "allow_code_branches", False)),
                "autonomous_mode": self.autonomous_mode,
            }})
        baseline = Proposal(
            "Reproduce the selected starting configuration.",
            "A freshly measured validation checkpoint anchors this run while prior evidence "
            "prevents repeated experiments.", {}, "system",
        )
        stop_reason, stop_detail = "max_iterations", None
        for iteration in range(cap):
            blocked = self._budget_block(budget_seconds, minimum_reserve)
            if blocked is not None:
                stop_reason = blocked
                break
            if iteration == 0:
                self._execute(
                    0,
                    self.best_config,
                    baseline,
                    None,
                    cached_checkpoint=self.initial_checkpoint,
                    cached_metrics=self.initial_metrics,
                )
                if self.best_checkpoint is None:
                    stop_reason = "baseline_failed"
                    break
                continue
            if self._converged():
                stop_reason = "converged"
                break
            # Keep the no-argument hook stable for custom controllers while exposing
            # run progress to the default progressive tree policy.
            self._search_progress = (iteration - 1) / max(1, cap - 2)
            parent = self._select_parent()
            if parent is None:
                stop_reason, stop_detail = "search_exhausted", "no expandable parent node"
                break
            planning_history = self._planning_history()
            remaining_seconds = max(0.0, budget_seconds - self._elapsed())
            candidates = legal_candidate_catalog(
                parent.config,
                planning_history,
                autonomous=self.autonomous_mode,
            )
            research = build_research_context(
                parent.config,
                planning_history,
                candidates,
                iteration=iteration,
                total_iterations=cap,
                parent_id=parent.node_id,
                remaining_seconds=remaining_seconds,
                shortlist_size=int(research_settings.get("shortlist_size", 5)),
            )
            self._research_context = {
                "iteration": iteration,
                "parent_id": parent.node_id,
                "remaining_iterations": cap - iteration,
                "remaining_seconds": remaining_seconds,
                "estimated_next_experiment_seconds": reserve,
                "total_iterations": cap,
                "research": research,
            }
            if self._pending_parent_selection is None:
                # Custom controller hooks may return a forced parent without using
                # TreeSearchPolicy. Preserve a complete audit record in that case.
                self._pending_parent_selection = {
                    **parent.as_record(),
                    "selection_mode": "custom",
                }
            self._pending_parent_selection.update({
                "research_phase": research["phase"],
                "expansion_mode": research["expansion_mode"],
                "exploration_probability": research["exploration_probability"],
            })
            self._last_llm_call_ids = []
            proposal, candidate, failure = self._propose(self.researcher, parent)
            if failure is not None and not isinstance(self.researcher, DeterministicResearcher):
                is_llm_fallback = bool(
                    getattr(self.researcher, "is_llm", False) or
                    getattr(self.researcher, "last_attempts", 0)
                )
                fallback_reason = {
                    "used": True,
                    "reason_code": (
                        getattr(self.researcher, "last_error_category", None) or failure[0]
                    ),
                    "reason": failure[1],
                    "proposal_episode_id": getattr(
                        self.researcher, "last_proposal_episode_id", None
                    ),
                    "llm_call_ids": list(self._last_llm_call_ids),
                }
                print(
                    "  Researcher proposal unavailable; using explicit deterministic fallback "
                    f"({fallback_reason['reason_code']}).",
                    flush=True,
                )
                if is_llm_fallback:
                    self.llm_fallbacks += 1
                proposal, candidate, failure = self._propose(
                    DeterministicResearcher(autonomous_mode=self.autonomous_mode), parent
                )
                if proposal is not None:
                    proposal = replace(
                        proposal,
                        source="deterministic_fallback",
                        llm_call_ids=tuple(fallback_reason["llm_call_ids"]),
                        fallback=fallback_reason,
                    )
            if failure is not None:
                stop_reason, stop_detail = failure
                break
            selected = next(
                (
                    row for row in research.get("ranked_candidates", [])
                    if row.get("candidate_id") == proposal.candidate_id
                ),
                {},
            )
            estimate = _as_float(selected.get("estimated_runtime_seconds")) or minimum_reserve
            safety = float(research_settings.get("runtime_safety_factor", 1.25))
            if budget_seconds - self._elapsed() < min(reserve, max(minimum_reserve, estimate * safety)):
                stop_reason = "insufficient_time_for_selected_experiment"
                stop_detail = (
                    f"estimated {estimate:.1f}s plus safety margin exceeds remaining budget"
                )
                break
            self._execute(iteration, candidate, proposal, parent)
        final_test = None
        if final_evaluation and self.best_checkpoint is not None:
            final_test = self.runner.finalize(self.best_config, self.best_checkpoint,
                                              self.submissions_dir / "final.csv")
            _write_json(self.artifacts_dir / "final_test_metrics.json", final_test)
        executed = len(self.history)
        elapsed = self._elapsed()
        valid_rows = [
            item for item in self.history
            if item.get("status") == "success"
            and isinstance(item.get("metrics"), dict)
            and _as_float(item["metrics"].get("nDCG@5")) is not None
        ]
        best_ndcg_row = max(
            valid_rows,
            key=lambda item: (
                float(item["metrics"]["nDCG@5"]),
                float(item["metrics"].get("primary", float("-inf"))),
            ),
            default=None,
        )
        summary = {"stop_reason": stop_reason, "stop_detail": stop_detail,
                   "iterations": executed,
                   "total_experiments": executed,
                   "candidate_experiments": max(0, executed - 1),
                   "prior_experiments_loaded": len(self.prior_history),
                   "best_primary": None if self.best_score == float("-inf") else self.best_score,
                   "best_ndcg_at_5": (
                       None if best_ndcg_row is None
                       else float(best_ndcg_row["metrics"]["nDCG@5"])
                   ),
                   "best_ndcg_iteration": (
                       None if best_ndcg_row is None else best_ndcg_row.get("iteration")
                   ),
                   "best_ndcg_primary": (
                       None if best_ndcg_row is None
                       else float(best_ndcg_row["metrics"].get("primary", 0.0))
                   ),
                   "shared_best_primary": (
                       None if self.shared_best_score == float("-inf")
                       else self.shared_best_score
                   ),
                   "best_iteration": self.best_iteration,
                   "manual_interventions": 0,
                   "autonomous_mode": self.autonomous_mode,
                   "candidate_generation": (
                       "dynamic_compatible_bundles" if self.autonomous_mode else "registered_catalog"
                   ),
                   "final_test_metrics": final_test,
                   "convergence_streak": self.convergence_streak,
                   "elapsed_seconds": elapsed,
                   "remaining_seconds": max(0.0, budget_seconds - elapsed),
                   "wall_clock_seconds": elapsed,
                   "llm_requests": self.llm_requests,
                   "llm_failures": self.llm_failures,
                   "llm_http_requests": self.llm_http_requests,
                   "llm_http_failures": self.llm_http_failures,
                   "llm_proposal_failures": self.llm_proposal_failures,
                   "llm_fallbacks": self.llm_fallbacks,
                   "llm_error_categories": dict(sorted(self.llm_error_categories.items())),
                   "llm_tokens": dict(self.llm_token_usage),
                   "limits": {"max_total_experiments": cap,
                              "official_max_iterations": int(limits["max_iterations"]),
                              "max_wall_clock_hours": limits["max_wall_clock_hours"],
                              "wall_clock_budget_seconds": budget_seconds,
                              "convergence_epsilon": float(limits["convergence_epsilon"]),
                              "convergence_rounds": int(limits["convergence_rounds"]),
                              "min_candidate_experiments": int(
                                  limits.get("min_candidate_experiments", 0)
                              ),
                              "experiment_cost_seconds": reserve,
                              "minimum_experiment_reserve_seconds": minimum_reserve,
                              "max_active_branches": self.tree_policy.config.max_active_branches}}
        trajectory = [
            {
                "iteration": item.get("iteration"),
                "parent_id": item.get("parent_id"),
                "source": item.get("source"),
                "llm_call_ids": item.get("llm_call_ids", []),
                "observation": item.get("observation"),
                "diagnosis": item.get("diagnosis"),
                "hypothesis": item.get("hypothesis"),
                "evidence_ids": item.get("evidence_ids", []),
                "changes": item.get("changes", {}),
                "expected_effect": item.get("expected_effect"),
                "estimated_cost": item.get("estimated_cost"),
                "research_phase": item.get("research_phase"),
                "expansion_mode": item.get("expansion_mode"),
                "reference_ids": item.get("reference_ids", []),
                "strategy_id": item.get("strategy_id"),
                "evolution_recipe": item.get("evolution_recipe"),
                "predicted_utility": item.get("predicted_utility"),
                "search_outcome": item.get("search_outcome"),
                "search_reward": item.get("search_reward"),
                "validation_metrics": _validation_metrics_only(item.get("metrics")),
                "metric_deltas": item.get("metric_deltas", {}),
                "decision": item.get("decision"),
                "critic": item.get("critique"),
                "fallback": item.get("fallback"),
                "token_usage": item.get("token_usage", {}),
            }
            for item in self.history
        ]
        _write_json(self.run_dir / "research_trajectory.json", trajectory)
        _write_json(self.run_dir / "summary.json", summary)
        print(f"\nStopped: {stop_reason} | best_primary={summary['best_primary']}", flush=True)
        return summary
