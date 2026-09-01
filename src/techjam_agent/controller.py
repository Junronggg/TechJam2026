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
from .evidence_escalator import ConfirmationAction, EvidenceEscalator
from .interventions import InterventionLogger
from .memory import build_structured_research_memory, is_duplicate_config
from .proposals import DeterministicResearcher, Proposal
from .research_diagnostics import placebo_verdict
from .skills import SKILL_REGISTRY_VERSION, default_skill_registry
from .sequence_features import SEQUENCE_FEATURE_DIMS
from .tree import (
    ExperimentParent,
    ExperimentTree,
    TreePolicyConfig,
    TreeSearchPolicy,
    node_id_for,
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
        self.tree_policy = TreeSearchPolicy(TreePolicyConfig(**project.get("tree_search", {})))
        self._pending_parent_selection: dict[str, Any] | None = None
        self.llm_token_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                                "total_tokens": 0}
        self.llm_requests = 0
        self.llm_failures = 0
        self.llm_fallbacks: list[dict[str, Any]] = []
        self.competition_converged_at: int | None = None
        self.competition_best_at_convergence: dict[str, Any] | None = None
        self.evidence_escalator = EvidenceEscalator.from_project(project)
        self.skill_registry = default_skill_registry()
        self.auto_confirm = False
        self._confirmation_queue: list[ConfirmationAction] = []
        self.confirmations: list[dict[str, Any]] = []
        self._research_context: dict[str, Any] = {}
        self._pending_candidate_selection: dict[str, Any] | None = None
        self._pending_diagnostic: dict[str, Any] | None = None
        self._diagnostic_queue: list[tuple[dict[str, Any], Proposal, ExperimentParent, dict[str, Any]]] = []
        self._placebo_groups: dict[str, dict[str, Any]] = {}
        self.interventions = InterventionLogger(self.run_dir / "manual_interventions.jsonl")

    def _record(self, item: dict[str, Any], parent_id: str | None) -> None:
        self.skill_registry.require("update_research_memory")
        self.history.append(item)
        self.tree.add(item["iteration"], parent_id, item)
        _write_json(self.run_dir / f"iteration_{item['iteration']:03d}.json", item)
        _write_json(self.run_dir / "tree_snapshot.json", self.tree.snapshot())
        _write_json(
            self.run_dir / "research_memory.json",
            build_structured_research_memory(self.history),
        )
        with (self.run_dir / "experiment_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")

    def record_intervention(self, reason: str, action: str, avoidable: bool) -> dict[str, Any]:
        """Record a human action explicitly; normal launch configuration is not intervention."""
        return self.interventions.record(reason, action, avoidable)

    @staticmethod
    def _is_control(config: dict[str, Any]) -> bool:
        return config["hyperparameters"].get("feature_control", "real") != "real"

    def _diagnose(self, checkpoint: Path, champion: Path | None) -> dict[str, Any] | None:
        self.skill_registry.require("profile_candidate")
        self.skill_registry.require("analyze_prediction_diversity")
        diagnose = getattr(self.runner, "diagnose", None)
        if not callable(diagnose):
            return None
        try:
            return diagnose(checkpoint, champion)
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def _research_decision(
        self,
        score: float,
        parent_primary: float | None,
        diagnostics: dict[str, Any] | None,
        *,
        is_control: bool,
    ) -> str:
        if is_control:
            return "CONTROL"
        if parent_primary is None:
            return "REFERENCE"
        delta = score - parent_primary
        policy = self.project.get("autonomy", {})
        noise = float(policy.get("noise_threshold", 0.0002))
        slice_threshold = float(policy.get("strong_slice_gain", 0.002))
        strongest_slice = 0.0
        diverse = False
        if isinstance(diagnostics, dict):
            strongest_slice = float(diagnostics.get("strongest_slice_gain", 0.0) or 0.0)
            diverse = bool(diagnostics.get("diversity_advantage"))
        if abs(delta) < noise and strongest_slice < slice_threshold and not diverse:
            return "STOP_DIRECTION"
        if delta > 0:
            return "KEEP_CANDIDATE"
        if strongest_slice >= slice_threshold and diverse:
            return "ENSEMBLE_ONLY"
        return "REJECT"

    def _execute(self, iteration: int, config: dict[str, Any], proposal: Proposal,
                 parent: ExperimentParent | None = None) -> None:
        """Run one experiment against an explicit parent; global best stays a separate concept."""
        checkpoint = self.run_dir / "checkpoints" / f"iteration_{iteration:03d}.npz"
        global_best_before = None if self.best_score == float("-inf") else self.best_score
        global_best_node_id = node_id_for(self.best_iteration)
        parent_id = None if parent is None else parent.node_id
        parent_primary = None if parent is None else parent.primary
        # Node identity, not score equality: two nodes can share a Primary value.
        expanded_global_best = parent_id is not None and parent_id == global_best_node_id
        item = {"iteration": iteration, "timestamp": datetime.now(timezone.utc).isoformat(),
                **proposal.as_dict(),
                "parent_id": parent_id,
                "parent_primary": parent_primary,
                # Pre-P2.6 name for parent_primary. Kept so old readers stay valid.
                "parent_score": parent_primary,
                "global_best_primary_before": global_best_before,
                "global_best_node_id_before": global_best_node_id,
                "expanded_global_best": expanded_global_best,
                "config": config, "manual_intervention": False}
        item["candidate_selection"] = self._pending_candidate_selection
        item["diagnostic_group"] = self._pending_diagnostic
        if (self._pending_parent_selection is not None and
                self._pending_parent_selection.get("parent_id") == parent_id):
            item["parent_selection"] = self._pending_parent_selection
        else:
            item["parent_selection"] = None
        changes = "baseline" if not proposal.changes else ", ".join(
            f"{key}={value}" for key, value in proposal.changes.items())
        print(f"\nIteration {iteration}: {changes}", flush=True)
        print(f"  Hypothesis: {proposal.hypothesis}", flush=True)
        try:
            metrics = self.runner.run(config, checkpoint)
            score = float(metrics["primary"])
            is_control = self._is_control(config)
            diagnostics = self._diagnose(
                checkpoint,
                None if is_control else self.best_checkpoint,
            )
            decision = (
                "CONTROL" if is_control
                else "KEEP" if score > self.best_score else "REJECT"
            )
            item.update({"status": "success", "metrics": metrics,
                         "delta_from_parent":
                             None if parent_primary is None else score - parent_primary,
                         "delta_from_best":
                             None if global_best_before is None else score - global_best_before,
                         "decision": decision, "error": None,
                         "diagnostics": diagnostics})
            item["research_decision"] = self._research_decision(
                score, parent_primary, diagnostics, is_control=is_control
            )
            item["critique"] = review(
                metrics, parent_primary,
                self.project["run_limits"]["convergence_epsilon"], "success",
                history=self.history, changes=proposal.changes,
            )
            if iteration > 0 and not is_control:
                self._update_convergence_streak(
                    score, global_best_before, expanded_global_best=expanded_global_best)
            if decision == "KEEP":
                self.best_score, self.best_config, self.best_checkpoint = score, config, checkpoint
                self.best_iteration = iteration
                _write_json(self.artifacts_dir / "best_config.json", config)
                _write_json(self.artifacts_dir / "best_metrics.json", metrics)
                shutil.copy2(checkpoint, self.artifacts_dir / "best_model.npz")
            print(
                f"  Result: primary={score:.6f} | {decision}"
                f" | research={item['research_decision']}",
                flush=True,
            )
        except Exception as exc:
            item.update({"status": "error", "metrics": None,
                         "delta_from_parent": None, "delta_from_best": None,
                         "decision": "REJECT", "error": {"type": type(exc).__name__, "message": str(exc)}})
            item["research_decision"] = "REJECT"
            item["diagnostics"] = None
            item["critique"] = review(
                None, parent_primary,
                self.project["run_limits"]["convergence_epsilon"],
                "error", item["error"],
                history=self.history, changes=proposal.changes,
            )
            print(f"  Error: {type(exc).__name__}: {exc} | REJECT", flush=True)
        self._record(item, parent_id)
        if item["status"] == "success":
            if self._is_control(config):
                self._record_placebo_control(item)
            else:
                self._maybe_schedule_placebos(item, checkpoint)
                diagnostics = item.get("diagnostics")
                waiting_for_placebo = bool(
                    isinstance(diagnostics, dict)
                    and diagnostics.get("placebo_status") == "scheduled"
                )
                if not waiting_for_placebo:
                    self._maybe_schedule_confirmation(
                        item, None if parent is None else parent.config
                    )
        self._pending_parent_selection = None
        self._pending_candidate_selection = None
        self._pending_diagnostic = None

    def _maybe_schedule_placebos(self, item: dict[str, Any], checkpoint: Path) -> None:
        self.skill_registry.require("run_placebo")
        policy = self.project.get("autonomy", {})
        if not bool(policy.get("automatic_placebo", True)):
            return
        feature = next(
            (key for key in SEQUENCE_FEATURE_DIMS if item.get("changes", {}).get(key) is True),
            None,
        )
        delta = _as_float(item.get("delta_from_parent"))
        trigger = float(policy.get("placebo_trigger_delta", 0.002))
        if feature is None or delta is None or not (0.0 < delta <= trigger):
            return
        group_id = f"placebo_{item['iteration']:03d}_{feature}"
        previous_node = item.get("global_best_node_id_before")
        previous_item = next(
            (row for row in self.history
             if node_id_for(row.get("iteration")) == previous_node),
            None,
        )
        group = {
            "group_id": group_id,
            "feature": feature,
            "real_iteration": item["iteration"],
            "real_primary": float(item["metrics"]["primary"]),
            "controls": {},
            "previous_best": {
                "score": item.get("global_best_primary_before"),
                "node_id": previous_node,
                "iteration": None if previous_item is None else previous_item.get("iteration"),
                "config": None if previous_item is None else previous_item.get("config"),
                "metrics": None if previous_item is None else previous_item.get("metrics"),
            },
            "real_checkpoint": str(checkpoint),
        }
        self._placebo_groups[group_id] = group
        if not isinstance(item.get("diagnostics"), dict):
            item["diagnostics"] = {}
        item["diagnostics"]["placebo_status"] = "scheduled"
        item["diagnostics"]["placebo_group_id"] = group_id
        item["diagnostics"]["attribution_reason"] = (
            "positive categorical feature gain requires matched structural controls"
        )
        parent = ExperimentParent(
            node_id_for(item["iteration"]), item["iteration"], item["config"],
            float(item["metrics"]["primary"]), "features",
        )
        for control in ("constant", "shuffled", "random_same_cardinality"):
            controlled = apply_changes(item["config"], {"feature_control": control})
            proposal = Proposal(
                f"Run the matched {control} placebo for {feature}.",
                "Attribution requires the real categorical feature to beat structural controls.",
                {"feature_control": control}, "system-control",
            )
            metadata = {"group_id": group_id, "feature": feature, "control": control}
            self._diagnostic_queue.append((controlled, proposal, parent, metadata))
        _write_json(self.run_dir / f"iteration_{item['iteration']:03d}.json", item)
        _write_json(self.run_dir / "research_memory.json", build_structured_research_memory(self.history))

    def _maybe_schedule_confirmation(
        self,
        item: dict[str, Any],
        reference_config: dict[str, Any] | None,
    ) -> None:
        if not self.auto_confirm:
            return
        decision = self.evidence_escalator.plan_discovery(item, reference_config)
        if not isinstance(item.get("diagnostics"), dict):
            item["diagnostics"] = {}
        item["diagnostics"]["evidence_escalation"] = decision.as_dict()
        action = decision.next_action
        if action is not None:
            known = {
                queued.action_id for queued in self._confirmation_queue
            } | {
                str(row.get("action", {}).get("action_id"))
                for row in self.confirmations
            }
            if action.action_id not in known:
                self._confirmation_queue.append(action)
        _write_json(self.run_dir / f"iteration_{item['iteration']:03d}.json", item)
        _write_json(
            self.run_dir / "research_memory.json",
            build_structured_research_memory(self.history),
        )

    def _execute_confirmation(
        self,
        action_iteration: int,
        action: ConfirmationAction,
    ) -> None:
        confirmation_skill = {
            "rolling": "run_rolling",
            "paired_seeds": "run_paired_seeds",
        }.get(action.kind)
        if confirmation_skill is None:
            raise ValueError(f"unsupported confirmation kind: {action.kind}")
        self.skill_registry.require(confirmation_skill)
        output_dir = self.run_dir / "confirmations" / action.action_id
        record: dict[str, Any] = {
            "action_iteration": action_iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": "confirmation",
            "action": action.as_dict(),
            "manual_intervention": False,
        }
        print(
            f"\nAction {action_iteration}: confirm={action.kind} "
            f"target=iteration_{action.target_iteration:03d}",
            flush=True,
        )
        target = next(
            (row for row in self.history
             if row.get("iteration") == action.target_iteration),
            None,
        )
        try:
            confirm = getattr(self.runner, "confirm", None)
            if not callable(confirm):
                raise RuntimeError("runner does not implement confirmation actions")
            result = confirm(action, output_dir)
            if result.get("test_labels_used") is not False:
                raise ValueError("confirmation result must explicitly exclude test labels")
            decision = self.evidence_escalator.evaluate(action, result)
            record.update({
                "status": "success",
                "result": result,
                "decision": decision.as_dict(),
                "error": None,
            })
            if decision.next_action is not None:
                self._confirmation_queue.append(decision.next_action)
            if isinstance(target, dict):
                if not isinstance(target.get("diagnostics"), dict):
                    target["diagnostics"] = {}
                confirmations = target["diagnostics"].setdefault("confirmations", [])
                confirmations.append({
                    "action_id": action.action_id,
                    "kind": action.kind,
                    "result": result,
                    "decision": decision.as_dict(),
                })
                target["diagnostics"]["scientific_status"] = decision.scientific_status
                target["diagnostics"]["competition_status"] = decision.competition_status
                target["diagnostics"]["confirmation_status"] = (
                    "scheduled" if decision.next_action is not None else "complete"
                )
                if decision.scientific_status == "REJECTED":
                    target["research_decision"] = "STOP_DIRECTION"
                elif decision.scientific_status in {"VALIDATED", "UNCERTAIN"}:
                    target["research_decision"] = "KEEP_CANDIDATE"
                _write_json(
                    self.run_dir / f"iteration_{target['iteration']:03d}.json",
                    target,
                )
            print(
                f"  Confirmation: scientific={decision.scientific_status} "
                f"competition={decision.competition_status}",
                flush=True,
            )
        except Exception as exc:
            record.update({
                "status": "error",
                "result": None,
                "decision": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
            if isinstance(target, dict):
                if not isinstance(target.get("diagnostics"), dict):
                    target["diagnostics"] = {}
                target["diagnostics"]["confirmation_status"] = "failed"
            print(f"  Confirmation error: {type(exc).__name__}: {exc}", flush=True)
        self.confirmations.append(record)
        _write_json(
            self.run_dir / f"confirmation_{action_iteration:03d}.json", record
        )
        with (self.run_dir / "confirmation_history.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        _write_json(
            self.run_dir / "research_memory.json",
            build_structured_research_memory(self.history),
        )

    def _record_placebo_control(self, item: dict[str, Any]) -> None:
        metadata = item.get("diagnostic_group")
        if not isinstance(metadata, dict):
            return
        group = self._placebo_groups.get(str(metadata.get("group_id")))
        if group is None:
            return
        group["controls"][str(metadata.get("control"))] = float(item["metrics"]["primary"])
        if len(group["controls"]) < 3:
            return
        verdict = placebo_verdict(group["real_primary"], group["controls"])
        group["verdict"] = verdict
        real_item = next(
            row for row in self.history if row.get("iteration") == group["real_iteration"]
        )
        if not isinstance(real_item.get("diagnostics"), dict):
            real_item["diagnostics"] = {}
        real_item["diagnostics"]["placebo_status"] = "complete"
        real_item["diagnostics"]["placebo_verdict"] = verdict["verdict"]
        real_item["diagnostics"]["placebo"] = verdict
        if verdict["verdict"] == "REINTERPRET":
            real_item["research_decision"] = "REINTERPRET"
            real_item["decision"] = "REINTERPRET"
            if self.best_iteration == group["real_iteration"]:
                previous = group["previous_best"]
                previous_iteration = previous.get("iteration")
                previous_config = previous.get("config")
                previous_metrics = previous.get("metrics")
                if (isinstance(previous_iteration, int)
                        and isinstance(previous_config, dict)
                        and isinstance(previous_metrics, dict)):
                    previous_checkpoint = (
                        self.run_dir / "checkpoints" /
                        f"iteration_{previous_iteration:03d}.npz"
                    )
                    self.best_iteration = previous_iteration
                    self.best_config = previous_config
                    self.best_score = float(previous_metrics["primary"])
                    self.best_checkpoint = previous_checkpoint
                    _write_json(self.artifacts_dir / "best_config.json", previous_config)
                    _write_json(self.artifacts_dir / "best_metrics.json", previous_metrics)
                    shutil.copy2(previous_checkpoint, self.artifacts_dir / "best_model.npz")
            for node in self.tree.nodes:
                if node.get("node_id") == node_id_for(group["real_iteration"]):
                    node["decision"] = "REINTERPRET"
                    break
            _write_json(self.run_dir / "tree_snapshot.json", self.tree.snapshot())
        _write_json(self.run_dir / f"placebo_{group['real_iteration']:03d}.json", group)
        _write_json(
            self.run_dir / f"iteration_{group['real_iteration']:03d}.json",
            real_item,
        )
        with (self.run_dir / "diagnostic_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(group, ensure_ascii=False, default=_json_default) + "\n")
        _write_json(self.run_dir / "research_memory.json", build_structured_research_memory(self.history))
        if verdict["verdict"] == "KEEP_CANDIDATE":
            previous_config = group["previous_best"].get("config")
            self._maybe_schedule_confirmation(real_item, previous_config)

    def _update_convergence_streak(self, score: Any, global_best_before: Any, *,
                                   expanded_global_best: bool) -> None:
        """Track how long the search has gone without moving the global best forward.

        A real improvement clears the streak whichever parent produced it, because a new
        global best is progress even when it came from an explored side branch. Only a
        flat attempt against the leading node counts as evidence of a stall; expanding a
        weaker branch is a deliberate search decision, and unusable results say nothing,
        so both leave the streak alone.
        """
        candidate, best = _as_float(score), _as_float(global_best_before)
        if candidate is None or best is None:
            return
        epsilon = float(self.project["run_limits"]["convergence_epsilon"])
        if candidate - best > epsilon:
            self.convergence_streak = 0
        elif expanded_global_best:
            self.convergence_streak += 1

    def _converged(self) -> bool:
        return self.convergence_streak >= int(self.project["run_limits"]["convergence_rounds"])

    def _record_competition_convergence(self) -> None:
        if self.competition_converged_at is not None or not self.history:
            return
        latest_research = next(
            (row for row in reversed(self.history)
             if not self._is_control(row["config"])),
            self.history[-1],
        )
        self.competition_converged_at = int(latest_research["iteration"])
        self.competition_best_at_convergence = {
            "iteration": self.best_iteration,
            "primary": self.best_score,
        }

    def _submission_candidates(self) -> list[dict[str, Any]]:
        """Separate scientific confidence from validation-based submission eligibility."""
        rows: list[dict[str, Any]] = []
        for item in self.history:
            if item.get("status") != "success" or self._is_control(item["config"]):
                continue
            selection = item.get("candidate_selection")
            pattern = (
                selection.get("retrieved_pattern")
                if isinstance(selection, dict) else None
            )
            if not isinstance(pattern, dict):
                pattern = {}
            diagnostics = (
                item.get("diagnostics")
                if isinstance(item.get("diagnostics"), dict) else {}
            )
            research = str(item.get("research_decision", "UNKNOWN"))
            scientific = (
                diagnostics.get("scientific_status")
                or pattern.get("scientific_verdict")
                or {
                    "REFERENCE": "REFERENCE",
                    "KEEP_CANDIDATE": "PROMISING",
                    "ENSEMBLE_ONLY": "PROMISING",
                    "STOP_DIRECTION": "NOT_CONFIRMED",
                    "REJECT": "REJECTED",
                    "REINTERPRET": "REINTERPRETED",
                }.get(research, "NOT_CONFIRMED")
            )
            competition = (
                diagnostics.get("competition_status")
                or pattern.get("competition_status")
            )
            if competition is None:
                competition = (
                    "ELIGIBLE"
                    if research in {"REFERENCE", "KEEP_CANDIDATE", "ENSEMBLE_ONLY"}
                    else "RESEARCH_ONLY"
                )
            eligible = (
                competition == "ELIGIBLE"
                and research != "REINTERPRET"
                and item.get("decision") != "REINTERPRET"
            )
            rows.append({
                "iteration": item["iteration"],
                "primary": float(item["metrics"]["primary"]),
                "scientific_status": scientific,
                "competition_status": competition,
                "submission_status": "ALLOW" if eligible else "EXCLUDE",
                "research_decision": research,
                "selected": bool(eligible and item["iteration"] == self.best_iteration),
                "config": item["config"],
            })
        return sorted(rows, key=lambda row: (-row["primary"], row["iteration"]))

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
            selection = self.tree_policy.select(self.history, remaining)
        except RuntimeError:
            self._pending_parent_selection = None
            return None
        self._pending_parent_selection = selection.as_dict()
        return selection.parent

    def _capture_researcher_accounting(self, researcher: Any, *, failed: bool) -> None:
        attempts = getattr(researcher, "last_attempts", 0)
        try:
            attempts = max(0, int(attempts))
        except (TypeError, ValueError):
            attempts = 0
        if attempts == 0:
            return
        usage = getattr(researcher, "last_token_usage", {})
        if isinstance(usage, dict):
            for key in self.llm_token_usage:
                try:
                    self.llm_token_usage[key] += max(0, int(usage.get(key, 0) or 0))
                except (TypeError, ValueError):
                    continue
        self.llm_requests += attempts
        if failed:
            self.llm_failures += 1

    def _propose(self, researcher, parent: ExperimentParent) -> tuple[Any, Any, tuple[str, str] | None]:
        """Resolve one legal, non-duplicate candidate, or report why the search stopped."""
        set_context = getattr(researcher, "set_run_context", None)
        if callable(set_context):
            set_context(dict(self._research_context))
        last_problem: tuple[str, str] | None = None
        for _ in range(MAX_PROPOSAL_RESOLUTION_ATTEMPTS):
            failed = False
            try:
                proposal = researcher.propose(parent.config, self.history)
                selection = getattr(researcher, "last_selection", None)
                self._pending_candidate_selection = (
                    dict(selection) if isinstance(selection, dict) else {
                        "selected_family": "llm_generated",
                        "selected_score": None,
                        "criteria": "LLM evaluated structured validation memory and legal search space",
                        "ranked_candidates": [],
                    }
                )
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
                candidate = apply_changes(parent.config, proposal.changes)
            except (KeyError, TypeError, ValueError) as exc:
                last_problem = ("search_exhausted", f"invalid proposal: {type(exc).__name__}: {exc}")
                continue
            selection = (
                self._pending_candidate_selection
                if isinstance(self._pending_candidate_selection, dict) else {}
            )
            family = str(selection.get("selected_family") or "unknown")
            expected_skill = self.skill_registry.primary_for_candidate(
                family, proposal.changes
            )
            try:
                self.skill_registry.require(expected_skill)
            except ValueError as exc:
                last_problem = ("missing_capability", str(exc))
                continue
            selected_skill = selection.get("selected_skill")
            if selected_skill not in (None, expected_skill):
                last_problem = (
                    "invalid_skill_binding",
                    f"planner selected {selected_skill!r}, expected {expected_skill!r}",
                )
                continue
            selection["selected_skill"] = expected_skill
            self._pending_candidate_selection = selection
            if is_duplicate_config(candidate, self.history):
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
        finalize_test: bool = True,
        research_after_convergence: bool = False,
        auto_confirm: bool = False,
    ) -> dict[str, Any]:
        limits = self.project["run_limits"]
        cap = self._iteration_cap(max_iterations)
        budget_seconds = float(limits["max_wall_clock_hours"]) * 3600.0
        reserve = self._experiment_cost_seconds()
        self.auto_confirm = bool(auto_confirm)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run log: {self.run_dir}", flush=True)
        planner = getattr(self.researcher, "planner", None)
        memory_mode = getattr(planner, "memory_mode", None)
        _write_json(self.run_dir / "run_meta.json", {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": self.project["benchmark"],
            "limits": limits,
            "max_total_experiments": cap,
            "planner_memory_mode": memory_mode,
            "researcher": type(self.researcher).__name__,
            "research_after_convergence": bool(research_after_convergence),
            "auto_confirm": self.auto_confirm,
            "skill_registry_version": SKILL_REGISTRY_VERSION,
            "available_skills": [
                row["skill_id"] for row in self.skill_registry.catalog()
            ],
        })
        baseline = Proposal("Reproduce the official FM baseline.",
                            "A verified baseline anchors every subsequent comparison.", {}, "system")
        stop_reason, stop_detail = "max_iterations", None
        for iteration in range(cap):
            if iteration > 0 and not self._diagnostic_queue and self._converged():
                self._record_competition_convergence()
                if not research_after_convergence:
                    stop_reason = "converged"
                    break
            next_reserve = reserve
            if self.auto_confirm and self._confirmation_queue:
                next_reserve *= max(
                    1, self._confirmation_queue[0].estimated_training_runs
                )
            blocked = self._budget_block(budget_seconds, next_reserve)
            if blocked is not None:
                stop_reason = blocked
                break
            if iteration == 0:
                self._execute(0, self.best_config, baseline, None)
                if self.best_checkpoint is None:
                    stop_reason = "baseline_failed"
                    break
                continue
            if self._diagnostic_queue:
                candidate, proposal, parent, metadata = self._diagnostic_queue.pop(0)
                self._pending_parent_selection = {
                    **parent.as_record(),
                    "priority": None,
                    "selection_reason": "automatic_placebo_control",
                }
                self._pending_candidate_selection = {
                    "selected_family": "placebo_control",
                    "selected_score": None,
                    "criteria": "mandatory attribution control",
                    "ranked_candidates": [],
                }
                self._pending_diagnostic = metadata
                self._execute(iteration, candidate, proposal, parent)
                continue
            if self.auto_confirm and self._confirmation_queue:
                self._execute_confirmation(
                    iteration, self._confirmation_queue.pop(0)
                )
                continue
            parent = self._select_parent()
            if parent is None:
                stop_reason, stop_detail = "search_exhausted", "no expandable parent node"
                break
            self._research_context = {
                "remaining_iterations": cap - iteration,
                "remaining_seconds": max(0.0, budget_seconds - self._elapsed()),
                "estimated_next_experiment_seconds": reserve,
            }
            proposal, candidate, failure = self._propose(self.researcher, parent)
            if failure is not None and not isinstance(self.researcher, DeterministicResearcher):
                self.llm_fallbacks.append({
                    "iteration": iteration,
                    "reason": failure[1],
                    "provider_error": getattr(self.researcher, "last_error", None),
                })
                prior_evidence = getattr(self.researcher, "prior_evidence", None)
                if prior_evidence is None:
                    planner = getattr(self.researcher, "planner", None)
                    prior_evidence = getattr(planner, "prior_evidence", None)
                proposal, candidate, failure = self._propose(
                    DeterministicResearcher(prior_evidence=prior_evidence), parent
                )
            if failure is not None:
                stop_reason, stop_detail = failure
                break
            self._execute(iteration, candidate, proposal, parent)
        if self._converged():
            # The streak may reach the threshold on the final allowed iteration,
            # leaving no next loop turn in which to record the official point.
            self._record_competition_convergence()
        final_test = None
        if finalize_test and self.best_checkpoint is not None:
            final_test = self.runner.finalize(self.best_config, self.best_checkpoint,
                                              self.submissions_dir / "final.csv")
            _write_json(self.artifacts_dir / "final_test_metrics.json", final_test)
        executed = len(self.history) + len(self.confirmations)
        elapsed = self._elapsed()
        submission_candidates = self._submission_candidates()
        _write_json(self.run_dir / "submission_candidates.json", submission_candidates)
        memory_influenced = sum(
            bool(item.get("candidate_selection", {}).get("memory_changed_choice"))
            for item in self.history
            if isinstance(item.get("candidate_selection"), dict)
        )
        summary = {"stop_reason": stop_reason, "stop_detail": stop_detail,
                   "iterations": executed,
                   "total_experiments": executed,
                   "candidate_experiments": max(0, executed - 1),
                   "discovery_actions": sum(
                       not self._is_control(row["config"])
                       for row in self.history[1:]
                   ),
                   "control_actions": sum(
                       self._is_control(row["config"])
                       for row in self.history
                   ),
                   "confirmation_actions": len(self.confirmations),
                   "confirmation_training_runs": sum(
                       int(row.get("result", {}).get("training_runs", 0) or 0)
                       for row in self.confirmations
                       if isinstance(row.get("result"), dict)
                   ),
                   "pending_confirmations": [
                       action.as_dict() for action in self._confirmation_queue
                   ],
                   "best_primary": None if self.best_score == float("-inf") else self.best_score,
                   "best_iteration": self.best_iteration,
                   "manual_interventions": self.interventions.count,
                   "avoidable_manual_interventions": self.interventions.avoidable_count,
                   "intervention_log": str(self.interventions.path),
                   "planner_memory_mode": memory_mode,
                   "memory_influenced_selections": memory_influenced,
                   "final_test_metrics": final_test,
                   "convergence_streak": self.convergence_streak,
                   "competition_converged": self.competition_converged_at is not None,
                   "competition_converged_at": self.competition_converged_at,
                   "competition_best_at_convergence": self.competition_best_at_convergence,
                   "research_after_convergence": bool(research_after_convergence),
                   "auto_confirm": self.auto_confirm,
                   "skill_registry_version": SKILL_REGISTRY_VERSION,
                   "available_skills": [
                       row["skill_id"] for row in self.skill_registry.catalog()
                   ],
                   "research_exhausted": stop_reason == "search_exhausted",
                   "submission_candidate_count": sum(
                       row["submission_status"] == "ALLOW"
                       for row in submission_candidates
                   ),
                   "submission_candidates": str(
                       self.run_dir / "submission_candidates.json"
                   ),
                   "elapsed_seconds": elapsed,
                   "remaining_seconds": max(0.0, budget_seconds - elapsed),
                   "wall_clock_seconds": elapsed,
                   "llm_requests": self.llm_requests,
                   "llm_failures": self.llm_failures,
                   "llm_fallbacks": list(self.llm_fallbacks),
                   "llm_tokens": dict(self.llm_token_usage),
                   "limits": {"max_total_experiments": cap,
                              "official_max_iterations": int(limits["max_iterations"]),
                              "max_wall_clock_hours": limits["max_wall_clock_hours"],
                              "wall_clock_budget_seconds": budget_seconds,
                              "convergence_epsilon": float(limits["convergence_epsilon"]),
                              "convergence_rounds": int(limits["convergence_rounds"]),
                              "experiment_cost_seconds": reserve,
                              "max_active_branches": self.tree_policy.config.max_active_branches}}
        _write_json(self.run_dir / "summary.json", summary)
        print(f"\nStopped: {stop_reason} | best_primary={summary['best_primary']}", flush=True)
        return summary
