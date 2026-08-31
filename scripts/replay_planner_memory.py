from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from techjam_agent.config import (  # noqa: E402
    ALLOWED_VALUES,
    FEATURE_KEYS,
    LIGHTGBM_KEYS,
    MODELS,
    OBJECTIVES,
    apply_changes,
    experiment_key,
    validate_config,
)
from techjam_agent.critic import review  # noqa: E402
from techjam_agent.evidence import (  # noqa: E402
    attach_feasibility_evidence,
    build_feasibility_evidence,
    build_generated_family_policies,
    merge_generated_policies,
)
from techjam_agent.experiment_planner import (  # noqa: E402
    MEMORY_MODES,
    choose_ranked,
    rank_candidates,
)


VALIDATION_KEYS = ("GAUC", "nDCG@5", "primary")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def normalize_logged_config(
    logged: Any,
    template: dict[str, Any],
) -> dict[str, Any] | None:
    """Map older log schemas onto today's complete, validated config schema."""
    if not isinstance(logged, dict):
        return None
    normalized = copy.deepcopy(template)
    if logged.get("model") in MODELS:
        normalized["model"] = logged["model"]
    if logged.get("training_objective") in OBJECTIVES:
        normalized["training_objective"] = logged["training_objective"]

    logged_hp = logged.get("hyperparameters")
    if isinstance(logged_hp, dict):
        for key, allowed in ALLOWED_VALUES.items():
            if key in logged_hp and logged_hp[key] in allowed:
                normalized["hyperparameters"][key] = logged_hp[key]

    logged_features = logged.get("features")
    if isinstance(logged_features, dict):
        for key in FEATURE_KEYS:
            if type(logged_features.get(key)) is bool:
                normalized["features"][key] = logged_features[key]

    logged_lgb = logged.get("lightgbm_hyperparameters")
    if isinstance(logged_lgb, dict):
        for key in LIGHTGBM_KEYS:
            if key in logged_lgb:
                normalized["lightgbm_hyperparameters"][key] = logged_lgb[key]

    try:
        validate_config(normalized)
    except (KeyError, TypeError, ValueError):
        return None
    return normalized


def _validation_metrics(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, float] = {}
    for key in VALIDATION_KEYS:
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        result[key] = number
    return result


def build_validation_archive(
    logs_dir: Path,
    template: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Aggregate validation-only outcomes; test summaries are never loaded."""
    observations: dict[str, list[dict[str, Any]]] = {}
    audit = {
        "history_files": 0,
        "rows_seen": 0,
        "successful_validation_rows": 0,
        "invalid_or_legacy_rows_skipped": 0,
    }
    for path in sorted(logs_dir.glob("run_*/experiment_history.jsonl")):
        audit["history_files"] += 1
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            audit["rows_seen"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                audit["invalid_or_legacy_rows_skipped"] += 1
                continue
            if not isinstance(row, dict) or row.get("status") not in {"success", "ok"}:
                continue
            metrics = _validation_metrics(row.get("metrics"))
            config = normalize_logged_config(row.get("config"), template)
            if metrics is None or config is None:
                audit["invalid_or_legacy_rows_skipped"] += 1
                continue
            audit["successful_validation_rows"] += 1
            observations.setdefault(experiment_key(config), []).append({
                "metrics": metrics,
                "source": str(path.relative_to(logs_dir)),
            })

    archive: dict[str, dict[str, Any]] = {}
    for key, rows in observations.items():
        aggregate = {
            metric: float(statistics.median(row["metrics"][metric] for row in rows))
            for metric in VALIDATION_KEYS
        }
        primary_values = [row["metrics"]["primary"] for row in rows]
        archive[key] = {
            "metrics": aggregate,
            "observations": len(rows),
            "primary_min": float(min(primary_values)),
            "primary_max": float(max(primary_values)),
            "sources": sorted({row["source"] for row in rows}),
        }
    audit["unique_normalized_configs"] = len(archive)
    return archive, audit


def replay_mode(
    mode: str,
    initial_config: dict[str, Any],
    archive: dict[str, dict[str, Any]],
    max_steps: int,
    epsilon: float,
    prior_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay supported actions without exposing an outcome before selection."""
    baseline = archive.get(experiment_key(initial_config))
    if baseline is None:
        raise ValueError("the logged archive does not contain the current baseline")

    baseline_metrics = copy.deepcopy(baseline["metrics"])
    history: list[dict[str, Any]] = [{
        "iteration": 0,
        "config": copy.deepcopy(initial_config),
        "changes": {},
        "status": "success",
        "decision": "KEEP",
        "research_decision": "REFERENCE",
        "metrics": baseline_metrics,
        "delta_from_parent": None,
        "candidate_selection": None,
        "critique": review(baseline_metrics, None, epsilon, "success"),
        "diagnostics": {},
    }]
    best_config = copy.deepcopy(initial_config)
    best_primary = float(baseline_metrics["primary"])
    best_iteration = 0
    support_skips = 0

    for iteration in range(1, max_steps + 1):
        ranked = choose_ranked(rank_candidates(
            best_config, history, memory_mode=mode,
            prior_evidence=prior_evidence,
        ))
        supported = []
        for ranked_row in ranked:
            candidate_config = apply_changes(
                best_config, ranked_row.candidate.changes
            )
            outcome = archive.get(experiment_key(candidate_config))
            if outcome is None:
                continue
            supported.append((ranked_row, candidate_config, outcome))
        support_skips += max(0, len(ranked) - len(supported))
        if not supported:
            break

        selected, candidate_config, outcome = supported[0]
        metrics = copy.deepcopy(outcome["metrics"])
        primary = float(metrics["primary"])
        delta = primary - best_primary
        critique = review(
            metrics,
            best_primary,
            epsilon,
            "success",
            history=history,
            changes=selected.candidate.changes,
        )
        decision = "KEEP" if primary > best_primary else "REJECT"
        history.append({
            "iteration": iteration,
            "hypothesis": selected.candidate.hypothesis,
            "reason": selected.candidate.reason,
            "config": candidate_config,
            "changes": selected.candidate.changes,
            "status": "success",
            "decision": decision,
            "research_decision": critique["verdict"].upper(),
            "metrics": metrics,
            "delta_from_parent": delta,
            "candidate_selection": {
                "memory_mode": mode,
                "selected_family": selected.candidate.family,
                "selected_score": float(selected.score),
                "observed_mean_delta": selected.observed_mean_delta,
                "family_trials": selected.family_trials,
                "logged_support": outcome["observations"],
            },
            "critique": critique,
            "diagnostics": {},
        })
        if primary > best_primary:
            best_config = copy.deepcopy(candidate_config)
            best_primary = primary
            best_iteration = iteration

    candidates = history[1:]
    return {
        "mode": mode,
        "best_primary": best_primary,
        "best_iteration": best_iteration,
        "experiments_replayed": len(candidates),
        "unproductive_experiments": sum(
            row["decision"] != "KEEP" for row in candidates
        ),
        "support_filter_skips": support_skips,
        "trajectory": [{
            "iteration": row["iteration"],
            "family": row["candidate_selection"]["selected_family"],
            "changes": row["changes"],
            "primary": row["metrics"]["primary"],
            "delta_from_best_before": row["delta_from_parent"],
            "decision": row["decision"],
            "critic_verdict": row["critique"]["verdict"],
            "logged_support": row["candidate_selection"]["logged_support"],
        } for row in candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay planner memory modes against previously measured validation outcomes."
        )
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=0.002)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    initial_config = _load_json(ROOT / "configs" / "experiment.json")
    prior_evidence = _load_json(ROOT / "configs" / "research_evidence.json")
    evidence_manifest = _load_json(ROOT / "configs" / "evidence_manifest.json")
    generated_policies = build_generated_family_policies(ROOT, evidence_manifest)
    prior_evidence = merge_generated_policies(prior_evidence, generated_policies)
    prior_evidence = attach_feasibility_evidence(
        prior_evidence, build_feasibility_evidence(ROOT, evidence_manifest),
    )
    validate_config(initial_config)
    archive, archive_audit = build_validation_archive(ROOT / "logs", initial_config)
    results = [
        replay_mode(
            mode, initial_config, archive, args.max_steps, args.epsilon,
            prior_evidence,
        )
        for mode in MEMORY_MODES
    ]
    trajectories = {
        result["mode"]: [row["family"] for row in result["trajectory"]]
        for result in results
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "offline_logged_validation_replay",
        "validation_only": True,
        "test_metrics_loaded": False,
        "outcome_visible_only_after_selection": True,
        "support_filter": (
            "Only actions with a logged validation outcome are eligible. Availability, "
            "not the metric value, is checked before selection."
        ),
        "limitations": [
            "Logged runs are reused evidence and are not independent fresh trials.",
            "The replay evaluates planning decisions, not training reproducibility.",
            "Unsupported actions are skipped equally for all memory modes.",
            "The fixed step cap intentionally disables official convergence for this stress test.",
        ],
        "max_steps": args.max_steps,
        "epsilon": args.epsilon,
        "archive_audit": archive_audit,
        "results": results,
        "memory_changed_trajectory": len({tuple(value) for value in trajectories.values()}) > 1,
        "trajectories": trajectories,
    }
    output = ROOT / "artifacts" / "planner_memory_replay.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Replay report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
