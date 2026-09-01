from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import apply_changes, experiment_key, normalize_config, validate_config


VALIDATION_KEYS = ("GAUC", "nDCG@5", "primary")
CRITIQUE_KEYS = (
    "observation", "interpretation", "confidence", "verdict", "delta",
    "meaningful_improvement", "next_test", "reasons", "hypothesis_status",
    "evidence_strength", "seed_count", "reflection_triggered", "reflection_reasons",
    "general_lesson", "next_questions",
    "bottleneck", "recommended_strategy_ids", "failure_category",
)


def _validation_metrics(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, float] = {}
    for key in VALIDATION_KEYS:
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result if "primary" in result else None


def _research_metrics(value: Any) -> dict[str, float] | None:
    """Keep safe execution metadata alongside validation metrics for budgeting."""
    result = _validation_metrics(value)
    if result is None or not isinstance(value, dict):
        return result
    for key in ("runtime_seconds", "best_epoch", "ensemble_size"):
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0:
            result[key] = number
    return result


def load_research_state(path: Path) -> dict[str, Any]:
    """Load the committed validation-selected incumbent, never test results."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    incumbent = payload.get("incumbent")
    if not isinstance(incumbent, dict):
        raise ValueError(f"{path} does not contain an incumbent")
    incumbent_config = incumbent.get("config")
    if not isinstance(incumbent_config, dict):
        raise ValueError(f"{path} incumbent does not contain a config")
    incumbent_config = normalize_config(incumbent_config)
    validate_config(incumbent_config)
    incumbent_metrics = _validation_metrics(incumbent.get("validation_metrics"))
    if incumbent_metrics is None:
        raise ValueError(f"{path} incumbent lacks validation Primary")
    baseline_path = path.parent / "experiment.json"
    baseline = normalize_config(json.loads(baseline_path.read_text(encoding="utf-8")))
    validate_config(baseline)
    findings: list[dict[str, Any]] = []
    for position, finding in enumerate(payload.get("findings", [])):
        if not isinstance(finding, dict):
            continue
        changes = finding.get("changes")
        metrics = _validation_metrics(finding.get("validation_metrics"))
        if not isinstance(changes, dict) or metrics is None:
            continue
        config = apply_changes(baseline, changes)
        findings.append({
            "evidence_id": str(
                finding.get("evidence_id") or f"committed_finding_{position:03d}"
            ),
            "description": str(finding.get("description") or "Committed finding"),
            "changes": changes,
            "config": config,
            "validation_metrics": metrics,
            "verdict": str(finding.get("verdict") or "reject"),
            "seed_count": int(finding.get("seed_count") or 1),
        })
    return {
        "incumbent": {
            "config": incumbent_config,
            "validation_metrics": incumbent_metrics,
            "evidence_id": str(incumbent.get("evidence_id") or "incumbent_reference"),
            "description": str(incumbent.get("description") or "Validated incumbent"),
        },
        "findings": findings,
    }


def load_durable_incumbent(
    state: dict[str, Any], artifacts_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    """Prefer a newer validation-best artifact when its config and checkpoint agree."""
    committed = state["incumbent"]
    config_path = artifacts_dir / "best_config.json"
    metrics_path = artifacts_dir / "best_metrics.json"
    checkpoint_path = artifacts_dir / "best_model.npz"
    if not all(path.is_file() for path in (config_path, metrics_path, checkpoint_path)):
        return committed, None
    try:
        config = normalize_config(json.loads(config_path.read_text(encoding="utf-8")))
        validate_config(config)
        metrics = _research_metrics(json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return committed, None
    if metrics is None:
        return committed, None
    committed_primary = float(committed["validation_metrics"]["primary"])
    if metrics["primary"] + 1e-12 < committed_primary:
        return committed, None
    return ({
        "config": config,
        "validation_metrics": metrics,
        "evidence_id": "durable_validation_best",
        "description": "Highest persisted validation-only checkpoint.",
    }, checkpoint_path)


def _sanitize_record(value: Any, evidence_id: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    config = value.get("config")
    if not isinstance(config, dict):
        return None
    config = normalize_config(config)
    try:
        validate_config(config)
    except (KeyError, TypeError, ValueError):
        return None
    metrics = _research_metrics(value.get("metrics"))
    status = value.get("status")
    if metrics is None and status in (None, "success", "ok"):
        return None
    critique_value = value.get("critique")
    critique = (
        {key: critique_value[key] for key in CRITIQUE_KEYS if key in critique_value}
        if isinstance(critique_value, dict) else {}
    )
    if isinstance(critique_value, dict):
        deltas = _validation_metrics(critique_value.get("metric_deltas"))
        if deltas:
            critique["metric_deltas"] = deltas
    record = {
        "evidence_id": evidence_id,
        "iteration": value.get("iteration"),
        "historical": True,
        "hypothesis": value.get("hypothesis"),
        "reason": value.get("reason"),
        "changes": value.get("changes") if isinstance(value.get("changes"), dict) else {},
        "decision": value.get("decision"),
        "source": value.get("source"),
        "status": status,
        "metrics": metrics,
        "critique": critique,
        "config": config,
        "error": value.get("error") if status not in (None, "success", "ok") else None,
    }
    try:
        execution_seconds = float(value.get("execution_seconds"))
    except (TypeError, ValueError):
        execution_seconds = -1.0
    if math.isfinite(execution_seconds) and execution_seconds >= 0:
        record["execution_seconds"] = execution_seconds
    return record


def _archive_changes(config: dict[str, Any]) -> dict[str, Any]:
    """Describe a standalone validation config compactly for Planner memory."""
    changes: dict[str, Any] = {
        "model": config.get("model"),
        "training_objective": config.get("training_objective"),
    }
    if config.get("model") == "custom" and config.get("code_branch"):
        changes["code_branch"] = config["code_branch"]
    hp = config.get("hyperparameters") if isinstance(config.get("hyperparameters"), dict) else {}
    defaults = {
        "embedding_dim": 16,
        "learning_rate": 0.001,
        "ensemble_size": 1,
        "ensemble_seed_set": "sequential",
        "negatives_per_positive": 1,
        "negative_sampling_strategy": "random",
    }
    for key, default in defaults.items():
        if key in hp and hp[key] != default:
            changes[key] = hp[key]
    features = config.get("features") if isinstance(config.get("features"), dict) else {}
    changes.update({key: True for key, enabled in features.items() if enabled})
    return changes


def load_prior_history(
    logs_dir: Path,
    state: dict[str, Any] | None = None,
    *,
    validation_dir: Path | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Load bounded, validation-only evidence from completed agent runs.

    Duplicate scientific configurations collapse to their strongest measured
    record. This prevents fresh runs from paying to rediscover old experiments.
    """
    records: list[dict[str, Any]] = []
    for history_path in sorted(logs_dir.glob("run_*/experiment_history.jsonl")):
        try:
            lines = history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for position, line in enumerate(lines):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            iteration = raw.get("iteration") if isinstance(raw, dict) else position
            evidence_id = f"prior_{history_path.parent.name}_{iteration}"
            record = _sanitize_record(raw, evidence_id)
            if record is not None:
                records.append(record)

    for finding in (state or {}).get("findings", []):
        verdict = finding["verdict"]
        records.append({
            "evidence_id": finding["evidence_id"],
            "iteration": None,
            "historical": True,
            "hypothesis": finding["description"],
            "reason": "Committed validation-only research evidence.",
            "changes": finding["changes"],
            "decision": "KEEP" if verdict == "promote" else "REJECT",
            "source": "committed_research_state",
            "status": "success",
            "metrics": finding["validation_metrics"],
            "critique": {
                "observation": finding["description"],
                "interpretation": "Persisted validation-only finding.",
                "confidence": "medium",
                "verdict": verdict,
                "hypothesis_status": (
                    "supported" if verdict == "promote" else
                    "inconclusive" if verdict == "noise" else "unsupported"
                ),
                "evidence_strength": "committed_validation_run",
                "seed_count": finding["seed_count"],
            },
            "config": finding["config"],
            "error": None,
        })

    if validation_dir is not None:
        for result_path in sorted(validation_dir.glob("*.json")):
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict) or raw.get("split") != "validation":
                continue
            normalized = {
                **raw,
                "status": "success",
                "decision": "REJECT",
                "source": "validation_archive",
                "hypothesis": f"Archived validation candidate {result_path.stem}.",
                "changes": raw.get("changes") or _archive_changes(raw.get("config", {})),
                "critique": raw.get("critique") or {
                    "verdict": "reject",
                    "hypothesis_status": "unsupported",
                    "evidence_strength": "archived_validation_run",
                    "seed_count": 1,
                    "observation": "Archived candidate did not beat the durable incumbent.",
                },
            }
            record = _sanitize_record(
                normalized, f"validation_archive_{result_path.stem}"
            )
            if record is not None:
                records.append(record)

    incumbent = (state or {}).get("incumbent")
    if isinstance(incumbent, dict):
        metrics = incumbent["validation_metrics"]
        records.append({
            "evidence_id": incumbent["evidence_id"],
            "iteration": None,
            "historical": True,
            "hypothesis": incumbent["description"],
            "reason": "Committed validation-selected research state.",
            "changes": {},
            "decision": "KEEP",
            "source": "validated_incumbent",
            "status": "success",
            "metrics": metrics,
            "critique": {
                "observation": (
                    f"Validation Primary={metrics['primary']:.6f}, "
                    f"GAUC={metrics.get('GAUC', float('nan')):.6f}, "
                    f"nDCG@5={metrics.get('nDCG@5', float('nan')):.6f}."
                ),
                "interpretation": "Current validation-selected incumbent; continue beyond it.",
                "confidence": "medium",
                "verdict": "promote",
                "hypothesis_status": "promising_unreplicated",
                "evidence_strength": "validation_subset_selection",
                "seed_count": 2,
            },
            "config": incumbent["config"],
            "error": None,
        })

    by_config: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        key = experiment_key(record["config"])
        if key not in by_config:
            order.append(key)
            by_config[key] = record
            continue
        old_primary = ((by_config[key].get("metrics") or {}).get("primary"))
        new_primary = ((record.get("metrics") or {}).get("primary"))
        old_metric_count = len(by_config[key].get("metrics") or {})
        new_metric_count = len(record.get("metrics") or {})
        if new_primary is not None and (
            old_primary is None or new_primary > old_primary
            or (new_primary == old_primary and new_metric_count >= old_metric_count)
        ):
            by_config[key] = record
    bounded_limit = max(0, int(limit))
    if bounded_limit == 0:
        return []
    return [by_config[key] for key in order][-bounded_limit:]
