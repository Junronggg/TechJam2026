from __future__ import annotations

import hashlib
import math
from typing import Any

from .config import experiment_key

LESSON_LIMIT = 5
SIGNATURE_LIMIT = 10
PLANNER_RECENT_HISTORY = 5
VALIDATION_METRIC_KEYS = ("GAUC", "nDCG@5", "primary")
VERDICT_CATEGORIES = {
    "promote": "promising",
    "noise": "uncertain",
    "reject": "negative",
    "failed": "failed",
}
MAX_HYPOTHESIS_CHARS = 160


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validation_primary(metrics: Any) -> float | None:
    if not isinstance(metrics, dict) or "primary" not in metrics:
        return None
    if not _finite(metrics["primary"]):
        return None
    return float(metrics["primary"])


def _validation_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key in VALIDATION_METRIC_KEYS:
        if key in metrics and _finite(metrics[key]):
            cleaned[key] = float(metrics[key])
    return cleaned


def _verdict(item: dict[str, Any]) -> str | None:
    critique = item.get("critique")
    if isinstance(critique, dict):
        value = critique.get("verdict")
        if value in VERDICT_CATEGORIES:
            return str(value)
    status = item.get("status")
    if status not in {None, "success", "ok"}:
        return "failed"
    return None


def _short_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    if len(text) > MAX_HYPOTHESIS_CHARS:
        return text[: MAX_HYPOTHESIS_CHARS - 1] + "…"
    return text


def _error_type(item: dict[str, Any]) -> str | None:
    error = item.get("error")
    if not isinstance(error, dict):
        return None
    value = error.get("type")
    if isinstance(value, str) and value.strip():
        return value.strip()[:80]
    return None


def _lesson(item: dict[str, Any], verdict: str) -> dict[str, Any]:
    critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
    changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
    lesson: dict[str, Any] = {
        "evidence_id": (
            item.get("evidence_id") if isinstance(item.get("evidence_id"), str)
            and item.get("evidence_id") else (
                f"iteration_{item['iteration']:03d}"
                if isinstance(item.get("iteration"), int) else "historical_record"
            )
        ),
        "iteration": item.get("iteration"),
        "changes": changes,
        "verdict": verdict,
        "primary": _validation_primary(item.get("metrics")),
        "metrics": _validation_metrics(item.get("metrics")),
        "delta": None,
        "metric_deltas": _validation_metrics(critique.get("metric_deltas")),
        "hypothesis_status": critique.get(
            "hypothesis_status",
            "supported" if verdict == "promote" else (
                "unsupported" if verdict == "reject" else "inconclusive"
            ),
        ),
        "evidence_strength": critique.get("evidence_strength", "single_seed"),
        "seed_count": critique.get("seed_count", 1),
        "decision": item.get("decision"),
        "source": item.get("source"),
        "next_test": _short_text(critique.get("next_test")),
        "general_lesson": _short_text(critique.get("general_lesson")),
        "reflection_triggered": bool(critique.get("reflection_triggered", False)),
        "bottleneck": critique.get("bottleneck"),
        "recommended_strategy_ids": critique.get("recommended_strategy_ids", []),
        "failure_category": critique.get("failure_category"),
    }
    if _finite(critique.get("delta")):
        lesson["delta"] = float(critique["delta"])
    hypothesis = _short_text(item.get("hypothesis"))
    if hypothesis:
        lesson["hypothesis"] = hypothesis
    if verdict == "failed":
        error_type = _error_type(item)
        if error_type:
            lesson["error_type"] = error_type
    return lesson


def _signature(item: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
    key = experiment_key(config)
    return {
        "iteration": item.get("iteration"),
        "model": config.get("model"),
        "training_objective": config.get("training_objective"),
        "changes": changes,
        "key_hash": hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
    }


def _is_baseline_reference(item: dict[str, Any], position: int) -> bool:
    config = item.get("config")
    changes = item.get("changes")
    if not isinstance(config, dict) or changes != {}:
        return False
    if not isinstance(config.get("model"), str) or not isinstance(
        config.get("training_objective"), str
    ):
        return False
    iteration = item.get("iteration")
    return iteration == 0 or (position == 0 and iteration is None)


def _baseline_reference(item: dict[str, Any]) -> dict[str, Any]:
    config = item["config"]
    return {
        "iteration": item.get("iteration"),
        "validation_primary": _validation_primary(item.get("metrics")),
        "model": config.get("model"),
        "training_objective": config.get("training_objective"),
    }


def _best_observed(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for item in history:
        if item.get("decision") != "KEEP":
            continue
        primary = _validation_primary(item.get("metrics"))
        if primary is None:
            continue
        if best is not None and primary <= best["validation_primary"]:
            continue
        config = item.get("config") if isinstance(item.get("config"), dict) else {}
        changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        best = {
            "iteration": item.get("iteration"),
            "validation_primary": primary,
            "validation_metrics": _validation_metrics(item.get("metrics")),
            "model": config.get("model"),
            "training_objective": config.get("training_objective"),
            "changes": changes,
            "critic_verdict": critique.get("verdict"),
            "hypothesis_status": critique.get("hypothesis_status", "inconclusive"),
            "evidence_strength": critique.get("evidence_strength", "single_seed"),
            "seed_count": critique.get("seed_count", 1),
        }
    return best


def _research_findings(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact measured findings; never upgrades a single run into a confirmed claim."""
    findings: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item.get("changes"), dict) or not item["changes"]:
            continue
        metrics = _validation_metrics(item.get("metrics"))
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        if not metrics and critique.get("verdict") != "failed":
            continue
        iteration = item.get("iteration")
        supplied_evidence_id = item.get("evidence_id")
        verdict = critique.get("verdict") or _verdict(item)
        seed_count = critique.get("seed_count", 1)
        status = critique.get(
            "hypothesis_status",
            "supported" if verdict == "promote" else (
                "unsupported" if verdict == "reject" else "inconclusive"
            ),
        )
        if status == "supported" and (not isinstance(seed_count, int) or seed_count < 2):
            status = "promising_unreplicated"
        findings.append({
            "evidence_id": (
                supplied_evidence_id if isinstance(supplied_evidence_id, str)
                and supplied_evidence_id else (
                    f"iteration_{iteration:03d}"
                    if isinstance(iteration, int) else "historical_record"
                )
            ),
            "changes": item["changes"],
            "metrics": metrics,
            "metric_deltas": _validation_metrics(critique.get("metric_deltas")),
            "critic_verdict": verdict,
            "evidence_status": status,
            "seed_count": seed_count,
        })
    return findings[-LESSON_LIMIT:]


def collect_tried_keys(history: list[dict[str, Any]] | None) -> list[str]:
    """Insertion-ordered unique scientific configs via existing experiment_key()."""
    keys: list[str] = []
    seen: set[str] = set()
    for item in history or []:
        config = item.get("config") if isinstance(item, dict) else None
        if not isinstance(config, dict):
            continue
        key = experiment_key(config)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def is_duplicate_config(config: dict[str, Any], history: list[dict[str, Any]] | None) -> bool:
    """True when config matches a previous experiment, ignoring hypothesis wording."""
    if not isinstance(config, dict):
        return False
    return experiment_key(config) in set(collect_tried_keys(history))


def build_memory_summary(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Derive compact planner memory. JSONL/history remains the evidence source."""
    rows = [item for item in (history or []) if isinstance(item, dict)]
    buckets: dict[str, list[dict[str, Any]]] = {
        "promising": [],
        "uncertain": [],
        "negative": [],
        "failed": [],
    }
    signatures: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    baseline: dict[str, Any] | None = None
    counts = {
        "total": len(rows),
        "baseline_reference": 0,
        "candidate_verdicts": 0,
        "unclassified_candidates": 0,
        "promising": 0,
        "uncertain": 0,
        "negative": 0,
        "failed": 0,
    }

    for position, item in enumerate(rows):
        is_baseline = baseline is None and _is_baseline_reference(item, position)
        if is_baseline:
            baseline = _baseline_reference(item)
            counts["baseline_reference"] += 1
        else:
            verdict = _verdict(item)
            if verdict is not None:
                category = VERDICT_CATEGORIES[verdict]
                counts["candidate_verdicts"] += 1
                counts[category] += 1
                buckets[category].append(_lesson(item, verdict))
            else:
                counts["unclassified_candidates"] += 1
        config = item.get("config") if isinstance(item.get("config"), dict) else None
        if config is None:
            continue
        key = experiment_key(config)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        signatures.append(_signature(item, config))

    findings = _research_findings(rows)
    return {
        "baseline_reference": baseline,
        "best_observed": _best_observed(rows),
        "counts": counts,
        "promising": buckets["promising"][-LESSON_LIMIT:],
        "uncertain": buckets["uncertain"][-LESSON_LIMIT:],
        "negative": buckets["negative"][-LESSON_LIMIT:],
        "failed": buckets["failed"][-LESSON_LIMIT:],
        "research_findings": findings,
        "confirmed_insights": [
            item for item in findings
            if (item["evidence_status"] == "supported" and
                isinstance(item["seed_count"], int) and item["seed_count"] >= 2)
        ],
        "tried_signatures": signatures[-SIGNATURE_LIMIT:],
    }
