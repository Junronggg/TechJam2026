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
        "iteration": item.get("iteration"),
        "changes": changes,
        "verdict": verdict,
        "primary": _validation_primary(item.get("metrics")),
        "delta": None,
        "next_test": _short_text(critique.get("next_test")),
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
        config = item.get("config") if isinstance(item.get("config"), dict) else {}
        changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
        best = {
            "iteration": item.get("iteration"),
            "validation_primary": primary,
            "model": config.get("model"),
            "training_objective": config.get("training_objective"),
            "changes": changes,
        }
    return best


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

    return {
        "baseline_reference": baseline,
        "best_observed": _best_observed(rows),
        "counts": counts,
        "promising": buckets["promising"][-LESSON_LIMIT:],
        "uncertain": buckets["uncertain"][-LESSON_LIMIT:],
        "negative": buckets["negative"][-LESSON_LIMIT:],
        "failed": buckets["failed"][-LESSON_LIMIT:],
        "tried_signatures": signatures[-SIGNATURE_LIMIT:],
    }
