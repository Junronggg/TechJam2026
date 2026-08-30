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
PATTERN_LIMIT = 20


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


def distill_research_patterns(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Distill reusable, validation-only research policies by experiment family.

    Raw experiment rows remain the source of truth.  Patterns are compact
    planning aids: successful, failed and reinterpreted branches all contribute,
    while placebo controls and test metrics cannot become positive evidence.
    """
    families: dict[str, dict[str, Any]] = {}
    for item in history or []:
        if not isinstance(item, dict) or item.get("iteration") == 0:
            continue
        selection = item.get("candidate_selection")
        family = selection.get("selected_family") if isinstance(selection, dict) else None
        if not isinstance(family, str) or not family or family == "placebo_control":
            continue
        row = families.setdefault(family, {
            "trials": 0,
            "positive": 0,
            "negative_or_noise": 0,
            "failed": 0,
            "reinterpreted": 0,
            "control_pending": 0,
            "control_passed": 0,
            "low_coverage": 0,
            "slice_or_diversity": 0,
            "deltas": [],
        })
        row["trials"] += 1
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        diagnostics = item.get("diagnostics") if isinstance(item.get("diagnostics"), dict) else {}
        verdict = critique.get("verdict")
        placebo_status = diagnostics.get("placebo_status")
        placebo_result = diagnostics.get("placebo_verdict")
        if placebo_status == "scheduled":
            row["control_pending"] += 1
        elif placebo_status == "complete" and placebo_result == "KEEP_CANDIDATE":
            row["control_passed"] += 1
        coverage = diagnostics.get("feature_coverage", diagnostics.get("coverage"))
        if _finite(coverage) and float(coverage) < 0.01:
            row["low_coverage"] += 1
        if item.get("status") != "success" or verdict == "failed":
            row["failed"] += 1
        elif diagnostics.get("placebo_verdict") == "REINTERPRET":
            row["reinterpreted"] += 1
        elif verdict == "promote":
            row["positive"] += 1
        elif verdict in {"noise", "reject"}:
            row["negative_or_noise"] += 1
        if diagnostics.get("strong_slice_gain") or diagnostics.get("diversity_advantage"):
            row["slice_or_diversity"] += 1
        if _finite(item.get("delta_from_parent")) and item.get("decision") != "CONTROL":
            row["deltas"].append(float(item["delta_from_parent"]))

    patterns: list[dict[str, Any]] = []
    for family, evidence in sorted(families.items()):
        deltas = evidence.pop("deltas")
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        best_delta = max(deltas) if deltas else None
        if ((evidence["control_pending"] > 0 or evidence["low_coverage"] > 0)
                and evidence["reinterpreted"] == 0
                and evidence["control_passed"] == 0):
            policy = "retest_with_control"
            solution = "The apparent gain is not yet attributable to the real signal."
            template = "Run matched constant, shuffled and same-cardinality controls before promotion."
        elif (evidence["negative_or_noise"] + evidence["failed"]
                + evidence["reinterpreted"] >= 2
                and evidence["slice_or_diversity"] == 0):
            policy = "stop_direction"
            solution = "Do not repeat equivalent variants without a distinct information source."
            template = "Select a different family; reopen only after new evidence changes the mechanism."
        elif evidence["slice_or_diversity"] > 0 and not (
            mean_delta is not None and mean_delta > 0
        ):
            policy = "ensemble_only"
            solution = "Treat the family as conditionally complementary, not as a global replacement."
            template = "Run fixed-slice and error-recovery checks, then one predeclared gate or blend."
        elif evidence["positive"] > 0 and mean_delta is not None and mean_delta > 0:
            policy = "exploit_with_confirmation"
            solution = "The family has positive validation evidence worth confirming."
            template = "Confirm with rolling folds or paired seeds before promotion; avoid fine-grid tuning."
        else:
            policy = "gather_evidence"
            solution = "Evidence is insufficient for a directional conclusion."
            template = "Run one cheap single-variable comparison, then attribute with controls if the gain is small."
        patterns.append({
            "family": family,
            "task_description": f"Evaluate the {family} experiment family.",
            "solution_description": solution,
            "thought_template": template,
            "policy": policy,
            "evidence": {
                **evidence,
                "mean_delta_from_parent": mean_delta,
                "best_delta_from_parent": best_delta,
            },
        })
    return patterns[-PATTERN_LIMIT:]


def build_structured_research_memory(
    history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build machine-readable hypothesis evidence without test metrics or prose parsing."""
    rows = [item for item in (history or []) if isinstance(item, dict)]
    hypotheses: list[dict[str, Any]] = []
    for item in rows:
        if item.get("iteration") == 0 and item.get("changes") == {}:
            continue
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        diagnostics = item.get("diagnostics") if isinstance(item.get("diagnostics"), dict) else {}
        selection = item.get("candidate_selection")
        if not isinstance(selection, dict):
            selection = {}
        verdict = critique.get("verdict") or (
            "failed" if item.get("status") != "success" else "unclassified"
        )
        if item.get("decision") == "CONTROL":
            status = "control"
        elif diagnostics.get("placebo_verdict") == "REINTERPRET":
            status = "reinterpreted"
        else:
            status = {
                "promote": "promising",
                "noise": "uncertain",
                "reject": "rejected",
                "failed": "failed",
            }.get(verdict, "unclassified")
        evidence = {
            "validation_primary": _validation_primary(item.get("metrics")),
            "delta_from_parent": (
                float(item["delta_from_parent"])
                if _finite(item.get("delta_from_parent")) else None
            ),
            "delta_from_best": (
                float(item["delta_from_best"])
                if _finite(item.get("delta_from_best")) else None
            ),
            "placebo_verdict": diagnostics.get("placebo_verdict"),
            "strongest_slice_gain": diagnostics.get("strongest_slice_gain"),
            "within_user_score_correlation": diagnostics.get(
                "within_user_score_correlation"
            ),
            "pair_error_recovery_rate": diagnostics.get("pair_error_recovery_rate"),
        }
        confidence = critique.get("confidence", "low")
        if status == "reinterpreted":
            confidence = "high"
        hypotheses.append({
            "iteration": item.get("iteration"),
            "hypothesis": _short_text(item.get("hypothesis")),
            "family": selection.get("selected_family"),
            "status": status,
            "evidence": evidence,
            "confidence": confidence,
            "reason": _short_text(critique.get("interpretation")),
            "next_test": _short_text(critique.get("next_test")),
        })
    return {
        "version": 2,
        "source": "validation_only",
        "test_metrics_included": False,
        "hypotheses": hypotheses,
        "research_patterns": distill_research_patterns(rows),
        "summary": build_memory_summary(rows),
    }
