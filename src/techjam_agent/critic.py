from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_EPSILON = 0.002
VALIDATION_METRIC_KEYS = ("GAUC", "nDCG@5", "primary")
VERDICTS = ("promote", "noise", "reject", "failed")


@dataclass(frozen=True)
class CriticResult:
    observation: str
    interpretation: str
    confidence: str
    verdict: str
    delta: float | None
    meaningful_improvement: bool
    next_test: str
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["delta"] is not None:
            payload["delta"] = float(payload["delta"])
        return payload


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validation_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key in VALIDATION_METRIC_KEYS:
        if key in metrics and _finite(metrics[key]):
            cleaned[key] = float(metrics[key])
    return cleaned


def _format_primary(value: float) -> str:
    return f"{value:.6f}"


def _change_label(changes: dict[str, Any] | None) -> str:
    if not changes:
        return "this configuration"
    return ", ".join(f"{key}={value}" for key, value in changes.items())


def _recent_validation_notes(history: list[dict[str, Any]], limit: int = 3) -> list[str]:
    notes: list[str] = []
    for item in reversed(history):
        metrics = _validation_metrics(item.get("metrics"))
        if "primary" not in metrics:
            continue
        iteration = item.get("iteration")
        label = f"iteration {iteration}" if iteration is not None else "a previous run"
        notes.append(f"{label} validation Primary={_format_primary(metrics['primary'])}")
        if len(notes) >= limit:
            break
    notes.reverse()
    return notes


def _noisy_same_direction_count(history: list[dict[str, Any]], changes: dict[str, Any] | None) -> int:
    keys = set(changes or ())
    count = 0
    for item in reversed(history):
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        if critique.get("verdict") != "noise":
            break
        previous_changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
        if keys and previous_changes and keys.isdisjoint(previous_changes):
            break
        count += 1
    return count


def _next_test(verdict: str, changes: dict[str, Any] | None, noisy_streak: int,
               epsilon: float) -> str:
    label = _change_label(changes)
    if verdict == "failed":
        return "Do not interpret ranking quality from this run; retry a validated configuration or inspect the failure type."
    if verdict == "promote":
        return (
            f"Repeat {label} on a new seed before treating the gain as stable. "
            "One seed is not a statistical significance test."
        )
    if verdict == "reject":
        return f"Do not spend further iterations on {label}; try a different mechanism (loss, model, or feature family)."
    if noisy_streak >= 2:
        return (
            f"Recent validation moves of {label} stayed inside epsilon {epsilon}. "
            "Switch to a distinct hypothesis rather than another small tweak."
        )
    return (
        f"The change in {label} is within epsilon {epsilon}. "
        "Try a different mechanism instead of a smaller nudge of the same knobs."
    )


def review(
    metrics: dict[str, Any] | None,
    parent_score: float | None,
    epsilon: float,
    status: str,
    error: dict[str, Any] | None = None,
    *,
    history: list[dict[str, Any]] | None = None,
    changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate measured validation facts from interpretation. Returns a JSON-safe dict."""
    return _review(
        metrics, parent_score, epsilon, status, error,
        history=history, changes=changes,
    ).as_dict()


def _review(
    metrics: dict[str, Any] | None,
    parent_score: float | None,
    epsilon: float,
    status: str,
    error: dict[str, Any] | None = None,
    *,
    history: list[dict[str, Any]] | None = None,
    changes: dict[str, Any] | None = None,
) -> CriticResult:
    history = history or []
    threshold = float(epsilon) if _finite(epsilon) else DEFAULT_EPSILON
    valid = _validation_metrics(metrics)
    failed_status = status not in {"success", "ok"}
    missing_primary = "primary" not in valid

    if failed_status or metrics is None or missing_primary:
        message = "unknown error"
        if isinstance(error, dict) and error.get("message"):
            message = str(error["message"])
        elif isinstance(error, dict) and error.get("type"):
            message = str(error["type"])
        elif missing_primary and status in {"success", "ok"}:
            message = "missing or non-finite validation Primary"
        observation = f"Experiment failed: {message}"
        return CriticResult(
            observation=observation,
            interpretation="No ranking-quality conclusion can be drawn from this run.",
            confidence="high",
            verdict="failed",
            delta=None,
            meaningful_improvement=False,
            next_test=_next_test("failed", changes, 0, threshold),
            reasons=["failed_or_invalid_metrics"],
        )

    primary = valid["primary"]
    parent = float(parent_score) if parent_score is not None and _finite(parent_score) else None
    delta = None if parent is None else primary - parent
    gauc = valid.get("GAUC")
    ndcg = valid.get("nDCG@5")

    observation = f"Validation Primary={_format_primary(primary)}"
    if gauc is not None:
        observation += f", GAUC={_format_primary(gauc)}"
    if ndcg is not None:
        observation += f", nDCG@5={_format_primary(ndcg)}"
    if parent is None:
        observation += "."
    else:
        observation += f" versus previous best {_format_primary(parent)} (delta={delta:+.6f})."

    previous_notes = _recent_validation_notes(history)
    if previous_notes:
        observation += " Recent validation evidence: " + "; ".join(previous_notes) + "."

    noisy_streak = _noisy_same_direction_count(history, changes)
    reasons: list[str] = ["validation_primary_only"]

    if parent is None or delta is None:
        verdict = "noise"
        meaningful = False
        confidence = "high"
        interpretation = (
            "This run establishes a validation reference. It is not an improvement over a previous best."
        )
        reasons.append("no_parent_to_compare")
    elif delta > threshold:
        verdict = "promote"
        meaningful = True
        confidence = "medium"
        interpretation = (
            f"Validation Primary increased by more than epsilon {threshold}. "
            "This is a single-seed observation, not a statistical significance test."
        )
        reasons.append("delta_above_epsilon")
    elif delta < -threshold:
        verdict = "reject"
        meaningful = False
        confidence = "medium"
        interpretation = (
            f"Validation Primary decreased by more than epsilon {threshold} versus the previous best."
        )
        reasons.append("delta_below_negative_epsilon")
    else:
        verdict = "noise"
        meaningful = False
        confidence = "low"
        interpretation = (
            f"Validation Primary changed by {delta:+.6f}, which is within epsilon {threshold}. "
            "Treat this as noise, not a confirmed improvement or regression."
        )
        reasons.append("delta_within_epsilon")
        if delta > 0:
            reasons.append("tiny_positive_not_meaningful")

    if noisy_streak >= 2 and verdict == "noise":
        reasons.append("repeated_noisy_changes")

    if previous_notes:
        reasons.append("used_recent_validation_history")

    return CriticResult(
        observation=observation,
        interpretation=interpretation,
        confidence=confidence,
        verdict=verdict,
        delta=None if delta is None else float(delta),
        meaningful_improvement=meaningful,
        next_test=_next_test(verdict, changes, noisy_streak, threshold),
        reasons=reasons,
    )
