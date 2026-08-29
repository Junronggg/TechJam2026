from __future__ import annotations

from typing import Any


def review(metrics: dict[str, Any] | None, parent_score: float | None,
           epsilon: float, status: str, error: dict[str, Any] | None = None) -> dict[str, Any]:
    """Separate measured evidence from interpretation and the next recommended check."""
    if status != "success" or metrics is None:
        return {
            "observation": f"Experiment failed: {(error or {}).get('message', 'unknown error')}",
            "interpretation": "No model-quality conclusion can be drawn from a failed run.",
            "confidence": "high",
            "next_test": "Recover safely or choose a different validated experiment.",
        }
    delta = None if parent_score is None else float(metrics["primary"]) - parent_score
    observation = f"Validation Primary={float(metrics['primary']):.6f}"
    if delta is not None:
        observation += f", delta versus previous best={delta:+.6f}."
    else:
        observation += "."
    if delta is None:
        interpretation, confidence = "This establishes the validation baseline.", "high"
    elif delta > epsilon:
        interpretation, confidence = "The change produced an improvement above the significance threshold.", "medium"
    elif delta > 0:
        interpretation, confidence = "The change improved validation, but the gain is below the significance threshold.", "low"
    else:
        interpretation, confidence = "The change did not improve the current validation best.", "medium"
    return {"observation": observation, "interpretation": interpretation,
            "confidence": confidence,
            "next_test": "Repeat promising results across seeds; otherwise test a distinct hypothesis."}
