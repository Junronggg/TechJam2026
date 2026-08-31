from __future__ import annotations

import math
import statistics
from typing import Any


def summarize_objective_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired validation-only BCE/BPR results by seed."""
    paired: list[dict[str, float | int]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        bce = row.get("bce") if isinstance(row.get("bce"), dict) else None
        bpr = row.get("bpr") if isinstance(row.get("bpr"), dict) else None
        try:
            bce_primary = float(bce["primary"])
            bpr_primary = float(bpr["primary"])
        except (KeyError, TypeError, ValueError):
            failures.append({"seed": seed, "error": row.get("error", "missing metrics")})
            continue
        if not math.isfinite(bce_primary) or not math.isfinite(bpr_primary):
            failures.append({"seed": seed, "error": "non-finite validation Primary"})
            continue
        paired.append({
            "seed": seed,
            "bce_primary": bce_primary,
            "bpr_primary": bpr_primary,
            "paired_delta": bpr_primary - bce_primary,
        })

    bce_values = [float(row["bce_primary"]) for row in paired]
    bpr_values = [float(row["bpr_primary"]) for row in paired]
    deltas = [float(row["paired_delta"]) for row in paired]

    def aggregate(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "std": None}
        return {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "split": "validation",
        "paired_results": paired,
        "bce": aggregate(bce_values),
        "bpr": aggregate(bpr_values),
        "paired_delta": aggregate(deltas),
        "seeds_improved": sum(delta > 0 for delta in deltas),
        "seeds_total": len(paired),
        "failures": failures,
    }


def critique_replications(summary: dict[str, Any], epsilon: float = 0.002) -> dict[str, Any]:
    total = int(summary.get("seeds_total", 0))
    improved = int(summary.get("seeds_improved", 0))
    paired = summary.get("paired_delta") if isinstance(summary.get("paired_delta"), dict) else {}
    mean_delta = paired.get("mean")
    std_delta = paired.get("std")
    if total == 0 or mean_delta is None:
        return {
            "observation": "No complete paired validation replications were available.",
            "interpretation": "No objective comparison can be made.",
            "confidence": "low",
            "verdict": "failed",
        }
    observation = (
        f"Paired validation mean delta={float(mean_delta):+.6f}, "
        f"std={float(std_delta):.6f}; BPR improved {improved}/{total} seeds."
    )
    if total == 1:
        interpretation, confidence, verdict = (
            "BPR is promising but this remains single-seed evidence.", "low", "single_seed"
        )
    elif improved == total:
        if float(mean_delta) > epsilon:
            interpretation, confidence, verdict = (
                "BPR improved every replicated seed by a meaningful average margin.",
                "high", "consistent_improvement",
            )
        else:
            interpretation, confidence, verdict = (
                "BPR improved every replicated seed, but the mean gain is within epsilon.",
                "high", "consistent_small_improvement",
            )
    elif improved > total / 2 and float(mean_delta) > 0:
        interpretation, confidence, verdict = (
            "BPR improved most seeds, but the effect is small or variable.",
            "medium", "promising_inconsistent",
        )
    elif float(mean_delta) < -epsilon:
        interpretation, confidence, verdict = (
            "BPR regressed by a meaningful average margin.", "high", "regression"
        )
    else:
        interpretation, confidence, verdict = (
            "The paired objective result is inconsistent or within the noise threshold.",
            "medium", "noise",
        )
    return {
        "observation": observation,
        "interpretation": interpretation,
        "confidence": confidence,
        "verdict": verdict,
    }
