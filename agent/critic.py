"""Critic contract and a metric-grounded fallback critic."""

from __future__ import annotations

from typing import Protocol

from experiment.schemas import (
    Confidence,
    CriticResult,
    Decision,
    ExperimentResult,
    ExperimentSpec,
    ExperimentStatus,
    MetricBundle,
)


class Critic(Protocol):
    def review(
        self,
        parent: MetricBundle,
        candidate: ExperimentResult,
        spec: ExperimentSpec,
    ) -> CriticResult:
        """Ground interpretation in the supplied metrics and exact experiment spec."""


class GroundedCritic:
    def __init__(self, meaningful_delta: float = 0.002) -> None:
        self.meaningful_delta = meaningful_delta

    def review(
        self,
        parent: MetricBundle,
        candidate: ExperimentResult,
        spec: ExperimentSpec,
    ) -> CriticResult:
        if candidate.status is not ExperimentStatus.SUCCESS or candidate.metrics is None:
            return CriticResult(
                observation=f"Experiment ended with status={candidate.status.value}.",
                interpretation="No scientific conclusion can be drawn until execution succeeds.",
                confidence=Confidence.HIGH,
                decision=Decision.REPAIR if candidate.recovery_attempts == 0 else Decision.REJECT,
                next_test="Repair once using the captured error, then abandon if it fails again.",
            )

        child = candidate.metrics
        gauc_delta = child.gauc - parent.gauc
        ndcg_delta = child.ndcg_at_5 - parent.ndcg_at_5
        primary_delta = child.primary - parent.primary
        observation = (
            f"Primary {primary_delta:+.4f}; GAUC {gauc_delta:+.4f}; "
            f"nDCG@5 {ndcg_delta:+.4f}."
        )

        if primary_delta > self.meaningful_delta:
            decision, confidence = Decision.KEEP, Confidence.HIGH
            interpretation = "The improvement exceeds the competition convergence scale."
        elif primary_delta > 0:
            decision, confidence = Decision.FOLLOW_UP, Confidence.LOW
            interpretation = "The gain is positive but small enough to be noise-sensitive."
        elif primary_delta <= -self.meaningful_delta:
            decision, confidence = Decision.REJECT, Confidence.HIGH
            interpretation = "The regression exceeds the competition convergence scale."
        else:
            decision, confidence = Decision.REJECT, Confidence.MEDIUM
            interpretation = "The change did not improve Primary beyond likely run noise."

        if gauc_delta * ndcg_delta < 0:
            interpretation += " The two ranking metrics moved in opposite directions."

        return CriticResult(
            observation=observation,
            interpretation=interpretation,
            confidence=confidence,
            decision=decision,
            next_test=f"Use this evidence to refine or contrast {spec.branch!r} on a new branch.",
        )

