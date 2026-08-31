from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any


CONFIRMATION_KINDS = ("rolling", "paired_seeds")
FINAL_SCIENTIFIC_STATUSES = ("VALIDATED", "UNCERTAIN", "REJECTED")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class ConfirmationAction:
    action_id: str
    kind: str
    target_iteration: int
    family: str
    candidate_config: dict[str, Any]
    reference_config: dict[str, Any]
    reason: str
    estimated_training_runs: int
    seeds: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["seeds"] = list(self.seeds)
        return value


@dataclass(frozen=True)
class EscalationDecision:
    scientific_status: str
    competition_status: str
    reason: str
    next_action: ConfirmationAction | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scientific_status": self.scientific_status,
            "competition_status": self.competition_status,
            "reason": self.reason,
            "next_action": (
                None if self.next_action is None else self.next_action.as_dict()
            ),
        }


class EvidenceEscalator:
    """Turn promising discovery results into deterministic confirmation actions.

    The LLM may propose a hypothesis, but thresholds and confirmation ordering are
    code-controlled. Test metrics are not accepted by this interface.
    """

    def __init__(
        self,
        *,
        minimum_delta: float = 0.0002,
        high_novelty: float = 0.8,
        rolling_wins_required: int = 2,
        paired_seeds: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        self.minimum_delta = float(minimum_delta)
        self.high_novelty = float(high_novelty)
        self.rolling_wins_required = int(rolling_wins_required)
        self.paired_seeds = tuple(int(seed) for seed in paired_seeds)

    @classmethod
    def from_project(cls, project: dict[str, Any]) -> "EvidenceEscalator":
        policy = project.get("autonomy", {}).get("evidence_escalation", {})
        return cls(
            minimum_delta=policy.get("minimum_delta", 0.0002),
            high_novelty=policy.get("high_novelty", 0.8),
            rolling_wins_required=policy.get("rolling_wins_required", 2),
            paired_seeds=tuple(policy.get("paired_seeds", (0, 1, 2, 3))),
        )

    @staticmethod
    def _selected_novelty(item: dict[str, Any]) -> float:
        selection = item.get("candidate_selection")
        if not isinstance(selection, dict):
            return 0.0
        changes = item.get("changes")
        for row in selection.get("ranked_candidates", []):
            if isinstance(row, dict) and row.get("changes") == changes:
                return float(row.get("novelty", 0.0) or 0.0)
        return 0.0

    @staticmethod
    def _existing_final_status(item: dict[str, Any]) -> tuple[str, str] | None:
        selection = item.get("candidate_selection")
        pattern = (
            selection.get("retrieved_pattern")
            if isinstance(selection, dict) else None
        )
        status = (
            pattern.get("scientific_verdict")
            if isinstance(pattern, dict) else None
        )
        if status not in FINAL_SCIENTIFIC_STATUSES:
            return None
        competition = pattern.get("competition_status")
        if competition not in {"ELIGIBLE", "RESEARCH_ONLY", "NOT_ELIGIBLE"}:
            competition = "NOT_ELIGIBLE" if status == "REJECTED" else "ELIGIBLE"
        return str(status), str(competition)

    def plan_discovery(
        self,
        item: dict[str, Any],
        reference_config: dict[str, Any] | None,
    ) -> EscalationDecision:
        if item.get("status") != "success" or reference_config is None:
            return EscalationDecision(
                "NOT_APPLICABLE", "RESEARCH_ONLY", "No successful parent comparison."
            )
        if item.get("decision") in {"CONTROL", "REINTERPRET"}:
            return EscalationDecision(
                "REJECTED", "NOT_ELIGIBLE", "Controls cannot trigger confirmation."
            )
        existing = self._existing_final_status(item)
        if existing is not None:
            existing_status, competition = existing
            return EscalationDecision(
                existing_status,
                competition,
                "Matching artifact evidence already contains the final confirmation status.",
            )
        diagnostics = item.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("placebo_status") == "scheduled":
            return EscalationDecision(
                "AWAITING_CONTROL", "RESEARCH_ONLY", "Matched placebo controls must finish first."
            )
        delta = _finite(item.get("delta_from_parent"))
        if delta is None or delta <= 0:
            return EscalationDecision(
                "REJECTED", "NOT_ELIGIBLE", "The discovery did not improve its declared parent."
            )
        diverse = bool(
            isinstance(diagnostics, dict) and diagnostics.get("diversity_advantage")
        )
        novelty = self._selected_novelty(item)
        if delta < self.minimum_delta and novelty < self.high_novelty and not diverse:
            return EscalationDecision(
                "INSUFFICIENT",
                "RESEARCH_ONLY",
                "The gain is below the confirmation threshold without high novelty or diversity.",
            )
        family = "unknown"
        selection = item.get("candidate_selection")
        if isinstance(selection, dict) and isinstance(selection.get("selected_family"), str):
            family = selection["selected_family"]
        action = ConfirmationAction(
            action_id=f"confirm_{int(item['iteration']):03d}_rolling",
            kind="rolling",
            target_iteration=int(item["iteration"]),
            family=family,
            candidate_config=copy.deepcopy(item["config"]),
            reference_config=copy.deepcopy(reference_config),
            reason="A positive discovery must survive expanding-window temporal validation.",
            estimated_training_runs=6,
        )
        return EscalationDecision(
            "PROMISING_NOT_CONFIRMED", "ELIGIBLE", action.reason, action
        )

    def evaluate(
        self,
        action: ConfirmationAction,
        result: dict[str, Any],
    ) -> EscalationDecision:
        if action.kind == "rolling":
            mean = _finite(result.get("mean_delta"))
            wins = int(result.get("wins", 0))
            folds = int(result.get("folds", 0))
            if mean is None or folds < 3 or mean <= 0 or wins < self.rolling_wins_required:
                return EscalationDecision(
                    "REJECTED",
                    "NOT_ELIGIBLE",
                    f"Rolling confirmation failed: wins={wins}/{folds}, mean_delta={mean}.",
                )
            next_action = ConfirmationAction(
                action_id=f"confirm_{action.target_iteration:03d}_paired_seeds",
                kind="paired_seeds",
                target_iteration=action.target_iteration,
                family=action.family,
                candidate_config=copy.deepcopy(action.candidate_config),
                reference_config=copy.deepcopy(action.reference_config),
                reason="Rolling passed; paired seeds now test optimization stability.",
                estimated_training_runs=2 * len(self.paired_seeds),
                seeds=self.paired_seeds,
            )
            return EscalationDecision(
                "PROMISING_NOT_CONFIRMED", "ELIGIBLE", next_action.reason, next_action
            )
        if action.kind == "paired_seeds":
            mean = _finite(result.get("paired_mean_delta"))
            wins = int(result.get("wins", 0))
            seeds = int(result.get("seeds", 0))
            interval = result.get("approx_95_interval")
            lower = (
                _finite(interval[0])
                if isinstance(interval, list) and len(interval) == 2 else None
            )
            if mean is None or mean <= 0 or wins < math.ceil(max(1, seeds) / 2):
                return EscalationDecision(
                    "REJECTED",
                    "NOT_ELIGIBLE",
                    f"Paired-seed confirmation failed: wins={wins}/{seeds}, mean_delta={mean}.",
                )
            if lower is not None and lower > 0:
                return EscalationDecision(
                    "VALIDATED", "ELIGIBLE", "Paired-seed interval is entirely positive."
                )
            return EscalationDecision(
                "UNCERTAIN",
                "ELIGIBLE",
                "Paired seeds are positive on average, but the uncertainty interval crosses zero.",
            )
        raise ValueError(f"unknown confirmation action kind: {action.kind}")


def action_from_dict(value: dict[str, Any]) -> ConfirmationAction:
    kind = value.get("kind")
    if kind not in CONFIRMATION_KINDS:
        raise ValueError(f"unknown confirmation action kind: {kind}")
    return ConfirmationAction(
        action_id=str(value["action_id"]),
        kind=str(kind),
        target_iteration=int(value["target_iteration"]),
        family=str(value["family"]),
        candidate_config=copy.deepcopy(value["candidate_config"]),
        reference_config=copy.deepcopy(value["reference_config"]),
        reason=str(value["reason"]),
        estimated_training_runs=int(value["estimated_training_runs"]),
        seeds=tuple(int(seed) for seed in value.get("seeds", ())),
    )
