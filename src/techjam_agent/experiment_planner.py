from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .config import (
    ALLOWED_VALUES,
    FEATURE_KEYS,
    FEATURE_SCHEMA_VERSION,
    apply_changes,
    experiment_key,
)
from .memory import collect_tried_keys, distill_research_patterns
from .skills import default_skill_registry


FEATURE_FAMILIES = {
    "user_long_view_rate": "global_target_statistics",
    "item_long_view_rate": "global_target_statistics",
    "continuous_history_stats": "global_target_statistics",
    "user_tab_long_view_rate": "target_rate_crosses",
    "user_tab_cross": "explicit_crosses",
    "user_author_cross": "explicit_crosses",
    "user_recent_3d_activity": "temporal_counts",
    "item_recent_3d_exposure": "temporal_counts",
    "prior_video_positive": "candidate_history",
    "author_positive_recency": "candidate_history",
    "prior_video_count": "candidate_history",
    "previous_author_same": "candidate_history",
    "prior_video_exposure": "candidate_history",
    "author_recency": "candidate_history",
    "global_context": "global_context",
    "video_tag": "content_metadata",
    "video_upload_type": "content_metadata",
    "user_active_degree": "user_metadata",
    "user_register_days_range": "user_metadata",
    "duration_semantic_bucket": "duration_nonlinearity",
    "video_music_type": "content_metadata",
    "video_tag_components": "content_metadata",
}

# Priors come from validation-only project evidence. They seed the search; measured
# results from the current run override them through _family_observations().
FAMILY_PRIORS = {
    "ranking_objective": (0.90, 0.95, 0.80, 0.25, 0.20),
    "heterogeneous_ensemble": (0.82, 0.90, 0.70, 0.70, 0.30),
    "multitask": (0.62, 0.75, 0.70, 0.65, 0.45),
    "pairwise_multitask": (0.64, 0.60, 0.80, 0.75, 0.35),
    "censored_watchtime": (0.58, 0.55, 0.90, 0.70, 0.30),
    "pairwise_censored_watchtime": (0.60, 0.50, 0.95, 0.80, 0.25),
    "cross_network": (0.52, 0.65, 0.65, 0.60, 0.85),
    "sequence_model": (0.05, 0.90, 0.95, 1.00, 0.20),
    "tree_model": (0.20, 0.90, 0.60, 0.45, 0.50),
    "global_context": (0.45, 0.45, 0.45, 0.20, 0.80),
    # Newly wired static metadata has no rolling confirmation yet, so it is
    # explored after established model families rather than masking them.
    "content_metadata": (0.35, 0.35, 0.75, 0.30, 0.60),
    "user_metadata": (0.30, 0.35, 0.70, 0.30, 0.65),
    # The fixed duration buckets were already tested and fell below BPR; keep
    # the capability legal, but give it a rejected-direction prior.
    "duration_nonlinearity": (0.05, 0.95, 0.30, 0.35, 0.90),
    "temporal_counts": (0.15, 0.85, 0.55, 0.35, 0.65),
    "candidate_history": (0.10, 0.95, 0.65, 0.35, 0.65),
    "global_target_statistics": (0.05, 0.95, 0.35, 0.35, 0.90),
    "target_rate_crosses": (0.08, 0.85, 0.45, 0.40, 0.85),
    "explicit_crosses": (0.05, 0.90, 0.40, 0.30, 0.90),
    "optimization": (0.25, 0.35, 0.20, 0.20, 0.90),
}
MEMORY_MODES = ("no_memory", "raw_history", "distilled_patterns")
PRIOR_POLICIES = {
    "exploit_with_confirmation",
    "ensemble_only",
    "gather_evidence",
    "retest_with_control",
    "stop_direction",
}
SKILL_REGISTRY = default_skill_registry()


@dataclass(frozen=True)
class CandidateExperiment:
    hypothesis: str
    reason: str
    changes: dict[str, Any]
    family: str
    expected_gain: float
    evidence_strength: float
    novelty: float
    compute_cost: float
    redundancy: float
    skill_id: str
    required_confirmation: tuple[str, ...]
    risk: str


@dataclass(frozen=True)
class RankedCandidate:
    candidate: CandidateExperiment
    score: float
    observed_mean_delta: float | None
    family_trials: int
    direction_stopped: bool
    retrieved_pattern: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self.candidate)
        result.update({
            "score": float(self.score),
            "observed_mean_delta": self.observed_mean_delta,
            "family_trials": self.family_trials,
            "direction_stopped": self.direction_stopped,
            "retrieved_pattern": self.retrieved_pattern,
        })
        return result


def _candidate(
    hypothesis: str,
    reason: str,
    changes: dict[str, Any],
    family: str,
    *,
    prior: tuple[float, float, float, float, float] | None = None,
) -> CandidateExperiment:
    expected, evidence, novelty, cost, redundancy = prior or FAMILY_PRIORS[family]
    skill_id = SKILL_REGISTRY.primary_for_candidate(family, changes)
    SKILL_REGISTRY.require(skill_id)
    confirmations = SKILL_REGISTRY.evidence_for_candidate(family, changes)
    if cost >= 0.8:
        risk = "high_compute_cost"
    elif redundancy >= 0.8:
        risk = "high_redundancy"
    elif evidence < 0.5:
        risk = "weak_prior_evidence"
    else:
        risk = "moderate"
    return CandidateExperiment(
        hypothesis, reason, changes, family,
        expected, evidence, novelty, cost, redundancy,
        skill_id, confirmations, risk,
    )


def _model_candidates(config: dict[str, Any]) -> list[CandidateExperiment]:
    model = config["model"]
    rows: list[CandidateExperiment] = []
    if model == "fm" and config["training_objective"] == "bce":
        rows.append(_candidate(
            "Align FM training with within-user ranking using BPR.",
            "The benchmark evaluates GAUC and nDCG within users, so pairwise supervision has strong prior evidence.",
            {"training_objective": "bpr", "learning_rate": 0.0003},
            "ranking_objective",
        ))
    if model == "fm" and config["training_objective"] == "bpr":
        rows.append(_candidate(
            "Blend FM+BPR with DeepFM+BCE.",
            "The two objectives learn complementary ranking errors and the blend improved all rolling folds.",
            {"model": "ensemble", "training_objective": "hybrid", "ensemble_deepfm_weight": 0.4},
            "heterogeneous_ensemble",
        ))
    if model == "ensemble":
        # The first calibration candidate is deliberately the direction that
        # preserves the FM score scale and rank-calibrates only the DeepFM
        # branch.  The reverse transform remains legal for explicit ablations,
        # but is not promoted before evidence exists for it.
        for normalization in ("fm_zscore_deepfm_rank",):
            if config["hyperparameters"]["ensemble_normalization"] == normalization:
                if config["hyperparameters"]["ensemble_deepfm_weight"] != 0.65:
                    rows.append(_candidate(
                        "Tune the DeepFM weight after rank calibration.",
                        "The calibrated DeepFM branch has a different scale; a small, "
                        "predeclared weight check tests whether the improvement is "
                        "calibration-plus-weight rather than a lucky blend.",
                        {"ensemble_deepfm_weight": 0.65},
                        "heterogeneous_ensemble",
                    ))
                continue
            rows.append(_candidate(
                f"Calibrate the ensemble with {normalization}.",
                "GAUC and nDCG depend on within-user ordering; rank calibration tests "
                "whether score scale, rather than model content, limits the blend.",
                {"ensemble_normalization": normalization},
                "heterogeneous_ensemble",
            ))
    if model != "multitask_deepfm":
        rows.append(_candidate(
            "Use like as auxiliary supervision in a multi-task DeepFM.",
            "Like-only supervision was the most consistent auxiliary signal and adds information unavailable at inference time.",
            {"model": "multitask_deepfm", "training_objective": "bce", "learning_rate": 0.001},
            "multitask",
        ))
    if not (model == "multitask_deepfm" and config["training_objective"] == "bpr"):
        rows.append(_candidate(
            "Combine within-user BPR ranking with like-only auxiliary supervision.",
            "FM benefits from BPR and like-only DeepFM has stable rolling evidence; this controlled action changes the main objective without changing the auxiliary target.",
            {"model": "multitask_deepfm", "training_objective": "bpr", "learning_rate": 0.001},
            "pairwise_multitask",
        ))
    if not (
        model == "multitask_deepfm"
        and config["training_objective"] == "bce"
        and config["hyperparameters"]["auxiliary_signals"] == "censored_watch"
    ):
        rows.append(_candidate(
            "Use a one-sided censored watch-time auxiliary objective.",
            "Incomplete plays provide exact log-watch targets, while completed plays "
            "provide only a duration lower bound; this is materially different from capped MSE.",
            {"model": "multitask_deepfm", "training_objective": "bce",
             "auxiliary_signals": "censored_watch", "learning_rate": 0.001},
            "censored_watchtime",
        ))
    if not (
        model == "multitask_deepfm"
        and config["training_objective"] == "bpr"
        and config["hyperparameters"]["auxiliary_signals"] == "censored_watch"
    ):
        rows.append(_candidate(
            "Combine within-user BPR with one-sided censored watch-time supervision.",
            "This aligns the main objective with ranking while retaining uncapped lower-bound "
            "information from completed plays.",
            {"model": "multitask_deepfm", "training_objective": "bpr",
             "auxiliary_signals": "censored_watch", "learning_rate": 0.001},
            "pairwise_censored_watchtime",
        ))
    if model != "dcnv2":
        rows.append(_candidate(
            "Test a low-rank DCNv2 interaction model.",
            "Explicit cross layers test a different interaction mechanism while retaining leakage-safe base fields.",
            {"model": "dcnv2", "training_objective": "bce", "learning_rate": 0.001},
            "cross_network",
        ))
    if model != "sequence_deepfm":
        rows.append(_candidate(
            "Test strict-causal last-16 candidate-conditioned sequence attention.",
            "Order-aware history is a distinct information source, but prior runtime and accuracy evidence make it a costly exploratory candidate.",
            {"model": "sequence_deepfm", "training_objective": "bce", "learning_rate": 0.001},
            "sequence_model",
        ))
    if model != "lightgbm":
        rows.append(_candidate(
            "Test LightGBM on the original fields.",
            "This provides a non-neural tabular control, though prior evidence is negative.",
            {"model": "lightgbm", "training_objective": "bce"},
            "tree_model",
        ))
    return rows


def _feature_candidates(config: dict[str, Any]) -> list[CandidateExperiment]:
    rows = []
    for feature in FEATURE_KEYS:
        if config["features"].get(feature):
            continue
        family = FEATURE_FAMILIES[feature]
        rows.append(_candidate(
            f"Test leakage-safe {feature} as a single feature intervention.",
            f"This isolates the {family} mechanism; categorical gains require matched placebo controls before attribution.",
            {feature: True},
            family,
        ))
    return rows


def _optimization_candidates(config: dict[str, Any]) -> list[CandidateExperiment]:
    hp = config["hyperparameters"]
    rows = []
    for key in ("learning_rate", "embedding_dim", "l2", "batch_size"):
        for value in ALLOWED_VALUES[key]:
            if value == hp[key]:
                continue
            rows.append(_candidate(
                f"Test {key}={value} as a controlled optimization change.",
                "This is a low-novelty fallback after higher-information mechanisms are exhausted.",
                {key: value},
                "optimization",
            ))
    return rows


def generate_candidates(config: dict[str, Any]) -> list[CandidateExperiment]:
    """Generate legal actions supported by the current executable search space."""
    candidates = _model_candidates(config) + _feature_candidates(config) + _optimization_candidates(config)
    legal: list[CandidateExperiment] = []
    for candidate in candidates:
        try:
            apply_changes(config, candidate.changes)
        except (KeyError, TypeError, ValueError):
            continue
        legal.append(candidate)
    return legal


def _family_for_item(item: dict[str, Any]) -> str:
    selection = item.get("candidate_selection")
    if isinstance(selection, dict) and isinstance(selection.get("selected_family"), str):
        return selection["selected_family"]
    changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
    for key in changes:
        if key in FEATURE_FAMILIES:
            return FEATURE_FAMILIES[key]
    if changes.get("training_objective") == "bpr":
        return "ranking_objective"
    model = changes.get("model")
    if model == "multitask_deepfm" and changes.get("training_objective") == "bpr":
        return "pairwise_multitask"
    return {
        "ensemble": "heterogeneous_ensemble",
        "multitask_deepfm": "multitask",
        "dcnv2": "cross_network",
        "sequence_deepfm": "sequence_model",
        "lightgbm": "tree_model",
    }.get(model, "optimization")


def _family_observations(history: list[dict[str, Any]], family: str) -> tuple[list[float], int, int]:
    deltas: list[float] = []
    weak = 0
    strong_slice_or_diversity = 0
    for item in history:
        if _family_for_item(item) != family:
            continue
        delta = item.get("delta_from_parent")
        try:
            value = float(delta)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            deltas.append(value)
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        if critique.get("verdict") in {"noise", "reject", "failed"}:
            weak += 1
        diagnostics = item.get("diagnostics") if isinstance(item.get("diagnostics"), dict) else {}
        if diagnostics.get("strong_slice_gain") or diagnostics.get("diversity_advantage"):
            strong_slice_or_diversity += 1
    return deltas, weak, strong_slice_or_diversity


def _has_unseen_ensemble_variant(
    config: dict[str, Any],
    candidate: CandidateExperiment,
    history: list[dict[str, Any]],
) -> bool:
    """Keep unexplored ensemble calibration values alive after noisy siblings.

    Ensemble calibration is a two-stage intervention: changing the score
    transform and then tuning its blend weight.  Treating both as one family
    is useful for memory, but a noisy result for one variant must not suppress
    a value that has never been measured.  This guard is deliberately narrow;
    it does not reopen unrelated rejected feature/model families.
    """
    if candidate.family != "heterogeneous_ensemble":
        return False
    if config.get("model") != "ensemble":
        return False
    variant_keys = {"ensemble_normalization", "ensemble_deepfm_weight"}
    hp = config.get("hyperparameters", {})
    for key in variant_keys.intersection(candidate.changes):
        target = candidate.changes[key]
        seen = set()
        for item in history:
            item_config = item.get("config") if isinstance(item, dict) else None
            if not isinstance(item_config, dict) or item_config.get("model") != "ensemble":
                continue
            item_hp = item_config.get("hyperparameters", {})
            if key in item_hp:
                seen.add(item_hp[key])
        if hp.get(key) != target and target not in seen:
            return True
    return False


def _prior_family_patterns(
    prior_evidence: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Read only explicit machine-readable policies; never infer policy from prose."""
    if not isinstance(prior_evidence, dict):
        return {}
    rows = prior_evidence.get("family_policies")
    if not isinstance(rows, list):
        return {}
    patterns: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        family, policy = row.get("family"), row.get("policy")
        if family not in FAMILY_PRIORS or policy not in PRIOR_POLICIES:
            continue
        pattern = {
            "family": str(family),
            "policy": str(policy),
            "confidence": row.get("confidence"),
            "evidence": row.get("evidence"),
            "policy_id": row.get("policy_id"),
            "scientific_verdict": row.get("scientific_verdict"),
            "competition_status": row.get("competition_status"),
            "applies_to": row.get("applies_to"),
            "expires_if": row.get("expires_if"),
            "created_from": row.get("created_from"),
            "source": "persistent_validation_memory",
        }
        patterns.setdefault(str(family), []).append(pattern)
    return patterns


def _policy_applies(pattern: dict[str, Any], candidate_config: dict[str, Any]) -> bool:
    """Return whether a persisted policy is valid for this concrete action."""
    scope = pattern.get("applies_to")
    if scope is None:
        return True  # Backward-compatible manual policy without an explicit scope.
    if not isinstance(scope, dict):
        return False
    task = scope.get("task")
    if task not in (None, "long_view"):
        return False
    schema = scope.get("feature_schema")
    if schema not in (None, FEATURE_SCHEMA_VERSION):
        return False
    models = scope.get("models", [])
    if not isinstance(models, list):
        return False
    if models and candidate_config.get("model") not in models:
        return False
    objectives = scope.get("training_objectives", [])
    if not isinstance(objectives, list):
        return False
    if objectives and candidate_config.get("training_objective") not in objectives:
        return False
    for section in ("features", "hyperparameters"):
        expected = scope.get(section, {})
        actual = candidate_config.get(section, {})
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return False
        if any(actual.get(key) != value for key, value in expected.items()):
            return False
    return True


def _matching_prior_pattern(
    patterns: dict[str, list[dict[str, Any]]],
    family: str,
    candidate_config: dict[str, Any],
) -> dict[str, Any] | None:
    applicable = [
        pattern for pattern in patterns.get(family, [])
        if _policy_applies(pattern, candidate_config)
    ]
    if not applicable:
        return None

    def confidence(pattern: dict[str, Any]) -> float:
        raw = pattern.get("confidence")
        if isinstance(raw, (int, float)):
            return float(raw)
        return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(str(raw), 0.0)

    return max(applicable, key=confidence)


def rank_candidates(
    config: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
    memory_mode: str = "distilled_patterns",
    prior_evidence: dict[str, Any] | None = None,
) -> list[RankedCandidate]:
    """Rank legal, untried experiments by value, evidence, novelty, cost and redundancy."""
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
    weights = {
        "expected_gain": 0.35,
        "evidence_strength": 0.25,
        "novelty": 0.20,
        "compute_cost": 0.10,
        "redundancy": 0.10,
        "observed_gain": 80.0,
        **(weights or {}),
    }
    tried = set(collect_tried_keys(history))
    current_patterns = {
        pattern["family"]: pattern for pattern in distill_research_patterns(history)
    } if memory_mode == "distilled_patterns" else {}
    prior_patterns = (
        _prior_family_patterns(prior_evidence)
        if memory_mode == "distilled_patterns" else {}
    )
    ranked: list[RankedCandidate] = []
    for candidate in generate_candidates(config):
        changed = apply_changes(config, candidate.changes)
        if experiment_key(changed) in tried:
            continue
        if memory_mode == "no_memory":
            deltas, weak, advantage = [], 0, 0
        else:
            deltas, weak, advantage = _family_observations(history, candidate.family)
        observed = sum(deltas) / len(deltas) if deltas else None
        unseen_ensemble_variant = _has_unseen_ensemble_variant(
            config, candidate, history
        )
        stopped = (
            weak >= 2
            and advantage == 0
            and not unseen_ensemble_variant
        )
        score = (
            weights["expected_gain"] * candidate.expected_gain
            + weights["evidence_strength"] * candidate.evidence_strength
            + weights["novelty"] * candidate.novelty / (1 + len(deltas))
            - weights["compute_cost"] * candidate.compute_cost
            - weights["redundancy"] * candidate.redundancy
        )
        if observed is not None:
            score += weights["observed_gain"] * max(-0.005, min(0.005, observed))
        if unseen_ensemble_variant:
            # A new calibration value is an information-gathering follow-up,
            # not a repetition of the noisy family mean. Give it a bounded
            # confirmation bonus so an old scoped policy cannot crowd it out
            # before the new value is measured.
            score += 0.08
        # A scoped artifact policy describes this concrete configuration and is
        # stronger than a family-level pattern distilled from a different variant
        # in the current run. Unscoped/current evidence remains the fallback.
        pattern = _matching_prior_pattern(
            prior_patterns, candidate.family, changed
        ) or current_patterns.get(candidate.family) or {}
        pattern_policy = pattern.get("policy")
        if pattern_policy == "exploit_with_confirmation":
            score += 0.08
        elif pattern_policy == "ensemble_only":
            score += 0.03
        elif pattern_policy == "gather_evidence":
            # Scientific uncertainty must not silently remove a promising leaderboard
            # candidate. Submission-eligible configurations receive one controlled turn.
            competition = pattern.get("competition_status")
            if competition == "ELIGIBLE":
                score += 0.10
            elif competition == "RESEARCH_ONLY":
                score -= 0.08
            else:
                score += 0.01
        if pattern_policy == "stop_direction" and not unseen_ensemble_variant:
            stopped = True
        if stopped:
            score -= 1.0
        ranked.append(RankedCandidate(
            candidate, score, observed, len(deltas), stopped,
            pattern if pattern else None,
        ))
    return sorted(ranked, key=lambda row: (-row.score, row.candidate.family,
                                            experiment_key(apply_changes(config, row.candidate.changes))))


class AutonomousExperimentPlanner:
    """Deterministic evidence-driven planner used offline and as the LLM fallback."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        memory_mode: str = "distilled_patterns",
        prior_evidence: dict[str, Any] | None = None,
    ) -> None:
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        self.weights = weights or {}
        self.memory_mode = memory_mode
        self.prior_evidence = prior_evidence or {}
        self.last_selection: dict[str, Any] | None = None

    def select(self, config: dict[str, Any], history: list[dict[str, Any]]) -> CandidateExperiment:
        ranked = [row for row in rank_candidates(
            config, history, weights=self.weights, memory_mode=self.memory_mode,
            prior_evidence=self.prior_evidence,
        )
                  if not row.direction_stopped]
        if not ranked:
            raise StopIteration("all legal experiment directions are exhausted or stopped")
        winner = ranked[0]
        counterfactual_choices: dict[str, dict[str, Any] | None] = {}
        selected_key = experiment_key(apply_changes(config, winner.candidate.changes))
        for mode in MEMORY_MODES:
            mode_ranked = [row for row in rank_candidates(
                config, history, weights=self.weights, memory_mode=mode,
                prior_evidence=self.prior_evidence,
            ) if not row.direction_stopped]
            if not mode_ranked:
                counterfactual_choices[mode] = None
                continue
            alternative = mode_ranked[0]
            alternative_key = experiment_key(
                apply_changes(config, alternative.candidate.changes)
            )
            counterfactual_choices[mode] = {
                "family": alternative.candidate.family,
                "changes": alternative.candidate.changes,
                "score": float(alternative.score),
                "differs_from_selected": alternative_key != selected_key,
            }
        current_patterns = {
            pattern["family"]: pattern for pattern in distill_research_patterns(history)
        } if self.memory_mode == "distilled_patterns" else {}
        prior_patterns = (
            _prior_family_patterns(self.prior_evidence)
            if self.memory_mode == "distilled_patterns" else {}
        )
        selected_config = apply_changes(config, winner.candidate.changes)
        retrieved_pattern = winner.retrieved_pattern or _matching_prior_pattern(
            prior_patterns, winner.candidate.family, selected_config
        ) or current_patterns.get(winner.candidate.family)
        self.last_selection = {
            "memory_mode": self.memory_mode,
            "selected_family": winner.candidate.family,
            "selected_score": float(winner.score),
            "selected_skill": winner.candidate.skill_id,
            "criteria": (
                "expected_gain + evidence_strength + novelty - compute_cost - redundancy; "
                "scoped artifact policies override family-level patterns only when the "
                "candidate configuration matches their declared scope"
            ),
            "retrieved_pattern": retrieved_pattern,
            "decision_stage": (
                "slow_confirmation"
                if retrieved_pattern
                and retrieved_pattern.get("policy") == "exploit_with_confirmation"
                else "fast_screen"
            ),
            "counterfactual_choices": counterfactual_choices,
            "memory_changed_choice": any(
                choice is not None and choice["differs_from_selected"]
                for choice in counterfactual_choices.values()
            ),
            "ranked_candidates": [row.as_dict() for row in ranked[:5]],
            "decision_record": {
                "hypothesis": winner.candidate.hypothesis,
                "mechanism_basis": winner.candidate.reason,
                "family": winner.candidate.family,
                "proposed_action": winner.candidate.skill_id,
                "expected_gain": winner.candidate.expected_gain,
                "novelty": winner.candidate.novelty,
                "risk": winner.candidate.risk,
                "required_confirmation": list(
                    winner.candidate.required_confirmation
                ),
            },
        }
        return winner.candidate
