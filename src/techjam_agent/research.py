from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from .config import apply_changes
from .operator_registry import MODEL_SPECS, OPERATORS, branch_for_changes


VALID_METRICS = ("GAUC", "nDCG@5", "primary")
DEFAULT_EPSILON = 0.002
COST_SECONDS = {"low": 75.0, "medium": 240.0, "high": 600.0}


@dataclass(frozen=True)
class ResearchStrategy:
    strategy_id: str
    title: str
    branches: tuple[str, ...]
    purpose: str
    success_signal: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


STRATEGIES = (
    ResearchStrategy(
        "strategy_ranking_alignment",
        "Align training with within-user ranking",
        ("ranking_objective",),
        "Test objectives or negative sampling when top-rank quality is the bottleneck.",
        "GAUC and nDCG@5 improve together against a matched parent.",
    ),
    ResearchStrategy(
        "strategy_personalization_signal",
        "Add leakage-safe personalization signal",
        ("features",),
        "Test a train-only candidate/context feature or affinity; user-only fields are useful only through candidate interactions.",
        "The feature adds Primary beyond the same model without metric trade-off.",
    ),
    ResearchStrategy(
        "strategy_interaction_capacity",
        "Change interaction inductive bias",
        ("model",),
        "Use a distinct interaction mechanism only when simpler representations appear saturated.",
        "A new model family improves both component metrics or exposes a useful trade-off.",
    ),
    ResearchStrategy(
        "strategy_sequence_context",
        "Model strictly-past sequence context",
        ("sequential",),
        "Test whether recent positive history adds signal beyond static identities.",
        "Sequence-aware ranking beats a matched static control within its runtime budget.",
    ),
    ResearchStrategy(
        "strategy_candidate_conditioning",
        "Condition history on the candidate",
        ("sequential", "features"),
        "Use candidate-aware attention or metadata sequence interactions when aggregate sequence signal is weak.",
        "Candidate-conditioned history improves nDCG@5 without sacrificing GAUC.",
    ),
    ResearchStrategy(
        "strategy_hard_negatives",
        "Mine informative ranking negatives",
        ("ranking_objective",),
        "Replace easy random negatives with a bounded high-scoring same-user pool after a scorer exists.",
        "Grouped/listwise or BPR ranking improves against a matched sampler.",
    ),
    ResearchStrategy(
        "strategy_graph_collaboration",
        "Add collaborative graph propagation",
        ("model",),
        "Use LightGCN only as a diversity signal or blend component when content features leave residual error.",
        "Graph or graph/content blend improves validation ranking with acceptable runtime.",
    ),
    ResearchStrategy(
        "strategy_optimization_stability",
        "Refine a promising executable branch",
        ("optimization",),
        "Tune capacity, regularization, learning rate, batch size, or early stopping after a branch shows signal.",
        "A controlled optimization change improves or stabilizes a promising family.",
    ),
    ResearchStrategy(
        "strategy_replication",
        "Replicate provisional gains",
        ("replication",),
        "Use matched seeds before treating a small validation winner as reliable.",
        "The gain keeps its direction across independent seeds.",
    ),
    ResearchStrategy(
        "strategy_robust_ensemble",
        "Combine independently useful predictors",
        ("model",),
        "Allocate late budget to an ensemble only after individual members are validated.",
        "Averaging improves validation Primary without test-label feedback.",
    ),
    ResearchStrategy(
        "strategy_runtime_repair",
        "Repair an execution bottleneck",
        ("optimization", "sequential", "model"),
        "Respond to timeout or memory evidence with a cheaper controlled scout, not a quality conclusion.",
        "The candidate finishes and produces valid validation metrics within budget.",
    ),
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metrics(item: dict[str, Any]) -> dict[str, float]:
    raw = item.get("metrics")
    if not isinstance(raw, dict):
        return {}
    return {
        key: number
        for key in VALID_METRICS
        if (number := _finite(raw.get(key))) is not None
    }


def _evidence_id(item: dict[str, Any], position: int) -> str:
    supplied = item.get("evidence_id")
    if isinstance(supplied, str) and supplied:
        return supplied
    iteration = item.get("iteration")
    if isinstance(iteration, int):
        return f"iteration_{iteration:03d}"
    return f"historical_record_{position:03d}"


def _tokens(value: Any) -> set[str]:
    if isinstance(value, dict):
        value = " ".join(f"{key} {item}" for key, item in value.items())
    return set(re.findall(r"[a-z0-9_]+", str(value).lower()))


def _verdict(item: dict[str, Any]) -> str:
    critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
    verdict = critique.get("verdict")
    if verdict in {"promote", "noise", "reject", "failed"}:
        return str(verdict)
    return "failed" if item.get("status") not in {None, "success", "ok"} else "noise"


def _delta(item: dict[str, Any]) -> float | None:
    critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
    direct = _finite(item.get("delta_from_parent"))
    return direct if direct is not None else _finite(critique.get("delta"))


def _outcome(item: dict[str, Any]) -> str:
    explicit = item.get("search_outcome")
    if explicit == "execution_failure":
        return "execution_failure"
    if explicit == "global_best" and item.get("decision") == "KEEP" and _verdict(item) == "promote" and item.get("iteration") not in (None, 0):
        return "global_best"
    if explicit in {"valid_nonimproving", "branch_best"}:
        return "valid_nonimproving"
    if item.get("status") not in {None, "success", "ok"}:
        return "execution_failure"
    # KEEP means "best seen so far", not necessarily a meaningful success.
    # Historical noise must not be converted into positive strategy evidence.
    if (
        _verdict(item) == "promote"
        and item.get("decision") == "KEEP"
        and item.get("iteration") not in (None, 0)
    ):
        return "global_best"
    return "valid_nonimproving"


def research_phase(iteration: int, total_iterations: int) -> tuple[str, float, float]:
    """Return phase, progress, and the MLEvolve-style exploration probability."""
    # Iteration zero is the cached/reproduced root, not a research allocation.
    # Therefore the first actual candidate always starts in exploration, even
    # for a short smoke run with only one candidate slot.
    denominator = max(1, int(total_iterations) - 2)
    progress = min(1.0, max(0.0, float(iteration - 1) / denominator))
    if progress < 0.45:
        phase = "explore"
    elif progress < 0.80:
        phase = "focus"
    else:
        phase = "confirm"
    exploration_probability = max(0.20, 0.90 - 0.70 * progress)
    return phase, progress, exploration_probability


def diagnose_history(history: list[dict[str, Any]], epsilon: float = DEFAULT_EPSILON) -> list[dict[str, Any]]:
    """Produce deterministic, metric-grounded bottleneck diagnoses."""
    rows = [item for item in history if isinstance(item, dict)]
    diagnoses: list[dict[str, Any]] = []
    for item in reversed(rows):
        report = item.get("error_slices") if isinstance(item.get("error_slices"), dict) else {}
        worst = report.get("worst_slices") if isinstance(report.get("worst_slices"), list) else []
        if worst:
            slice_row = worst[0]
            diagnoses.append({
                "code": "validation_slice_bottleneck",
                "observation": (
                    f"Worst measured validation slice is {slice_row.get('dimension')}="
                    f"{slice_row.get('value')} with Primary={slice_row.get('primary')}."
                ),
                "recommended_branches": ["features", "sequential", "model", "ranking_objective"],
                "evidence_ids": [_evidence_id(item, rows.index(item))],
            })
            break
    recent = rows[-8:]
    timeouts = []
    timeout_models: list[str] = []
    for position, item in enumerate(rows):
        error = item.get("error") if isinstance(item.get("error"), dict) else {}
        message = f"{error.get('type', '')} {error.get('message', '')}".lower()
        if "timeout" in message or "exceeded" in message:
            timeouts.append(_evidence_id(item, position))
            config = item.get("config") if isinstance(item.get("config"), dict) else {}
            model = config.get("model")
            if isinstance(model, str):
                timeout_models.append(model)
    if timeouts:
        diagnoses.append({
            "code": "runtime_bottleneck",
            "observation": "At least one candidate exhausted its execution budget without valid metrics.",
            "recommended_branches": ["optimization", "model", "sequential"],
            "affected_models": list(dict.fromkeys(timeout_models)),
            "evidence_ids": timeouts[-2:],
        })

    tradeoffs = []
    for position, item in enumerate(rows):
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        deltas = critique.get("metric_deltas") if isinstance(critique.get("metric_deltas"), dict) else {}
        gauc, ndcg = _finite(deltas.get("GAUC")), _finite(deltas.get("nDCG@5"))
        if gauc is not None and ndcg is not None and gauc * ndcg < 0:
            tradeoffs.append(_evidence_id(item, position))
    if tradeoffs:
        diagnoses.append({
            "code": "metric_tradeoff",
            "observation": "GAUC and nDCG@5 moved in opposite directions in prior experiments.",
            "recommended_branches": ["ranking_objective", "features"],
            "evidence_ids": tradeoffs[-2:],
        })

    model_rows = [item for item in recent if "model" in (item.get("changes") or {})]
    if len(model_rows) >= 2 and not any((_delta(item) or 0.0) > epsilon for item in model_rows):
        diagnoses.append({
            "code": "capacity_saturation",
            "observation": "Recent model-family changes did not produce a meaningful validation gain.",
            "recommended_branches": ["features", "ranking_objective"],
            "avoid_branches": ["model"],
            "evidence_ids": [
                _evidence_id(item, rows.index(item)) for item in model_rows[-2:]
            ],
        })

    feature_rows = [item for item in recent if any(
        key not in {"model", "training_objective"}
        and key not in {"embedding_dim", "learning_rate", "epochs", "l2", "batch_size", "patience", "seed", "ensemble_size", "ensemble_seed_set", "negatives_per_positive", "negative_sampling_strategy"}
        for key in (item.get("changes") or {})
    )]
    if len(feature_rows) >= 2 and not any((_delta(item) or 0.0) > epsilon for item in feature_rows):
        diagnoses.append({
            "code": "weak_feature_increment",
            "observation": "Recent engineered features were redundant, sparse, or too weak to clear the noise threshold.",
            "recommended_branches": ["model", "ranking_objective"],
            "avoid_branches": ["features"],
            "evidence_ids": [
                _evidence_id(item, rows.index(item)) for item in feature_rows[-2:]
            ],
        })

    provisional = [item for item in rows if item.get("decision") == "KEEP" and (
        _delta(item) is not None and 0 < float(_delta(item)) <= epsilon
    )]
    if provisional:
        item = provisional[-1]
        diagnoses.append({
            "code": "provisional_gain",
            "observation": "The newest validation winner is smaller than epsilon and needs matched replication.",
            "recommended_branches": ["replication", "optimization", "features"],
            "evidence_ids": [_evidence_id(item, rows.index(item))],
        })

    if not diagnoses:
        diagnoses.append({
            "code": "cold_start",
            "observation": "No dominant bottleneck has enough evidence yet; test distinct mechanisms broadly.",
            "recommended_branches": ["features", "ranking_objective", "model", "sequential"],
            "evidence_ids": ["benchmark_reference"],
        })
    return diagnoses[:4]


def retrieve_experiences(
    history: list[dict[str, Any]], query: Any, *, success_limit: int = 2,
    failure_limit: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Small bounded retrieval over all experience; 50 rows do not need FAISS."""
    query_tokens = _tokens(query)
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for position, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        searchable = {
            "hypothesis": item.get("hypothesis"),
            "diagnosis": item.get("diagnosis"),
            "changes": item.get("changes"),
            "model": (item.get("config") or {}).get("model") if isinstance(item.get("config"), dict) else None,
            "critique": item.get("critique"),
            "error": item.get("error"),
        }
        tokens = _tokens(searchable)
        lexical = len(query_tokens & tokens) / max(1, len(query_tokens | tokens))
        recency = 0.05 * (position + 1) / max(1, len(history))
        delta = _delta(item) or 0.0
        signal = min(0.10, abs(delta) * 20.0)
        ranked.append((lexical + recency + signal, position, item))

    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for relevance, position, item in sorted(ranked, key=lambda row: (-row[0], -row[1])):
        target = positive if _outcome(item) == "global_best" else negative
        limit = success_limit if target is positive else failure_limit
        if len(target) >= limit:
            continue
        critique = item.get("critique") if isinstance(item.get("critique"), dict) else {}
        target.append({
            "evidence_id": _evidence_id(item, position),
            "outcome": _outcome(item),
            "changes": item.get("changes") if isinstance(item.get("changes"), dict) else {},
            "metrics": _metrics(item),
            "verdict": _verdict(item),
            "diagnosis": item.get("diagnosis") or critique.get("interpretation"),
            "lesson": critique.get("general_lesson") or critique.get("next_test"),
            "relevance": round(relevance, 6),
        })
        if len(positive) >= success_limit and len(negative) >= failure_limit:
            break
    return {"successes": positive, "failures": negative}


def _strategy_for(candidate: dict[str, Any], resolved: dict[str, Any]) -> ResearchStrategy:
    model = resolved.get("model")
    changes = candidate.get("changes") or {}
    branch = str(candidate.get("branch") or "optimization")
    if "hard_negative_pool_size" in changes:
        return next(item for item in STRATEGIES if item.strategy_id == "strategy_hard_negatives")
    if model in {"din", "sasrec_meta"}:
        return next(item for item in STRATEGIES if item.strategy_id == "strategy_candidate_conditioning")
    if model in {"lightgcn", "lightgcn_hybrid"}:
        return next(item for item in STRATEGIES if item.strategy_id == "strategy_graph_collaboration")
    if model == "custom" or candidate.get("candidate_kind") == "code_branch":
        return next(item for item in STRATEGIES if item.strategy_id == "strategy_interaction_capacity")
    if model == "multitask" and "model" in changes:
        return next(item for item in STRATEGIES if item.strategy_id == "strategy_interaction_capacity")
    if model in MODEL_SPECS and MODEL_SPECS[model].supports_sequence:
        return next(item for item in STRATEGIES if item.strategy_id == "strategy_sequence_context")
    if "seed" in changes:
        return next(item for item in STRATEGIES if item.strategy_id == "strategy_replication")
    if changes.get("model") == "fm_ensemble" or "ensemble_size" in changes:
        return next(item for item in STRATEGIES if item.strategy_id == "strategy_robust_ensemble")
    return next(
        (item for item in STRATEGIES if branch in item.branches),
        next(item for item in STRATEGIES if item.strategy_id == "strategy_optimization_stability"),
    )


def _config_value(config: dict[str, Any], field: str) -> Any:
    spec = OPERATORS[field]
    return config.get(field) if spec.target == "root" else config[spec.target].get(field)


def _runtime_estimate(
    candidate: dict[str, Any], resolved: dict[str, Any], history: list[dict[str, Any]],
) -> tuple[float, bool]:
    model = resolved.get("model")
    runtimes: list[float] = []
    timeout = False
    for item in history:
        config = item.get("config") if isinstance(item, dict) else None
        if not isinstance(config, dict) or config.get("model") != model:
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        runtime = _finite(item.get("execution_seconds"))
        if runtime is None:
            runtime = _finite(metrics.get("runtime_seconds"))
        if runtime is not None:
            runtimes.append(runtime)
        error = item.get("error") if isinstance(item.get("error"), dict) else {}
        message = f"{error.get('type', '')} {error.get('message', '')}".lower()
        timeout = timeout or "timeout" in message or "exceeded" in message
    estimate = statistics.median(runtimes) if runtimes else COST_SECONDS.get(candidate.get("cost"), 240.0)
    if timeout:
        estimate = max(estimate, COST_SECONDS["high"])
    return float(estimate), timeout


def rank_candidates(
    parent: dict[str, Any],
    history: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    diagnoses: list[dict[str, Any]],
    phase: str,
    remaining_seconds: float | None,
) -> list[dict[str, Any]]:
    epsilon = DEFAULT_EPSILON
    diagnosis_codes = {str(item.get("code")) for item in diagnoses}
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            resolved = apply_changes(parent, candidate["changes"])
        except (KeyError, TypeError, ValueError):
            continue
        strategy = _strategy_for(candidate, resolved)
        branch = str(candidate["branch"])
        applicable_diagnoses = [
            diagnosis for diagnosis in diagnoses
            if branch in diagnosis.get("recommended_branches", [])
            and (
                not diagnosis.get("affected_models")
                or resolved.get("model") in diagnosis.get("affected_models", [])
            )
        ]
        contradicted = any(
            branch in diagnosis.get("avoid_branches", []) for diagnosis in diagnoses
        )
        matching: list[float] = []
        exact_matching: list[float] = []
        outcome_rewards: list[float] = []
        mechanism_attempts = 0
        branch_attempts = 0
        for item in history:
            config = item.get("config") if isinstance(item, dict) else None
            changes = item.get("changes") if isinstance(item, dict) and isinstance(item.get("changes"), dict) else {}
            same_model = isinstance(config, dict) and config.get("model") == resolved.get("model")
            previous_branch = branch_for_changes(changes)
            same_branch = same_model and previous_branch == branch
            same_mechanism = same_model and all(
                _config_value(config, field) == _config_value(resolved, field)
                for field in candidate["changes"]
            )
            if (
                candidate.get("evolution_recipe")
                and item.get("evolution_recipe") == candidate.get("evolution_recipe")
            ):
                same_mechanism = True
            if same_mechanism:
                mechanism_attempts += 1
            if same_branch:
                branch_attempts += 1
            if not same_mechanism and not same_branch:
                continue
            delta = _delta(item)
            if delta is not None:
                scaled = max(-1.0, min(1.0, delta / epsilon))
                matching.append(scaled)
                if same_mechanism:
                    exact_matching.append(scaled)
            outcome_rewards.append({
                "execution_failure": -1.0,
                "valid_nonimproving": 0.0,
                "branch_best": 0.0,
                "global_best": 1.0,
            }[_outcome(item)])
        empirical = statistics.mean(matching) if matching else 0.0
        experience = statistics.mean(outcome_rewards) if outcome_rewards else 0.0
        # Novelty is local to the model and changed mechanism. Failed FM tag
        # experiments should not make a first DCNv2 tag experiment look exhausted.
        novelty = 1.0 / (1.0 + mechanism_attempts + 0.25 * branch_attempts)
        diagnostic_fit = 1.0 if applicable_diagnoses else 0.25
        runtime_seconds, timeout_risk = _runtime_estimate(candidate, resolved, history)
        # Cost utility is absolute (10 minutes is expensive), while the Controller
        # separately enforces whether the estimate fits the remaining budget.
        cost_penalty = min(1.0, runtime_seconds / 600.0)
        confirmation = 1.0 if strategy.strategy_id in {
            "strategy_replication", "strategy_robust_ensemble"
        } else 0.0
        if phase == "explore":
            utility = 0.42 * novelty + 0.30 * diagnostic_fit + 0.18 * empirical + 0.10 * experience
        elif phase == "focus":
            utility = 0.20 * novelty + 0.30 * diagnostic_fit + 0.32 * empirical + 0.18 * experience
        else:
            utility = 0.08 * novelty + 0.20 * diagnostic_fit + 0.25 * empirical + 0.17 * experience + 0.30 * confirmation
        utility -= 0.20 * cost_penalty
        if contradicted:
            utility -= 0.25
        if mechanism_attempts and exact_matching and max(exact_matching) <= 0:
            utility -= 0.35
        if strategy.strategy_id == "strategy_replication":
            if "provisional_gain" not in diagnosis_codes:
                utility -= 0.60
            elif phase == "explore":
                utility -= 0.15
            elif phase == "focus":
                utility += 0.10
        if strategy.strategy_id == "strategy_robust_ensemble" and phase == "explore":
            utility -= 0.20
        if candidate.get("evolution_recipe"):
            # Coherent, pre-validated mechanism bundles get a small exploration
            # prior so the shortlist is not monopolized by dozens of cheap
            # one-flag toggles. Measured failures still dominate this prior.
            utility += 0.15
        # When component metrics disagree, give the explicitly top-k-aware
        # control a small exploration prior. This does not bypass Primary
        # promotion; it only prevents GAUC-optimized candidates from crowding
        # every nDCG experiment out of the shortlist.
        if candidate.get("changes", {}).get("validation_metric") == "nDCG@5":
            utility += 0.15 if "metric_tradeoff" in diagnosis_codes else 0.04
        # This transition is the one benchmark-specific cold-start prior already
        # established by the repository's reproduced baseline experiments.
        if (
            parent.get("model") == "fm"
            and parent.get("training_objective") == "bce"
            and candidate.get("changes") == {"training_objective": "bpr"}
        ):
            utility += 0.60
        if timeout_risk:
            utility -= 1.0
        evidence_ids = []
        for diagnosis in applicable_diagnoses:
            evidence_ids.extend(diagnosis.get("evidence_ids", []))
        ranked.append({
            **candidate,
            "strategy_id": strategy.strategy_id,
            "predicted_utility": round(utility, 6),
            "estimated_runtime_seconds": round(runtime_seconds, 3),
            "timeout_risk": timeout_risk,
            "evidence_ids": list(dict.fromkeys(evidence_ids or ["benchmark_reference"]))[:3],
            "hypothesis": f"{strategy.title}: test whether {candidate['changes']} improves validation ranking.",
            "diagnosis": next(
                (item["observation"] for item in applicable_diagnoses),
                diagnoses[0]["observation"],
            ),
            "reason": strategy.purpose,
            "risk": (
                "Prior runs in this model family timed out; treat this only as an execution repair."
                if timeout_risk else
                "The mechanism may be redundant or add cost without incremental ranking signal."
            ),
        })
    return sorted(
        ranked,
        key=lambda item: (-item["predicted_utility"], item["estimated_runtime_seconds"], item["candidate_id"]),
    )


def _stagnation(history: list[dict[str, Any]], parent_id: str | None) -> tuple[int, int]:
    current = [item for item in history if isinstance(item, dict) and not item.get("historical")]
    global_streak = 0
    for item in reversed(current):
        if item.get("iteration") in (None, 0):
            continue
        if item.get("decision") == "KEEP":
            break
        global_streak += 1
    branch_streak = 0
    for item in reversed(current):
        if item.get("parent_id") != parent_id:
            continue
        if item.get("decision") == "KEEP":
            break
        branch_streak += 1
    return branch_streak, global_streak


def build_research_context(
    parent: dict[str, Any],
    history: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    iteration: int,
    total_iterations: int,
    parent_id: str | None,
    remaining_seconds: float | None,
    shortlist_size: int = 5,
) -> dict[str, Any]:
    phase, progress, exploration_probability = research_phase(iteration, total_iterations)
    diagnoses = diagnose_history(history)
    ranked = rank_candidates(parent, history, candidates, diagnoses, phase, remaining_seconds)
    shortlist_limit = max(1, int(shortlist_size))
    shortlist: list[dict[str, Any]] = []
    if ranked:
        shortlist.append(ranked[0])
    for item in ranked:
        if item.get("evolution_recipe") and item not in shortlist:
            shortlist.append(item)
        if len([row for row in shortlist if row.get("evolution_recipe")]) >= 2:
            break
    for item in ranked:
        if item not in shortlist:
            shortlist.append(item)
        if len(shortlist) >= shortlist_limit:
            break
    shortlist = shortlist[:shortlist_limit]
    generated = next(
        (item for item in ranked if item.get("candidate_kind") == "code_branch"),
        None,
    )
    if generated is not None and generated not in shortlist:
        # The open-ended escape hatch must be visible to an LLM even when
        # cheaper registry actions occupy the utility top-k.
        if shortlist:
            shortlist[-1] = generated
        else:
            shortlist = [generated]
    branch_streak, global_streak = _stagnation(history, parent_id)
    if global_streak >= 6:
        expansion_mode = "aggregation"
    elif branch_streak >= 3 and progress >= 0.5:
        expansion_mode = "cross_branch"
    elif branch_streak >= 3:
        expansion_mode = "intra_branch"
    else:
        expansion_mode = "primary"
    query = {
        "parent_model": parent.get("model"),
        "parent_objective": parent.get("training_objective"),
        "diagnoses": diagnoses,
        "candidates": [item.get("changes") for item in shortlist],
    }
    experiences = retrieve_experiences(history, query)
    reference_ids = [
        item["evidence_id"]
        for group in ("successes", "failures")
        for item in experiences[group]
    ]
    strategy_ids = list(dict.fromkeys(
        item["strategy_id"] for item in shortlist
    ))
    strategies = [
        item.as_dict() for item in STRATEGIES if item.strategy_id in strategy_ids
    ]
    return {
        "phase": phase,
        "progress": round(progress, 6),
        "exploration_probability": round(exploration_probability, 6),
        "expansion_mode": expansion_mode,
        "branch_stagnation": branch_streak,
        "global_stagnation": global_streak,
        "diagnoses": diagnoses,
        "retrieved_strategies": strategies,
        "retrieved_experiences": experiences,
        "reference_ids": reference_ids,
        "ranked_candidates": shortlist,
    }
