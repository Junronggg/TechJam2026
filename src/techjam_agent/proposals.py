from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .config import ALLOWED_VALUES, FEATURE_KEYS, MODELS, OBJECTIVES, apply_changes, experiment_key
from .memory import PLANNER_RECENT_HISTORY, build_memory_summary, collect_tried_keys
from .tree import branch_name, node_id_for, select_parent

# One experiment may change 1-3 allow-listed fields. A single field is preferred,
# but model switches such as FM+BPR -> LightGBM+BCE must set model and
# training_objective together (existing DeterministicResearcher behavior).
MAX_CHANGE_FIELDS = 3
ALLOWED_CHANGE_KEYS = set(ALLOWED_VALUES) | set(FEATURE_KEYS) | {"model", "training_objective"}
PROPOSAL_RESPONSE_KEYS = frozenset({"hypothesis", "reason", "changes"})
ZERO_TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
OFFICIAL_BASELINE_PRIMARY = 0.6016
CONVERGENCE_EPSILON = 0.002
HTTP_TIMEOUT_SECONDS = 60
MAX_LLM_ATTEMPTS = 3
RECENT_HISTORY = 20
VALIDATION_METRIC_KEYS = ("GAUC", "nDCG@5", "primary")
BLOCKED_EVIDENCE_KEYS = frozenset({
    "test",
    "test_metrics",
    "final_test_metrics",
    "test_primary",
    "test_gauc",
    "test_ndcg",
})
CHANGE_RULE = (
    "Propose exactly one legal experiment. Change between 1 and "
    f"{MAX_CHANGE_FIELDS} allow-listed fields. Prefer a single-field change; "
    "use two or three fields only when they are one atomic switch "
    "(for example model plus training_objective when moving FM+BPR to LightGBM+BCE). "
    "Do not repeat a configuration already present in history. "
    "Your changes are applied to expansion_parent, which may differ from global_best. "
    "Do not supply lineage fields such as parent_id; the Controller owns lineage. "
    "Use validation Primary only; never use or request test metrics."
)
REPAIR_INSTRUCTION = (
    "Your previous reply was not valid. Reply with one JSON object and exactly these "
    "keys: hypothesis (non-empty string), reason (non-empty string), changes "
    "(non-empty object of allow-listed fields only). No markdown, no extra keys."
)


@dataclass(frozen=True)
class Proposal:
    hypothesis: str
    reason: str
    changes: dict[str, Any]
    source: str
    token_usage: dict[str, int] | None = None
    llm_attempts: int = 0

    @classmethod
    def parse(cls, value: dict[str, Any], source: str) -> "Proposal":
        if not isinstance(value, dict):
            raise ValueError("proposal response must be a JSON object")
        unknown = set(value) - PROPOSAL_RESPONSE_KEYS
        if unknown:
            raise ValueError(f"unsupported proposal fields: {sorted(unknown)}")
        if not isinstance(value.get("hypothesis"), str) or not value["hypothesis"].strip():
            raise ValueError("proposal requires a non-empty hypothesis")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise ValueError("proposal requires a non-empty reason")
        changes = value.get("changes")
        if not isinstance(changes, dict) or not 1 <= len(changes) <= MAX_CHANGE_FIELDS:
            raise ValueError("proposal must contain one atomic action (at most three config fields)")
        illegal = set(changes) - ALLOWED_CHANGE_KEYS
        if illegal:
            raise ValueError(f"unsupported proposal keys: {sorted(illegal)}")
        return cls(value["hypothesis"].strip(), value["reason"].strip(), changes, source)

    def as_dict(self) -> dict[str, Any]:
        usage = ZERO_TOKEN_USAGE if self.token_usage is None else self.token_usage
        return {
            "hypothesis": self.hypothesis,
            "reason": self.reason,
            "changes": self.changes,
            "source": self.source,
            "token_usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            "llm_attempts": int(self.llm_attempts),
        }


def empty_token_usage() -> dict[str, int]:
    return dict(ZERO_TOKEN_USAGE)


def extract_token_usage(payload: dict[str, Any] | None) -> dict[str, int]:
    """Read OpenAI-compatible usage; missing fields become zero and never crash."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return empty_token_usage()
    prompt = _nonneg_int(usage.get("prompt_tokens"))
    completion = _nonneg_int(usage.get("completion_tokens"))
    total = _nonneg_int(usage.get("total_tokens"))
    if total == 0:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def add_token_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": left["prompt_tokens"] + right["prompt_tokens"],
        "completion_tokens": left["completion_tokens"] + right["completion_tokens"],
        "total_tokens": left["total_tokens"] + right["total_tokens"],
    }


def _nonneg_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def validation_metrics_only(metrics: Any) -> dict[str, Any] | None:
    """Keep validation ranking metrics only. Drop test split numbers if present."""
    if not isinstance(metrics, dict):
        return None
    cleaned = {key: metrics[key] for key in VALIDATION_METRIC_KEYS if key in metrics}
    return cleaned or None


def sanitize_prior_evidence(value: Any) -> Any:
    """Recursively remove test-split fields before evidence enters an LLM prompt."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in BLOCKED_EVIDENCE_KEYS or normalized.startswith("test_"):
                continue
            cleaned[str(key)] = sanitize_prior_evidence(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_prior_evidence(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_prior_evidence(item) for item in value]
    return value


def compact_history_for_planner(history: list[dict[str, Any]], recent: int = RECENT_HISTORY) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in history[-recent:]:
        compact.append({
            "iteration": item.get("iteration"),
            "hypothesis": item.get("hypothesis"),
            "critique": item.get("critique"),
            "changes": item.get("changes"),
            "decision": item.get("decision"),
            "metrics": validation_metrics_only(item.get("metrics")),
        })
    return compact


def remaining_search_space(best_config: dict[str, Any]) -> dict[str, Any]:
    hp = best_config.get("hyperparameters") or {}
    features = best_config.get("features") or {}
    return {
        "models": [model for model in MODELS if model != best_config.get("model")],
        "training_objectives": [obj for obj in OBJECTIVES if obj != best_config.get("training_objective")],
        "features_still_off": [key for key in FEATURE_KEYS if not features.get(key)],
        "hyperparameter_alternatives": {
            key: [value for value in allowed if value != hp.get(key)]
            for key, allowed in ALLOWED_VALUES.items()
        },
    }


def latest_best_validation_metrics(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(history):
        if item.get("decision") == "KEEP":
            cleaned = validation_metrics_only(item.get("metrics"))
            if cleaned is not None:
                return cleaned
    return None


def default_allowed_values() -> dict[str, Any]:
    return {
        "model": MODELS,
        "training_objective": OBJECTIVES,
        **ALLOWED_VALUES,
        **{key: (False, True) for key in FEATURE_KEYS},
        "max_change_fields": MAX_CHANGE_FIELDS,
        "allowed_change_keys": sorted(ALLOWED_CHANGE_KEYS),
    }


def expansion_parent_view(
    history: list[dict[str, Any]],
    global_best_config: dict[str, Any],
) -> dict[str, Any]:
    """Describe the node being expanded, and whether it is the global best.

    The parent config is sent only when it differs from the global best, so the
    single-branch case costs no extra tokens.
    """
    parent = select_parent(history)
    if parent is None:
        return {"node_id": None, "iteration": None, "branch": None,
                "validation_primary": None, "same_as_global_best": True}
    view = {"node_id": parent.node_id, "iteration": parent.iteration,
            "branch": parent.branch, "validation_primary": parent.primary,
            "same_as_global_best": parent.config == global_best_config}
    if not view["same_as_global_best"]:
        view["config"] = parent.config
    return view


def expansion_parent_view_for_config(
    history: list[dict[str, Any]],
    parent_config: dict[str, Any],
    global_best_config: dict[str, Any],
) -> dict[str, Any]:
    """Build a lineage view for a tree-selected config, including weaker parents."""
    parent_key = experiment_key(parent_config)
    for item in reversed(history):
        config = item.get("config") if isinstance(item, dict) else None
        iteration = item.get("iteration") if isinstance(item, dict) else None
        metrics = item.get("metrics") if isinstance(item, dict) else None
        if (not isinstance(config, dict) or not isinstance(iteration, int) or
                experiment_key(config) != parent_key):
            continue
        primary = validation_metrics_only(metrics)
        changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
        view = {
            "node_id": node_id_for(iteration),
            "iteration": iteration,
            "branch": branch_name(changes),
            "validation_primary": None if primary is None else primary.get("primary"),
            "same_as_global_best": parent_config == global_best_config,
        }
        if not view["same_as_global_best"]:
            view["config"] = parent_config
        return view
    return {
        "node_id": None,
        "iteration": None,
        "branch": None,
        "validation_primary": None,
        "same_as_global_best": parent_config == global_best_config,
        **({} if parent_config == global_best_config else {"config": parent_config}),
    }


def build_planner_prompt(
    best_config: dict[str, Any],
    history: list[dict[str, Any]],
    allowed_values: dict[str, Any] | None = None,
    *,
    official_baseline_primary: float = OFFICIAL_BASELINE_PRIMARY,
    epsilon: float = CONVERGENCE_EPSILON,
    recent: int = RECENT_HISTORY,
    expansion_parent: dict[str, Any] | None = None,
    prior_evidence: dict[str, Any] | None = None,
    expansion_config: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the planner context. Does not call an HTTP API."""
    return {
        "objective": (
            "Propose exactly one experiment that can improve validation Primary "
            "(mean of validation GAUC and nDCG@5)."
        ),
        "change_rule": CHANGE_RULE,
        "official_baseline_primary": official_baseline_primary,
        "epsilon": epsilon,
        "budget": budget or {
            "remaining_iterations": None,
            "remaining_seconds": None,
            "estimated_next_experiment_seconds": None,
        },
        "allowed_values": allowed_values if allowed_values is not None else default_allowed_values(),
        "remaining": remaining_search_space(expansion_config or best_config),
        "prior_evidence": sanitize_prior_evidence(prior_evidence or {}),
        "memory": build_memory_summary(history),
        "global_best": {
            "config": best_config,
            "validation_metrics": latest_best_validation_metrics(history),
        },
        "expansion_parent": (expansion_parent if expansion_parent is not None
                             else expansion_parent_view(history, best_config)),
        "history": compact_history_for_planner(history, recent=min(recent, PLANNER_RECENT_HISTORY)),
        "response_contract": {
            "type": "object",
            "required": ["hypothesis", "reason", "changes"],
            "additionalProperties": False,
            "properties": {
                "hypothesis": "non-empty string",
                "reason": "non-empty string",
                "changes": "non-empty object with 1-3 allow-listed keys only",
            },
        },
    }


class DeterministicResearcher:
    """Safe offline policy; also provides a fallback when an LLM is unavailable."""

    def propose(self, best: dict[str, Any], history: list[dict[str, Any]]) -> Proposal:
        tried = set(collect_tried_keys(history))
        hp = best["hyperparameters"]
        if best["model"] == "fm" and best["training_objective"] == "bce":
            bpr_switch = {"training_objective": "bpr", "learning_rate": 0.0003}
            candidate = apply_changes(best, bpr_switch)
            if experiment_key(candidate) not in tried:
                return Proposal(
                    "Reproduce the best FM+BPR configuration with learning_rate=0.0003.",
                    "Earlier controlled experiments found BPR at 0.0003 outperformed BCE and higher BPR learning rates.",
                    bpr_switch, "deterministic", empty_token_usage(),
                )
        if best["model"] == "fm" and best["training_objective"] == "bpr":
            for value in (0.4, 0.3, 0.5):
                changes = {
                    "model": "ensemble",
                    "training_objective": "hybrid",
                    "ensemble_deepfm_weight": value,
                }
                candidate = apply_changes(best, changes)
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Blend FM+BPR with DeepFM+BCE using DeepFM weight {value}.",
                        "Validation analysis found complementary ranking errors after within-user normalization.",
                        changes, "deterministic", empty_token_usage(),
                    )
            for value in (0.75, 0.5, 0.25):
                changes = {"training_objective": "hybrid", "hybrid_bpr_weight": value}
                candidate = apply_changes(best, changes)
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test FM hybrid BCE+BPR with BPR weight {value}.",
                        "Pointwise supervision may regularize the ranking objective while BPR remains dominant.",
                        changes, "deterministic", empty_token_usage(),
                    )
        if best["model"] == "ensemble":
            changes = {
                "model": "multitask_deepfm",
                "training_objective": "bce",
                "learning_rate": 0.001,
            }
            candidate = apply_changes(best, changes)
            if experiment_key(candidate) not in tried:
                return Proposal(
                    "Train DeepFM jointly on long_view and like.",
                    "Like-only auxiliary supervision improved all three rolling folds without becoming an inference feature.",
                    changes, "deterministic", empty_token_usage(),
                )
            dcn_changes = {
                "model": "dcnv2",
                "training_objective": "bce",
                "learning_rate": 0.001,
            }
            dcn_candidate = apply_changes(best, dcn_changes)
            if experiment_key(dcn_candidate) not in tried:
                return Proposal(
                    "Test low-rank DCNv2 with two explicit cross layers.",
                    "DCNv2 beat DeepFM in all three rolling folds, while keeping the same base fields and BCE objective.",
                    dcn_changes, "deterministic", empty_token_usage(),
                )
            for value in (0.4, 0.3, 0.5):
                if value == hp["ensemble_deepfm_weight"]:
                    continue
                candidate = apply_changes(best, {"ensemble_deepfm_weight": value})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test ensemble DeepFM weight {value}.",
                        "Blend weight controls the balance between pairwise FM and pointwise DeepFM.",
                        {"ensemble_deepfm_weight": value}, "deterministic", empty_token_usage(),
                    )
            deepfm_candidates = (
                (
                    {"model": "deepfm", "training_objective": "bce", "learning_rate": 0.001},
                    "Test DeepFM with pointwise BCE on the same base fields.",
                    "This isolates whether a nonlinear interaction tower improves over FM.",
                ),
                (
                    {"model": "deepfm", "training_objective": "bce", "learning_rate": 0.0005},
                    "Test DeepFM+BCE with learning_rate=0.0005.",
                    "The first DeepFM result was competitive, so a lower step size may improve its peak.",
                ),
                (
                    {"model": "deepfm", "training_objective": "bce", "learning_rate": 0.002},
                    "Test DeepFM+BCE with learning_rate=0.002.",
                    "A higher step size checks whether the nonlinear tower benefits from faster fitting.",
                ),
            )
            for changes, hypothesis, reason in deepfm_candidates:
                candidate = apply_changes(best, changes)
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        hypothesis, reason, changes, "deterministic", empty_token_usage()
                    )
        if best["model"] == "deepfm":
            if best["training_objective"] == "bce":
                changes = {"training_objective": "bpr", "learning_rate": 0.0003}
                candidate = apply_changes(best, changes)
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        "Test DeepFM with the ranking-aligned BPR objective.",
                        "This separates the objective effect while retaining nonlinear interactions.",
                        changes, "deterministic", empty_token_usage(),
                    )
            for value in (64, 16):
                if value == hp["deepfm_hidden_dim"]:
                    continue
                candidate = apply_changes(best, {"deepfm_hidden_dim": value})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test DeepFM hidden dimension {value}.",
                        "Hidden width controls the capacity of nonlinear feature interactions.",
                        {"deepfm_hidden_dim": value}, "deterministic", empty_token_usage(),
                    )
            if hp["learning_rate"] == 0.0003:
                for value in (1, 2, 3, 4):
                    if value == hp["seed"]:
                        continue
                    candidate = apply_changes(best, {"seed": value})
                    if experiment_key(candidate) not in tried:
                        return Proposal(
                            f"Repeat the best FM+BPR configuration with seed={value}.",
                            "A multi-seed check tests whether the observed ranking gain is stable.",
                            {"seed": value}, "deterministic", empty_token_usage(),
                        )
            for value in (0.0005, 0.0003):
                if value == hp["learning_rate"]:
                    continue
                candidate = apply_changes(best, {"learning_rate": value})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test BPR learning_rate={value} with one negative per positive.",
                        "A lower step size may preserve BPR ranking gains and reduce late-epoch degradation.",
                        {"learning_rate": value}, "deterministic", empty_token_usage(),
                    )
            for value in (2, 4):
                if value == hp["pairs_per_positive"]:
                    continue
                candidate = apply_changes(best, {"pairs_per_positive": value})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test {value} same-user negatives per positive for BPR.",
                        "More pairwise comparisons may improve ranking supervision without changing the model or features.",
                        {"pairs_per_positive": value}, "deterministic", empty_token_usage(),
                    )
        if best["model"] == "fm":
            model_change = ({"model": "lightgbm", "training_objective": "bce"}
                            if best["training_objective"] == "bpr" else {"model": "lightgbm"})
            candidate = apply_changes(best, model_change)
            if experiment_key(candidate) not in tried:
                return Proposal(
                    "Test whether LightGBM outperforms pointwise FM on the original fields.",
                    "This isolates model choice before adding continuous history statistics.",
                    model_change, "deterministic", empty_token_usage(),
                )
            user_tab_changes = {**model_change, "user_tab_long_view_rate": True}
            user_tab = apply_changes(best, user_tab_changes)
            if experiment_key(user_tab) not in tried:
                return Proposal(
                    "Test whether smoothed user-by-tab long-view preference improves ranking.",
                    "Category-specific preference may be useful even when global user propensity is not.",
                    user_tab_changes, "deterministic", empty_token_usage(),
                )
            stats_changes = {**model_change, "continuous_history_stats": True}
            with_stats = apply_changes(best, stats_changes)
            if experiment_key(with_stats) not in tried:
                return Proposal(
                    "Test continuous train-only user/item rates and log-counts with LightGBM.",
                    "This follows the pure LightGBM control even when that branch was not the global best.",
                    stats_changes, "deterministic", empty_token_usage(),
                )
        if best["model"] == "lightgbm" and not best["features"]["continuous_history_stats"]:
            if not best["features"]["user_tab_long_view_rate"]:
                candidate = apply_changes(best, {"user_tab_long_view_rate": True})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        "Test whether smoothed user-by-tab long-view preference improves ranking.",
                        "Category-specific preference may be useful even when global user propensity is not.",
                        {"user_tab_long_view_rate": True}, "deterministic", empty_token_usage(),
                    )
            candidate = apply_changes(best, {"continuous_history_stats": True})
            if experiment_key(candidate) not in tried:
                return Proposal(
                    "Test continuous train-only user/item rates and log-counts with LightGBM.",
                    "This distinguishes weak statistics from an unsuitable categorical-bucket representation.",
                    {"continuous_history_stats": True}, "deterministic", empty_token_usage(),
                )
        for key in FEATURE_KEYS:
            if key in ("continuous_history_stats", "user_tab_long_view_rate"):
                continue
            if not best["features"][key]:
                candidate = apply_changes(best, {key: True})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test whether train-only {key} improves within-user ranking.",
                        "Target-rate history may expose behavioral propensity unavailable to the base fields.",
                        {key: True}, "deterministic", empty_token_usage(),
                    )
        order = ("learning_rate", "l2", "embedding_dim", "epochs", "batch_size", "patience")
        for key in order:
            for value in ALLOWED_VALUES[key]:
                if value == hp[key]:
                    continue
                candidate = apply_changes(best, {key: value})
                if experiment_key(candidate) not in tried:
                    return Proposal(
                        f"Test whether {key}={value} improves ranking quality.",
                        "A controlled one-variable experiment preserves attribution and reproducibility.",
                        {key: value}, "deterministic", empty_token_usage(),
                    )
        raise StopIteration("the configured FM experiment space is exhausted")


class OpenAICompatibleResearcher:
    """Small OpenAI-compatible JSON client using only the Python standard library."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        *,
        api_key: str | None = None,
        urlopen: Callable[..., Any] | None = None,
        timeout: float = HTTP_TIMEOUT_SECONDS,
        prior_evidence: dict[str, Any] | None = None,
        retry_backoff_seconds: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for --researcher llm")
        self._urlopen = urlopen or urllib.request.urlopen
        self.timeout = timeout
        self.prior_evidence = sanitize_prior_evidence(prior_evidence or {})
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._sleep = sleeper
        self.last_token_usage = empty_token_usage()
        self.last_attempts = 0
        self._run_context: dict[str, Any] = {}

    def set_run_context(self, context: dict[str, Any]) -> None:
        """Receive deterministic budget context from the Controller before planning."""
        self._run_context = {
            "remaining_iterations": context.get("remaining_iterations"),
            "remaining_seconds": context.get("remaining_seconds"),
            "estimated_next_experiment_seconds": context.get(
                "estimated_next_experiment_seconds"
            ),
        }

    def propose(self, best: dict[str, Any], history: list[dict[str, Any]]) -> Proposal:
        global_best_parent = select_parent(history)
        global_best_config = best if global_best_parent is None else global_best_parent.config
        parent_view = expansion_parent_view_for_config(history, best, global_best_config)
        prompt = build_planner_prompt(
            global_best_config,
            history,
            expansion_parent=parent_view,
            expansion_config=best,
            budget=self._run_context,
            prior_evidence=self.prior_evidence,
        )
        tried = set(collect_tried_keys(history))
        accumulated = empty_token_usage()
        self.last_token_usage = empty_token_usage()
        self.last_attempts = 0
        last_error = "unknown error"
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            self.last_attempts = attempt
            try:
                payload = self._chat(prompt, repair=(attempt > 1))
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = type(exc).__name__
                if attempt >= MAX_LLM_ATTEMPTS:
                    raise RuntimeError(
                        f"LLM proposal failed after {MAX_LLM_ATTEMPTS} attempts: {last_error}"
                    ) from exc
                self._backoff(attempt)
                continue
            except ValueError as exc:
                last_error = type(exc).__name__
                if attempt >= MAX_LLM_ATTEMPTS:
                    raise RuntimeError(
                        f"LLM proposal failed after {MAX_LLM_ATTEMPTS} attempts: {last_error}"
                    ) from exc
                continue
            accumulated = add_token_usage(accumulated, extract_token_usage(payload))
            self.last_token_usage = accumulated
            try:
                raw = _message_json(payload)
                proposal = Proposal.parse(raw, "llm")
                candidate = apply_changes(best, proposal.changes)
                if experiment_key(candidate) in tried:
                    raise ValueError("LLM repeated a previous experiment")
            except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = type(exc).__name__
                if attempt >= MAX_LLM_ATTEMPTS:
                    raise RuntimeError(
                        f"LLM proposal failed after {MAX_LLM_ATTEMPTS} attempts: {last_error}"
                    ) from exc
                continue
            return Proposal(
                proposal.hypothesis,
                proposal.reason,
                proposal.changes,
                "llm",
                dict(accumulated),
                attempt,
            )
        raise RuntimeError(f"LLM proposal failed after {MAX_LLM_ATTEMPTS} attempts: {last_error}")

    def _backoff(self, attempt: int) -> None:
        if self.retry_backoff_seconds:
            self._sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))

    def _chat(self, prompt: dict[str, Any], *, repair: bool) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cautious autonomous ML researcher. Treat prior_evidence as "
                    "authoritative experimental memory. Do not repeat rejected mechanisms "
                    "unless the proposed legal action tests a materially different hypothesis. "
                    "Prefer evidence from multiple rolling folds over a single split. Propose "
                    "only values explicitly listed in allowed_values; roadmap items are not "
                    "currently executable actions."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        if repair:
            messages.append({"role": "user", "content": REPAIR_INSTRUCTION})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with self._urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM HTTP body was not JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("LLM HTTP body must be a JSON object")
        return payload


def _message_json(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response content must be a JSON object")
    return parsed
