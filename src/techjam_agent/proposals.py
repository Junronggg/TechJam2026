from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .config import ALLOWED_VALUES, FEATURE_KEYS, MODELS, OBJECTIVES, apply_changes, experiment_key
from .memory import PLANNER_RECENT_HISTORY, build_memory_summary, collect_tried_keys
from .tree import select_parent

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


def build_planner_prompt(
    best_config: dict[str, Any],
    history: list[dict[str, Any]],
    allowed_values: dict[str, Any] | None = None,
    *,
    official_baseline_primary: float = OFFICIAL_BASELINE_PRIMARY,
    epsilon: float = CONVERGENCE_EPSILON,
    recent: int = RECENT_HISTORY,
    expansion_parent: dict[str, Any] | None = None,
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
        "allowed_values": allowed_values if allowed_values is not None else default_allowed_values(),
        "remaining": remaining_search_space(best_config),
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
            candidate = apply_changes(best, {"training_objective": "bpr"})
            if experiment_key(candidate) not in tried:
                return Proposal(
                    "Align FM training with within-user ranking by replacing BCE with pairwise BPR.",
                    "GAUC and nDCG reward positive items ranking above negatives, not calibrated classification.",
                    {"training_objective": "bpr"}, "deterministic", empty_token_usage(),
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
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for --researcher llm")
        self._urlopen = urlopen or urllib.request.urlopen
        self.timeout = timeout
        self.last_token_usage = empty_token_usage()

    def propose(self, best: dict[str, Any], history: list[dict[str, Any]]) -> Proposal:
        prompt = build_planner_prompt(best, history)
        tried = set(collect_tried_keys(history))
        accumulated = empty_token_usage()
        last_error = "unknown error"
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            try:
                payload = self._chat(prompt, repair=(attempt > 1))
            except (urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError("LLM proposal failed: HTTP error") from exc
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
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
                last_error = type(exc).__name__
                if attempt >= MAX_LLM_ATTEMPTS:
                    raise RuntimeError(
                        f"LLM proposal failed after {MAX_LLM_ATTEMPTS} attempts: {last_error}"
                    ) from exc
                continue
            candidate = apply_changes(best, proposal.changes)
            if experiment_key(candidate) in tried:
                raise ValueError("LLM repeated a previous experiment")
            return Proposal(
                proposal.hypothesis,
                proposal.reason,
                proposal.changes,
                "llm",
                dict(accumulated),
            )
        raise RuntimeError(f"LLM proposal failed after {MAX_LLM_ATTEMPTS} attempts: {last_error}")

    def _chat(self, prompt: dict[str, Any], *, repair: bool) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a cautious autonomous ML researcher."},
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
